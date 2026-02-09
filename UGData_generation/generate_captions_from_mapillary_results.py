import json
import time
import argparse
import os
from tqdm import tqdm
import pickle
from datetime import datetime
import pandas as pd
import geopandas as gpd
import traceback
import sys
import numpy as np
import networkx as nx
from shapely.geometry import Point
from geopy.distance import geodesic
import re
from typing import Optional, Tuple
# Import necessary functions from image_caption
from image_caption import (
    use_qwen_vl_for_single_image_captioning,
    use_yinli_for_single_image_captioning,
    extract_captions_from_enhanced_format,
    get_existing_captioned_images,
    SpatialContextQAGenerator
)

# Get base paths from environment variables
DATA_ROOT = os.getenv('URBANKG_DATA_ROOT', './data')
OUTPUT_ROOT = os.getenv('URBANKG_OUTPUT_ROOT', './output')


def load_checkpoint(checkpoint_file):
    """
    Load checkpoint file containing processed mapillary_ids.
    
    Args:
        checkpoint_file: Path to checkpoint JSONL file
        
    Returns:
        dict: Checkpoint data with processed mapillary_ids and stats
    """
    checkpoint_data = {
        'processed_mapillary_ids': set(),
        'stats': {
            'new_captions': 0,
            'failed': 0,
            'no_subgraph': 0
        }
    }
    
    if not os.path.exists(checkpoint_file):
        print(f"   ℹ️  No checkpoint file found, starting fresh")
        return checkpoint_data
    
    try:
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    data = json.loads(line)
                    mapillary_id = data.get('mapillary_id')
                    if mapillary_id:
                        checkpoint_data['processed_mapillary_ids'].add(str(mapillary_id))
                        
                        # Update stats
                        if data.get('status') == 'success':
                            checkpoint_data['stats']['new_captions'] += 1
                        elif data.get('status') == 'failed':
                            checkpoint_data['stats']['failed'] += 1
                        elif data.get('status') == 'no_subgraph':
                            checkpoint_data['stats']['no_subgraph'] += 1
                            
                except json.JSONDecodeError:
                    continue
        
        print(f"   ✅ Resuming from checkpoint: {len(checkpoint_data['processed_mapillary_ids'])} mapillary_ids already processed")
        print(f"      Previous stats: {checkpoint_data['stats']}")
        
    except Exception as e:
        print(f"   ⚠️  Error reading checkpoint file: {e}")
    
    return checkpoint_data


def save_checkpoint_entry(checkpoint_file, mapillary_id, status, error_msg=None):
    """
    Save a checkpoint entry for a processed mapillary_id.
    
    Args:
        checkpoint_file: Path to checkpoint JSONL file
        mapillary_id: Mapillary ID that was processed
        status: Status of processing ('success', 'failed', 'no_subgraph')
        error_msg: Optional error message
    """
    try:
        checkpoint_entry = {
            'mapillary_id': mapillary_id,
            'status': status,
            'timestamp': datetime.now().isoformat()
        }

        if error_msg:
            checkpoint_entry['error'] = error_msg

        # Ensure directory exists
        checkpoint_dir = os.path.dirname(checkpoint_file)
        if checkpoint_dir and not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir, exist_ok=True)

        # Safer append with flush + fsync to reduce loss on crash
        with open(checkpoint_file, 'a', encoding='utf-8') as f:
            json.dump(checkpoint_entry, f)
            f.write('\n')
            f.flush()
            try:
                os.fsync(f.fileno())
            except Exception:
                # fsync might not be available on some systems
                pass

    except Exception as e:
        print(f"   ⚠️  Error saving checkpoint: {e}")


def extract_mapillary_ids_from_training_data(training_data_file):
    """
    Extract mapillary_ids from training data by parsing image paths.
    
    Args:
        training_data_file: Path to training data JSONL file
        
    Returns:
        set: Set of mapillary_ids (as strings)
    """
    mapillary_ids = set()
    
    if not os.path.exists(training_data_file):
        print(f"   ⚠️  Training data file not found: {training_data_file}")
        return mapillary_ids
    
    try:
        with open(training_data_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    data = json.loads(line)
                    
                    # Extract images from various possible locations
                    images = []
                    if 'images' in data:
                        if isinstance(data['images'], list):
                            images.extend(data['images'])
                        elif isinstance(data['images'], str):
                            images.append(data['images'])
                    
                    if 'image_paths' in data:
                        if isinstance(data['image_paths'], list):
                            images.extend(data['image_paths'])
                        elif isinstance(data['image_paths'], str):
                            images.append(data['image_paths'])
                    
                    swift_format = data.get('swift_format', {})
                    if isinstance(swift_format, dict) and 'images' in swift_format:
                        if isinstance(swift_format['images'], list):
                            images.extend(swift_format['images'])
                        elif isinstance(swift_format['images'], str):
                            images.append(swift_format['images'])
                    
                    # Extract mapillary_id from each image path
                    for image_path in images:
                        if image_path:
                            # Extract filename (without extension) which is the mapillary_id
                            filename = os.path.basename(str(image_path))
                            mapillary_id = os.path.splitext(filename)[0]
                            if mapillary_id:
                                mapillary_ids.add(mapillary_id)
                                
                except json.JSONDecodeError as e:
                    print(f"   ⚠️ JSON decode error at line {line_num}: {e}")
                    continue
        
        print(f"   ✅ Extracted {len(mapillary_ids)} unique mapillary_ids from training data")
        
    except Exception as e:
        print(f"   ⚠️  Error reading training data: {e}")
        traceback.print_exc()
    
    return mapillary_ids


def create_subgraph_from_coords(nodes_gdf, edges_gdf, lat, lon, radius_m=500):
    """
    Create a subgraph around given coordinates using spatial indexing for efficiency.
    
    Args:
        nodes_gdf: GeoDataFrame with all nodes
        edges_gdf: GeoDataFrame with all edges
        lat: Latitude of center point
        lon: Longitude of center point
        radius_m: Radius in meters to include nodes (default: 500)
        
    Returns:
        NetworkX graph with nodes and edges within radius
    """
    # Create center point
    center_point = Point(lon, lat)
    
    # Use spatial indexing for faster queries
    # Rough conversion: 1 degree lat ≈ 111km, 1 degree lon ≈ 111km * cos(lat)
    lat_buffer = radius_m / 111000.0
    lon_buffer = radius_m / (111000.0 * np.cos(np.radians(lat)))
    
    # Create bounding box for initial filtering
    bbox = (
        lon - lon_buffer,
        lat - lat_buffer,
        lon + lon_buffer,
        lat + lat_buffer
    )
    
    # Filter nodes in bounding box first (much faster)
    nodes_in_bbox = nodes_gdf.cx[bbox[0]:bbox[2], bbox[1]:bbox[3]]
    
    # Now calculate exact distances for nodes in bbox
    nodes_within_radius = []
    for idx, row in nodes_in_bbox.iterrows():
        if row.geometry is not None:
            try:
                # Get coordinates from geometry
                node_lon = row.geometry.x
                node_lat = row.geometry.y
                distance_m = geodesic((lat, lon), (node_lat, node_lon)).meters
                if distance_m <= radius_m:
                    nodes_within_radius.append(int(row['id']))
            except:
                continue
    
    if not nodes_within_radius:
        return nx.Graph()
    
    # Create subgraph with these nodes
    subgraph = nx.Graph()
    node_ids_set = set(nodes_within_radius)
    
    # Add nodes
    for node_id in node_ids_set:
        node_row = nodes_gdf[nodes_gdf['id'] == node_id]
        if not node_row.empty:
            node_data = node_row.iloc[0].to_dict()
            # Store geometry coordinates as attributes for distance calculations
            if 'geometry' in node_data and node_data['geometry'] is not None:
                geom = node_data['geometry']
                node_data['lat'] = geom.y
                node_data['lon'] = geom.x
                del node_data['geometry']
            subgraph.add_node(node_id, **node_data)
    
    # Add edges between nodes in subgraph
    edges_in_subgraph = edges_gdf[
        (edges_gdf['id1'].isin(node_ids_set)) | (edges_gdf['id2'].isin(node_ids_set))
    ]
    
    for idx, row in edges_in_subgraph.iterrows():
        id1 = int(row.get('id1'))
        id2 = int(row.get('id2'))
        
        if id1 in node_ids_set and id2 in node_ids_set:
            edge_data = row.to_dict()
            if 'geometry' in edge_data:
                del edge_data['geometry']
            # Add weight for pathfinding
            if 'crossing_distance_meters' in edge_data:
                edge_data['weight'] = edge_data['crossing_distance_meters']
            elif 'distance' in edge_data:
                edge_data['weight'] = edge_data['distance']
            else:
                edge_data['weight'] = 100  # Default weight
            subgraph.add_edge(id1, id2, **edge_data)
    
    return subgraph


def add_mapillary_node_to_subgraph(subgraph, mapillary_id, image_coords, nodes_gdf, edges_gdf, max_hops=2):
    """
    Add a Mapillary node to the subgraph and expand with neighbors if needed.
    
    Args:
        subgraph: Existing NetworkX subgraph
        mapillary_id: Mapillary node ID (as string, will be converted to int)
        image_coords: Tuple of (lat, lon) for image location
        nodes_gdf: GeoDataFrame containing all nodes
        edges_gdf: GeoDataFrame containing all edges
        max_hops: Number of hops to expand (default: 2)
        
    Returns:
        NetworkX graph with mapillary node and neighbors
    """
    # Convert mapillary_id to integer
    try:
        mapillary_id_int = int(mapillary_id)
    except (ValueError, TypeError):
        print(f"   ⚠️ Cannot convert Mapillary ID {mapillary_id} to integer")
        return subgraph
    
    lat, lon = image_coords
    
    # Check if mapillary node exists in nodes_gdf
    mapillary_row = nodes_gdf[nodes_gdf['id'] == mapillary_id_int]
    
    if mapillary_row.empty:
        # Create Mapillary node from image coordinates
        print(f"   🔧 Creating Mapillary node {mapillary_id_int} from coordinates ({lat:.6f}, {lon:.6f})")
        mapillary_node_data = {
            'id': mapillary_id_int,
            'type': 'mapillary',
            'category': 'mapillary',
            'lat': lat,
            'lon': lon
        }
    else:
        # Use existing node data
        mapillary_node_data = mapillary_row.iloc[0].to_dict()
        if 'geometry' in mapillary_node_data:
            del mapillary_node_data['geometry']
        print(f"   ✅ Found Mapillary node {mapillary_id_int} in nodes_gdf")
    
    # Add the Mapillary node to the graph if not already present
    if mapillary_id_int not in subgraph.nodes():
        subgraph.add_node(mapillary_id_int, **mapillary_node_data)
        print(f"   ✅ Added Mapillary node {mapillary_id_int} to subgraph")
    
    # Expand subgraph to include neighbors of mapillary node
    existing_nodes = set(subgraph.nodes())
    nodes_to_add = {mapillary_id_int}
    current_frontier = {mapillary_id_int}
    
    # Expand by hops
    for hop in range(max_hops):
        next_frontier = set()
        
        # Find all edges connected to current frontier
        connected_edges = edges_gdf[
            (edges_gdf['id1'].isin(current_frontier)) | 
            (edges_gdf['id2'].isin(current_frontier))
        ]
        
        # Add connected nodes
        for _, edge_row in connected_edges.iterrows():
            id1, id2 = int(edge_row['id1']), int(edge_row['id2'])
            
            if id1 in current_frontier and id2 not in nodes_to_add:
                next_frontier.add(id2)
                nodes_to_add.add(id2)
            if id2 in current_frontier and id1 not in nodes_to_add:
                next_frontier.add(id1)
                nodes_to_add.add(id1)
        
        current_frontier = next_frontier
        
        if not next_frontier:
            print(f"      Hop {hop + 1}: No more neighbors found")
            break
        else:
            print(f"      Hop {hop + 1}: Found {len(next_frontier)} new neighbors")
    
    # Add new nodes with attributes
    new_nodes = nodes_to_add - existing_nodes
    for node_id in new_nodes:
        if node_id == mapillary_id_int:
            continue  # Already added
        node_row = nodes_gdf[nodes_gdf['id'] == node_id]
        if not node_row.empty:
            node_data = node_row.iloc[0].to_dict()
            node_attrs = {}
            for col in node_data:
                if col != 'geometry' and pd.notna(node_data[col]):
                    node_attrs[col] = node_data[col]
            subgraph.add_node(node_id, **node_attrs)
        else:
            subgraph.add_node(node_id, id=node_id)
    
    # Add edges between nodes in expanded set
    edges_to_add = edges_gdf[
        (edges_gdf['id1'].isin(nodes_to_add)) & 
        (edges_gdf['id2'].isin(nodes_to_add))
    ]
    
    for _, edge_row in edges_to_add.iterrows():
        id1, id2 = int(edge_row['id1']), int(edge_row['id2'])
        if not subgraph.has_edge(id1, id2):
            edge_attrs = {}
            for col in edge_row.index:
                if col not in ['id1', 'id2', 'geometry'] and pd.notna(edge_row[col]):
                    edge_attrs[col] = edge_row[col]
            subgraph.add_edge(id1, id2, **edge_attrs)
    
    print(f"   ✅ Expanded subgraph: {subgraph.number_of_nodes()} nodes, {subgraph.number_of_edges()} edges")
    
    return subgraph


def extract_coordinates_from_geometry(geometry) -> Optional[Tuple[float, float]]:
    """
    Extract (lon, lat) coordinates from geometry object.
    
    Handles Point, LineString, Polygon, and other geometry types by using centroid.
    For LineString geometries, uses the centroid of the line.
    
    Args:
        geometry: Shapely geometry object
        
    Returns:
        Tuple of (lon, lat) or None if extraction fails
    """
    if geometry is None:
        return None
    
    try:
        # Check if geometry is empty or invalid
        if hasattr(geometry, 'is_empty') and geometry.is_empty:
            return None
        
        # Method 1: Try to get centroid (works for Point, LineString, Polygon, etc.)
        # This is the preferred method for LineString
        try:
            if hasattr(geometry, 'centroid'):
                centroid = geometry.centroid
                # Check if centroid is valid (not empty)
                if not (hasattr(centroid, 'is_empty') and centroid.is_empty):
                    lon = float(centroid.x)
                    lat = float(centroid.y)
                    # Validate coordinate ranges
                    if -180 <= lon <= 180 and -90 <= lat <= 90:
                        return (lon, lat)
        except (AttributeError, ValueError, TypeError):
            pass
        
        # Method 2: For LineString, get the first coordinate point
        try:
            if hasattr(geometry, 'geom_type'):
                geom_type = geometry.geom_type
                if geom_type == 'LineString':
                    coords = list(geometry.coords)
                    if len(coords) > 0:
                        # Use the first point of the LineString
                        lon = float(coords[0][0])
                        lat = float(coords[0][1])
                        # Validate coordinate ranges
                        if -180 <= lon <= 180 and -90 <= lat <= 90:
                            return (lon, lat)
        except (AttributeError, ValueError, TypeError, IndexError):
            pass
        
        # Method 3: For Point geometries, use x and y directly
        try:
            if hasattr(geometry, 'x') and hasattr(geometry, 'y'):
                lon = float(geometry.x)
                lat = float(geometry.y)
                # Validate coordinate ranges
                if -180 <= lon <= 180 and -90 <= lat <= 90:
                    return (lon, lat)
        except (AttributeError, ValueError, TypeError):
            pass
        
        # Method 4: Fallback - try to extract coordinates from geometry string
        try:
            geom_str = str(geometry)
            coords = re.findall(r'[-+]?\d+\.?\d*', geom_str)
            if len(coords) >= 2:
                lon = float(coords[0])
                lat = float(coords[1])
                # Validate coordinate ranges
                if -180 <= lon <= 180 and -90 <= lat <= 90:
                    return (lon, lat)
        except (ValueError, IndexError):
            pass
        
        return None
    except Exception as e:
        # Return None on any unexpected error
        return None


def create_node_text_and_coords(graph, nodes_gdf, place='singapore'):
    """
    Create 'node_text' and 'coords' attributes for all nodes in the graph.
    
    This function matches the format used in clean_graph_attributes.py to avoid post-processing.
    
    IMPORTANT: This function assumes node IDs are already integers (as required by graph_encoder_spatial.py).
    
    Args:
        graph: NetworkX graph object (node IDs should be integers)
        nodes_gdf: GeoDataFrame containing node information (IDs should be integers)
        place: Place name (default: 'singapore')
    """
    # Collect nodes to remove (nodes not found in nodes_gdf)
    nodes_to_remove = []
    
    for node_id in list(graph.nodes()):  # Use list() to avoid modification during iteration
        # Find node in GeoDataFrame (IDs should be integers as converted in main())
        node_row = nodes_gdf[nodes_gdf.id == node_id]
        
        if not node_row.empty:
            row = node_row.iloc[0]
            
            # Start with basic info
            node_name = 'mapillary image' if row.get('type') == 'mapillary' else row.get('name', 'unknown')
            node_text = f"Name: {node_name}, "
            node_text += f"ID: {row.get('id', 'unknown')}, "
            node_text += f"Category: {row.get('category', row.get('type', 'unknown'))}, "
            node_text += f"Address: {row.get('address', 'unknown')}, "
            
            # Add location-specific fields based on place
            if place == "newyork":
                if pd.notna(row.get('neighborhood')):
                    node_text += f"Neighborhood: {row.get('neighborhood')}, "
                if pd.notna(row.get('borough')):
                    node_text += f"Borough: {row.get('borough')}, "
                node_text += f"Country: USA, "
            else:  # default to singapore
                if pd.notna(row.get('planning_area')):
                    node_text += f"Planning area: {row.get('planning_area')}, "
                if pd.notna(row.get('district')):
                    node_text += f"District: {row.get('district')}, "
                node_text += f"Country: Singapore, "
            
            # Add rich attributes (optional, for completeness) - exclude country since it's already added
            rich_attrs = []
            for attr_name, col_name in [
                ('city', 'city'),
                ('street', 'street'),
                ('housenumber', 'housenumber'),
                ('postcode', 'postcode'),
                ('building use', 'building use'),
                ('historic district', 'historic district'),
                ('building', 'building'),
                ('historic', 'historic'),
                ('architect', 'architect')
            ]:
                value = row.get(col_name)
                if value is not None and pd.notna(value) and str(value).strip() != "" and str(value).lower() not in ['unknown', 'none', 'null']:
                    rich_attrs.append(f"{attr_name}={value}")
            
            # Extract coordinates from geometry
            coords = None
            # Access geometry from GeoDataFrame row (geometry is a column in GeoPandas)
            geometry = getattr(row, 'geometry', None)
            if geometry is not None and pd.notna(geometry):
                coords = extract_coordinates_from_geometry(geometry)
            
            # Add rich attributes and coordinates
            if rich_attrs:
                node_text += ", " + ", ".join(rich_attrs)
            
            if coords is not None:
                lon, lat = coords
                node_text += f", Coordinates: ({lon:.6f}, {lat:.6f})"
            else:
                node_text += ", Coordinates: unavailable"
            
            # Set node_text and coords attributes
            graph.nodes[node_id]['node_text'] = node_text
            if coords is not None:
                graph.nodes[node_id]['coords'] = coords
        else:
            # Check if node has lat/lon in attributes (for mapillary nodes created from coords)
            node_attrs = graph.nodes[node_id]
            if 'lat' in node_attrs and 'lon' in node_attrs:
                lat = node_attrs['lat']
                lon = node_attrs['lon']
                node_text = f"Name: mapillary image, ID: {node_id}, Category: mapillary, "
                node_text += f"Address: unknown, "
                if place == "newyork":
                    node_text += f"Country: USA, "
                else:
                    node_text += f"Country: Singapore, "
                node_text += f"Coordinates: ({lon:.6f}, {lat:.6f})"
                
                graph.nodes[node_id]['node_text'] = node_text
                graph.nodes[node_id]['coords'] = (lon, lat)
            else:
                # Node not found in GeoDataFrame - mark for removal
                nodes_to_remove.append(node_id)
    
    # Remove nodes that were not found in nodes_gdf
    if nodes_to_remove:
        graph.remove_nodes_from(nodes_to_remove)


def clean_node_attributes(graph):
    """
    Remove all node attributes except 'node_text' and 'coords'.
    
    This matches the format used in clean_graph_attributes.py.
    
    Args:
        graph: NetworkX graph object
    """
    for node_id in graph.nodes():
        node_attrs = dict(graph.nodes[node_id])
        node_text = node_attrs.get('node_text', '')
        coords = node_attrs.get('coords', None)
        
        # Clear all attributes
        graph.nodes[node_id].clear()
        
        # Add back only node_text and coords
        if node_text:
            graph.nodes[node_id]['node_text'] = node_text
        if coords is not None:
            graph.nodes[node_id]['coords'] = coords


def generate_subgraph_description(subgraph_data, nodes_gdf, edges_gdf):
    """
    Generate description from loaded subgraph using SpatialContextQAGenerator.
    
    Args:
        subgraph_data: Subgraph data (networkx graph or dict containing graph)
        nodes_gdf: GeoDataFrame containing node information
        edges_gdf: GeoDataFrame containing edge information
        
    Returns:
        str: Generated subgraph description
    """
    try:
        # Handle different subgraph data formats
        if isinstance(subgraph_data, dict):
            subgraph = subgraph_data.get('subgraph') or subgraph_data.get('subgraph_data')
            center_nodes = subgraph_data.get('center_nodes', [])
        else:
            subgraph = subgraph_data
            center_nodes = []
        
        if subgraph is None:
            print("   ⚠️ Could not extract subgraph from data")
            return ""
        
        # Print subgraph size
        num_nodes = subgraph.number_of_nodes()
        num_edges = subgraph.number_of_edges()
        print(f"   📊 Subgraph size: {num_nodes} nodes, {num_edges} edges, {len(center_nodes)} center nodes")
        
        # Create edges DataFrame from subgraph edges
        edges_data = []
        for u, v, data in subgraph.edges(data=True):
            edge_dict = {'id1': u, 'id2': v}
            edge_dict.update(data)
            edges_data.append(edge_dict)
        edges_gdf_subset = pd.DataFrame(edges_data) if edges_data else pd.DataFrame(columns=['id1', 'id2'])
        
        # Use SpatialContextQAGenerator to create description
        qa_generator = SpatialContextQAGenerator()
        
        subgraph_desc, node_categories = qa_generator._create_network_description(
            subgraph, center_nodes, nodes_gdf, edges_gdf_subset
        )
        
        print(f"   ✅ Generated subgraph description ({len(subgraph_desc)} characters)")
        return subgraph_desc
        
    except Exception as e:
        print(f"   ❌ Error generating subgraph description: {e}")
        traceback.print_exc()
        return ""


def create_enhanced_prompt_with_subgraph_description(subgraph_desc):
    """
    Create enhanced prompt using subgraph description.
    
    Args:
        subgraph_desc: Generated subgraph description
        
    Returns:
        str: Enhanced prompt for image captioning
    """
    prompt = f"""You are an advanced vision model tasked with generating detailed captions for urban street images within a comprehensive spatial network context.

# Comprehensive Spatial Context:

## Network Structure:
{subgraph_desc}

# Enhanced Caption Generation Instructions:

You will be provided with 1 street image from the spatial area described in the Network Structure above. The street image is within a spatial context represented with graph information.

For the image, generate a detailed caption and then summarize those image features based on the understanding of spatial context. 

1. **Detailed image caption**: Describe the unique visual features specific to this individual image, including:
   - Locations
   - Distinctive architectural features 
   - Notable businesses, signage, or landmarks
   - Vegetation and greenery, water features or other natural sensory elements
   - Visual indicators of acoustic environments (traffic, construction, natural settings, quiet spaces)
   - Elements that emit or suggest odors: vegetation, restaurants/cafes, trash bins, exhaust from traffic
   - Colors, materials, and textures that influence thermal and psychological perceptions
   

2. **Summarization of image features within spatial context**: Focus on the visual elements in the image that represent or reflect the spatial context, including:
   - Visual clues that indicate which neighborhood or area this belongs to
   - Street features that align with the network information (like street crossings, street width, street type)
   - Visual indicators of proximity to other locations mentioned in the spatial context
   - Architectural or urban design elements characteristic of this region
   - Street signs, direction indicators, or landmarks that help situate this image in the broader network
   - Overall multi-sensory quality that influences psychological wellbeing (pleasant natural scents, restorative sounds, appealing colors vs. exhaust fumes, traffic noise, harsh urban aesthetics)

# Example Format:

**Image:** The first image shows urban scene located in 1st Avenue. It is a wide commercial boulevard with modernist storefronts, pedestrians on the sidewalk, and street trees in planters. A bike parking facility is visible on the right side of the frame. 

**Summarization:** 
Given the spatial context where the image was taken, the wide multi-lane design and commercial storefronts are typical of Roosevelt Avenue in North Corona. The street signs visible at the corner indicate this is near the intersection with 104 Street. The density of retail establishments and the urban setting with mixed-use buildings are characteristic of this commercial district, with architectural styles common to this part of Queens.

**Instructions:**
- For the image provided, generate "**Image:**" symbol before captions
- End with a "**Summarization:**" section
- Connect individual observations to the broader spatial understanding and connectivity patterns
- Refer to the places and locations in network descriptions when describing the image
- Also consider the interactive effects of visual, acoustic, and olfactory environments on mental wellbeing and emotional responses
"""
    
    return prompt


def generate_captions_from_mapillary_results(
    mapillary_results_file,
    training_data_file,
    output_file,
    nodes_gdf,
    edges_gdf,
    place='singapore',
    api_key=None,
    api_provider='qwen',
    delay_between_calls=3,
    subgraph_radius_m=500
):
    """
    Generate captions for mapillary results by building subgraphs from coordinates.
    
    Args:
        mapillary_results_file: Path to mapillary_results_cleaned.jsonl file
        training_data_file: Path to training data JSONL file (to filter out existing mapillary_ids)
        output_file: Output file for new captions
        nodes_gdf: GeoDataFrame containing node information
        edges_gdf: GeoDataFrame containing edge information
        place: Place name (default: 'singapore')
        api_key: API key for caption generation
        api_provider: API provider ('qwen' or 'yinli')
        delay_between_calls: Delay between API calls in seconds
        subgraph_radius_m: Radius in meters for subgraph creation (default: 500)
        
    Returns:
        dict: Statistics about caption generation
    """
    print("🚀 Starting caption generation from mapillary results...")
    
    # Step 1: Load checkpoint
    checkpoint_file = output_file.replace('.jsonl', '_checkpoint.jsonl')
    print(f"\n📊 Step 1: Loading checkpoint from {os.path.basename(checkpoint_file)}...")
    checkpoint_data = load_checkpoint(checkpoint_file)
    
    # Step 2: Extract mapillary_ids from training data to filter
    print("\n📊 Step 2: Extracting mapillary_ids from training data...")
    training_mapillary_ids = extract_mapillary_ids_from_training_data(training_data_file)
    
    # Step 3: Read mapillary results
    print("\n📊 Step 3: Loading mapillary results...")
    mapillary_items = []
    
    if not os.path.exists(mapillary_results_file):
        print(f"❌ Mapillary results file not found: {mapillary_results_file}")
        return {'total_items': 0, 'filtered_out': 0, 'new_captions': 0, 'failed': 0}
    
    try:
        with open(mapillary_results_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    data = json.loads(line)
                    mapillary_items.append(data)
                except json.JSONDecodeError as e:
                    print(f"⚠️ JSON decode error at line {line_num}: {e}")
                    continue
        
        print(f"✅ Loaded {len(mapillary_items)} items from mapillary results")
        
    except Exception as e:
        print(f"❌ Error reading mapillary results: {e}")
        traceback.print_exc()
        return {'total_items': 0, 'filtered_out': 0, 'new_captions': 0, 'failed': 0}
    
    # Step 4: Filter out items already in training data
    print("\n📊 Step 4: Filtering items...")
    items_to_process = []
    processed_from_checkpoint = checkpoint_data['processed_mapillary_ids']
    
    for item in mapillary_items:
        mapillary_id = str(item.get('mapillary_id', ''))
        if not mapillary_id:
            continue
        
        # Skip if already in training data
        if mapillary_id in training_mapillary_ids:
            continue
        
        # Skip if already processed in checkpoint
        if mapillary_id in processed_from_checkpoint:
            continue
        
        # Check if coordinates are available
        geometry = item.get('geometry', {})
        if not geometry or 'coordinates' not in geometry:
            continue
        
        coords = geometry['coordinates']
        if not coords or len(coords) != 2:
            continue
        
        items_to_process.append(item)
    
    # Initialize stats
    stats = {
        'total_items': len(mapillary_items),
        'filtered_out': len(mapillary_items) - len(items_to_process),
        'resumed_from_checkpoint': len(processed_from_checkpoint),
        'to_process': len(items_to_process),
        'new_captions': checkpoint_data['stats']['new_captions'],
        'failed': checkpoint_data['stats']['failed'],
        'no_subgraph': checkpoint_data['stats']['no_subgraph']
    }
    
    print(f"\n📈 Statistics:")
    print(f"   Total items in mapillary results: {stats['total_items']}")
    print(f"   Filtered out (in training data or checkpoint): {stats['filtered_out']}")
    if stats['resumed_from_checkpoint'] > 0:
        print(f"   Resumed from checkpoint: {stats['resumed_from_checkpoint']}")
    print(f"   Need new captions: {stats['to_process']}")
    
    if not items_to_process:
        print("\n✅ All items already processed or filtered out!")
        return stats
    
    # Create output directory if needed
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Step 5: Process each item
    print(f"\n📊 Step 5: Generating captions for {len(items_to_process)} items...")
    
    for i, item in enumerate(items_to_process):
        mapillary_id = str(item.get('mapillary_id', ''))
        geometry = item.get('geometry', {})
        coords = geometry.get('coordinates', [])
        lon, lat = coords[0], coords[1]  # GeoJSON format: [lon, lat]
        
        print(f"\n📸 Processing item {i+1}/{len(items_to_process)}: mapillary_id={mapillary_id}")
        print(f"   📍 Coordinates: ({lat:.6f}, {lon:.6f})")
        
        # Build image path
        image_path = os.path.join(DATA_ROOT, 'geo', 'SR', 'osm_data', place, 'images', f'{mapillary_id}.jpg')
        
        # Check if image exists
        if not os.path.exists(image_path):
            print(f"   ⚠️ Image not found: {image_path}")
            save_checkpoint_entry(checkpoint_file, mapillary_id, 'failed', 'Image file not found')
            stats['failed'] += 1
            continue
        
        # Build subgraph from coordinates
        print(f"   🔧 Building subgraph from coordinates (radius: {subgraph_radius_m}m)...")
        try:
            subgraph = create_subgraph_from_coords(nodes_gdf, edges_gdf, lat, lon, radius_m=subgraph_radius_m)
            
            if subgraph.number_of_nodes() == 0:
                print(f"   ⚠️ No nodes found in subgraph - skipping")
                stats['no_subgraph'] += 1
                save_checkpoint_entry(checkpoint_file, mapillary_id, 'no_subgraph', 'No nodes in subgraph')
                continue
            
            print(f"   ✅ Created subgraph: {subgraph.number_of_nodes()} nodes, {subgraph.number_of_edges()} edges")
            
        except Exception as e:
            error_msg = str(e)
            print(f"   ❌ Error building subgraph: {error_msg}")
            traceback.print_exc()
            stats['no_subgraph'] += 1
            save_checkpoint_entry(checkpoint_file, mapillary_id, 'no_subgraph', f'Error building subgraph: {error_msg}')
            continue
        
        # Add mapillary node to subgraph and expand
        print(f"   🔧 Adding mapillary node and expanding subgraph...")
        try:
            subgraph = add_mapillary_node_to_subgraph(
                subgraph, 
                mapillary_id, 
                (lat, lon), 
                nodes_gdf, 
                edges_gdf, 
                max_hops=2
            )
        except Exception as e:
            error_msg = str(e)
            print(f"   ❌ Error adding mapillary node: {error_msg}")
            traceback.print_exc()
            stats['no_subgraph'] += 1
            save_checkpoint_entry(checkpoint_file, mapillary_id, 'no_subgraph', f'Error adding mapillary node: {error_msg}')
            continue
        
        # Convert mapillary_id to int for center node
        try:
            mapillary_id_int = int(mapillary_id)
        except:
            print(f"   ⚠️ Cannot convert mapillary_id to int: {mapillary_id}")
            stats['failed'] += 1
            save_checkpoint_entry(checkpoint_file, mapillary_id, 'failed', 'Invalid mapillary_id format')
            continue
        
        # Set center nodes
        center_nodes = [mapillary_id_int]
        
        # Build node_text and coords attributes (matching clean_graph_attributes.py format)
        print(f"   🔄 Building node_text and coords attributes...")
        create_node_text_and_coords(subgraph, nodes_gdf, place)
        clean_node_attributes(subgraph)
        print(f"   ✅ Built node_text and coords attributes for {subgraph.number_of_nodes()} nodes")
        
        # Create subgraph_data dict
        subgraph_data = {
            'subgraph': subgraph,
            'center_nodes': center_nodes
        }
        
        # Save subgraph to pickle file
        subgraph_path = None
        try:
            subgraph_dir = os.path.join(OUTPUT_ROOT, f'enhanced_image_data_with_paths_and_captions_{place}', 'subgraphs_with_paths')
            os.makedirs(subgraph_dir, exist_ok=True)
            
            subgraph_filename = f"mapillary_{mapillary_id}_subgraph.pkl"
            subgraph_path = os.path.join(subgraph_dir, subgraph_filename)
            
            with open(subgraph_path, 'wb') as f:
                pickle.dump(subgraph_data, f)
            print(f"   💾 Saved subgraph to: {os.path.basename(subgraph_path)}")
            
        except Exception as e:
            print(f"   ⚠️ Failed to save subgraph: {e}")
        
        # Generate subgraph description
        print(f"   🔄 Generating subgraph description...")
        subgraph_desc = generate_subgraph_description(subgraph_data, nodes_gdf, edges_gdf)
        
        if not subgraph_desc:
            print(f"   ⚠️ Failed to generate subgraph description - skipping")
            stats['no_subgraph'] += 1
            save_checkpoint_entry(checkpoint_file, mapillary_id, 'no_subgraph', 'Failed to generate subgraph description')
            continue
        
        # Create enhanced prompt
        prompt = create_enhanced_prompt_with_subgraph_description(subgraph_desc)
        
        # Generate caption
        try:
            if api_provider == 'qwen':
                caption, valid_path = use_qwen_vl_for_single_image_captioning(
                    prompt_text=prompt,
                    image_path=image_path,
                    api_key=api_key
                )
            elif api_provider == 'yinli':
                caption, valid_path = use_yinli_for_single_image_captioning(
                    prompt_text=prompt,
                    image_path=image_path,
                    api_key=api_key
                )
            else:
                raise ValueError(f"Unsupported API provider: {api_provider}")
            
            if caption and valid_path:
                # Extract image caption and summarization
                image_caption, summarization_text = extract_captions_from_enhanced_format(caption)
                
                if image_caption is None or summarization_text is None:
                    print(f"   ⏭️  Skipping - no valid summarization format")
                    save_checkpoint_entry(checkpoint_file, mapillary_id, 'failed', 'No summarization section found')
                    stats['failed'] += 1
                    continue
                
                # Create result structure
                result = {
                    "image_paths": valid_path,
                    "prompt": prompt,
                    "image_caption": image_caption,
                    "summarization": summarization_text,
                    "swift_format": {
                        "messages": [
                            {
                                "role": "user",
                                "content": f"<graph><image>{image_caption} <summarization>{summarization_text}"
                            },
                            {
                                "role": "assistant",
                                "content": summarization_text
                            }
                        ],
                        "images": [valid_path],
                        "graphs": [subgraph_path] if subgraph_path and os.path.exists(subgraph_path) else [],
                        "label": 1.0
                    },
                    "source": "mapillary_results_with_built_subgraphs",
                    "mapillary_id": mapillary_id,
                    "coordinates": [lon, lat],
                    "timestamp": datetime.now().isoformat()
                }
                
                # Append to output file
                with open(output_file, 'a', encoding='utf-8') as f:
                    json.dump(result, f)
                    f.write('\n')
                
                stats['new_captions'] += 1
                save_checkpoint_entry(checkpoint_file, mapillary_id, 'success')
                print(f"   ✅ Caption generated and saved")
                print(f"   Caption length: {len(caption)} characters")
                
            else:
                print(f"   ⚠️ Failed to generate caption")
                stats['failed'] += 1
                save_checkpoint_entry(checkpoint_file, mapillary_id, 'failed', 'Caption generation returned empty result')
                
        except Exception as e:
            error_msg = str(e)
            print(f"   ❌ Error generating caption: {error_msg}")
            traceback.print_exc()
            stats['failed'] += 1
            save_checkpoint_entry(checkpoint_file, mapillary_id, 'failed', error_msg)
        
        # Add delay between calls
        if i < len(items_to_process) - 1:
            print(f"   ⏳ Waiting {delay_between_calls} seconds before next call...")
            time.sleep(delay_between_calls)
    
    # Print final statistics
    print(f"\n🎉 Caption generation complete!")
    print(f"📊 Final Statistics:")
    print(f"   Total items: {stats['total_items']}")
    print(f"   Filtered out: {stats['filtered_out']}")
    print(f"   Items to process: {stats['to_process']}")
    print(f"   Successfully generated: {stats['new_captions']}")
    print(f"   Failed: {stats['failed']}")
    print(f"   No subgraph available: {stats['no_subgraph']}")
    if stats['to_process'] > 0:
        print(f"   Success rate: {stats['new_captions']/stats['to_process']*100:.1f}%")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='Generate captions from mapillary results by building subgraphs')
    parser.add_argument('--place', type=str, default='singapore', help='Place name')
    parser.add_argument('--mapillary_results_file', type=str, default=None,
                        help='Path to mapillary_results_cleaned.jsonl file')
    parser.add_argument('--training_data_file', type=str, default=None,
                        help='Path to training data JSONL file (to filter existing mapillary_ids)')
    parser.add_argument('--output_file', type=str, default=None,
                        help='Output file for new captions')
    parser.add_argument('--api_key', type=str, default='sk-e50ba2aae6d54d57ae6aca1f4c4aee4c', help='API key for caption generation')
    parser.add_argument('--api_provider', type=str, default='qwen', choices=['qwen', 'yinli'],
                        help='API provider for caption generation')
    parser.add_argument('--delay', type=int, default=6, help='Delay between API calls in seconds')
    parser.add_argument('--subgraph_radius', type=int, default=700, help='Radius in meters for subgraph creation')

    args = parser.parse_args()
    
    # Define data_folder first  (used for default paths)
    data_folder = os.path.join(DATA_ROOT, 'geo', 'SR', 'osm_data', args.place)
    
    # Set default paths based on place if not provided
    if args.mapillary_results_file is None:
        args.mapillary_results_file = os.path.join(data_folder, 'mapillary_results_cleaned.jsonl')
        print(f"📝 Using default mapillary results file: {args.mapillary_results_file}")
    
    if args.training_data_file is None:
        args.training_data_file = os.path.join(OUTPUT_ROOT, 'multiview_three_pairs', 'both_images_and_graphs.jsonl')
        print(f"📝 Using default training data file: {args.training_data_file}")
    
    if args.output_file is None:
        output_dir = os.path.join(OUTPUT_ROOT, f'enhanced_image_data_with_paths_and_captions_{args.place}')
        os.makedirs(output_dir, exist_ok=True)
        args.output_file = os.path.join(output_dir, 'mapillary_results_captions_with_subgraphs.jsonl')
        print(f"📝 Using default output file: {args.output_file}")
    
    # Get API key from environment if not provided
    if args.api_key is None:
        if args.api_provider == 'qwen':
            args.api_key = os.getenv("DASHSCOPE_API_KEY")
        elif args.api_provider == 'yinli':
            args.api_key = os.getenv("NEWAPI_API_KEY")
    
    # Load GeoDataFrames
    
    print(f"🔄 Loading GeoDataFrames from {data_folder}...")
    
    NODES_PATH = f'{data_folder}/nodes_with_districts.geojson'
    EDGES_PATH = f'{data_folder}/edges.geojson'
    MAPILLARY_NODES_PATH = f'{data_folder}/nodes_mapillary_with_districts.geojson'
    MAPILLARY_EDGES_PATH = f'{data_folder}/edges_mapillary.geojson'
    
    nodes_gdf = gpd.read_file(NODES_PATH)
    mapillary_nodes_gdf = gpd.read_file(MAPILLARY_NODES_PATH)
    nodes_gdf['id'] = nodes_gdf['id'].astype(int)
    mapillary_nodes_gdf['id'] = mapillary_nodes_gdf['id'].astype(int)
    nodes_gdf = pd.concat([nodes_gdf, mapillary_nodes_gdf], ignore_index=True)
    
    edges_gdf = gpd.read_file(EDGES_PATH)
    mapillary_edges_gdf = gpd.read_file(MAPILLARY_EDGES_PATH)
    edges_gdf = pd.concat([edges_gdf, mapillary_edges_gdf], ignore_index=True)
    edges_gdf['id1'] = edges_gdf['id1'].astype(int)
    edges_gdf['id2'] = edges_gdf['id2'].astype(int)
    
    # Clean up node names - handle both Complex_Crossing and regular Crossing
    print(f"🔧 Cleaning up crossing node names...")
    
    # First handle Complex_Crossing
    complex_crossing_mask = nodes_gdf['name'].str.contains('Complex_Crossing', na=False)
    if complex_crossing_mask.any():
        nodes_gdf.loc[complex_crossing_mask, 'name'] = \
            nodes_gdf.loc[complex_crossing_mask, 'name'].str.replace(
                'Complex_Crossing_', 'Complex Crossing of ', regex=False
            ).str.replace('_', ' and ', n=1).str.replace('_', ' ', regex=False)
        print(f"   Cleaned {complex_crossing_mask.sum()} Complex_Crossing names")
    
    # Then handle regular Crossing_ prefix
    crossing_mask = nodes_gdf['name'].str.contains('^Crossing_', na=False, regex=True)
    if crossing_mask.any():
        nodes_gdf.loc[crossing_mask, 'name'] = \
            nodes_gdf.loc[crossing_mask, 'name'].str.replace(
                'Crossing_', 'Intersection of ', regex=False
            ).str.replace('_', ' and ', regex=False)
        print(f"   Cleaned {crossing_mask.sum()} Crossing_ names")
    
    # Replace crossing node names with their address field (which should be cleaner)
    crossing_type_mask = nodes_gdf['type'] == 'crossing'
    if crossing_type_mask.any():
        # Use address where available, keep cleaned name otherwise
        nodes_gdf.loc[crossing_type_mask & nodes_gdf['address'].notna(), 'name'] = \
            nodes_gdf.loc[crossing_type_mask & nodes_gdf['address'].notna(), 'address']
        print(f"   Replaced {(crossing_type_mask & nodes_gdf['address'].notna()).sum()} crossing names with address")
    
    print(f"✅ Loaded {len(nodes_gdf)} nodes and {len(edges_gdf)} edges")
    
    # Generate captions
    stats = generate_captions_from_mapillary_results(
        mapillary_results_file=args.mapillary_results_file,
        training_data_file=args.training_data_file,
        output_file=args.output_file,
        nodes_gdf=nodes_gdf,
        edges_gdf=edges_gdf,
        place=args.place,
        api_key=args.api_key,
        api_provider=args.api_provider,
        delay_between_calls=args.delay,
        subgraph_radius_m=args.subgraph_radius
    )
    
    print(f"\n🎉 Processing completed!")
    print(f"📁 Results saved to: {args.output_file}")


if __name__ == "__main__":
    main()

