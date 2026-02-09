"""
Training Dataset Composer for Urban Knowledge Graph
Extracts and processes node information from JSONL data to create training datasets.
"""

import json
import os
import re
from typing import Dict, List, Set, Tuple, Any, Optional
from collections import defaultdict
import pandas as pd

# Get base paths from environment variables
DATA_ROOT = os.getenv('URBANKG_DATA_ROOT', './data')
OUTPUT_ROOT = os.getenv('URBANKG_OUTPUT_ROOT', './output')

ENHANCED_SUBGRAPH_DIR_NAME = "enhanced_big"


def get_max_graph_nodes_limit(data_folder: Optional[str]) -> int:
    """
    Determine the max node limit for graph pruning based on data folder.
    """
    data_key = (data_folder or "").lower()
    if data_key == "newyork":
        return 2000
    if data_key == "singapore":
        return 500
    return 2000

def convert_bearing_to_direction(bearing):
    """
    Convert bearing angle to direction string format.
    
    Args:
        bearing: Compass angle in degrees
        
    Returns:
        Formatted direction string like "94°(E)"
    """
    try:
        bearing = float(bearing)
        bearing = bearing % 360

        directions = [
            (0, 'N'), (45, 'NE'), (90, 'E'), (135, 'SE'),
            (180, 'S'), (225, 'SW'), (270, 'W'), (315, 'NW'), (360, 'N')
        ]

        min_diff = float('inf')
        best_direction = 'N'

        for deg, direction in directions:
            diff = min(abs(bearing - deg), abs(bearing - deg + 360), abs(bearing - deg - 360))
            if diff < min_diff:
                min_diff = diff
                best_direction = direction

        return f"{bearing:.0f}°({best_direction})"
    except:
        return str(bearing)


def convert_bearing_to_cardinal_direction(bearing):
    """
    Convert bearing angle to cardinal direction name for user messages.
    
    Args:
        bearing: Compass angle in degrees
        
    Returns:
        Cardinal direction name like "north", "east", etc.
    """
    try:
        bearing = float(bearing)
        bearing = bearing % 360

        directions = [
            (0, 'north'), (45, 'northeast'), (90, 'east'), (135, 'southeast'),
            (180, 'south'), (225, 'southwest'), (270, 'west'), (315, 'northwest'), (360, 'north')
        ]

        min_diff = float('inf')
        best_direction = 'north'

        for deg, direction in directions:
            diff = min(abs(bearing - deg), abs(bearing - deg + 360), abs(bearing - deg - 360))
            if diff < min_diff:
                min_diff = diff
                best_direction = direction

        return best_direction
    except:
        return 'north'


def has_valid_source_name(data: Dict) -> bool:
    """
    Check if the item has valid source_name and target_name (not all digits).
    Exception: Allow all-digit names if the node is a Mapillary node (for reversed paths).
    
    Args:
        data: JSONL item data
        
    Returns:
        True if source_name and target_name are valid (not all digits or are Mapillary nodes), False otherwise
    """
    try:
        # Check path_meta for source_name and target_name in path_triples
        if 'path_meta' in data and 'paths_parsed' in data['path_meta']:
            for path_data in data['path_meta']['paths_parsed']:
                if 'path_triples' in path_data and path_data['path_triples']:
                    # Check the first triple's source_name
                    first_triple = path_data['path_triples'][0]
                    source_name = first_triple.get('source_name', '')
                    
                    # Skip if source_name is empty or None
                    if not source_name:
                        return False
                    
                    # Check if source_name is ALL digits (like "3579354165502880")
                    if source_name.isdigit():
                        # Allow if source is a Mapillary node (for reversed paths)
                        source_info = first_triple.get('source_info', {})
                        if source_info and source_info.get('type') == 'mapillary':
                            pass  # Valid, continue to check target
                        else:
                            # Otherwise reject (invalid OSM node name that's all digits)
                            return False
                    
                    # Check the last triple's target_name
                    last_triple = path_data['path_triples'][-1]
                    target_name = last_triple.get('target_name', '')
                    
                    # Skip if target_name is empty or None
                    if not target_name:
                        return False
                    
                    # Check if target_name is ALL digits
                    if target_name.isdigit():
                        # Allow if target is a Mapillary node (for regular paths)
                        target_info = last_triple.get('target_info', {})
                        if target_info and target_info.get('type') == 'mapillary':
                            return True  # Valid Mapillary target
                        else:
                            # Otherwise reject (invalid OSM node name that's all digits)
                            return False
                    
                    return True
        
        return False
    except Exception as e:
        print(f"Warning: Error validating source_name/target_name: {e}")
        return False


def load_compass_angles(compass_file_path: str) -> Dict[str, float]:
    """
    Load compass angles from JSONL file.
    
    Args:
        compass_file_path: Path to compass angles JSONL file
        
    Returns:
        Dictionary mapping mapillary_id to compass_angle
    """
    compass_angles = {}
    
    try:
        with open(compass_file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    mapillary_id = data.get('mapillary_id')
                    compass_angle = data.get('compass_angle')
                    
                    if mapillary_id and compass_angle is not None:
                        compass_angles[mapillary_id] = compass_angle
                        
                except json.JSONDecodeError as e:
                    print(f"Warning: Error parsing compass data line {line_num}: {e}")
                    continue
                except Exception as e:
                    print(f"Warning: Error processing compass data line {line_num}: {e}")
                    continue
                    
    except FileNotFoundError:
        print(f"Warning: Compass angles file not found: {compass_file_path}")
    except Exception as e:
        print(f"Warning: Error loading compass angles: {e}")
    
    print(f"Loaded {len(compass_angles)} compass angles")
    return compass_angles


def clean_formatted_path_enhanced(formatted_path: str) -> str:
    """
    Clean up formatted_path_enhanced by removing duplicate transition tuples.
    
    The function removes the first occurrence of consecutive transition tuples
    by splitting on "->" and keeping only the last tuple when consecutive tuples are found.
    
    Args:
        formatted_path: The formatted path string with potential duplicate tuples
        
    Returns:
        Cleaned formatted path string
    """
    # Split by "->" to get all parts
    parts = formatted_path.split(" -> ")
    
    cleaned_parts = []
    i = 0
    
    while i < len(parts):
        current_part = parts[i]
        
        # Check if current part is a transition tuple
        # Transition tuples: (distance, direction) like (8.1m, 266°(W)) - have exactly 1 comma
        # Triples: (node, relation, node) like (A, nearest, B) - have exactly 2 commas
        if current_part.startswith("(") and current_part.endswith(")"):
            content = current_part[1:-1]  # Remove parentheses
            comma_count = content.count(",")
            
            # Transition tuple has 1 comma (distance, direction)
            if comma_count == 1:
                # Check if next part is also a transition tuple
                if i + 1 < len(parts):
                    next_part = parts[i + 1]
                    if next_part.startswith("(") and next_part.endswith(")"):
                        next_content = next_part[1:-1]
                        next_comma_count = next_content.count(",")
                        
                        # If next part is also a transition tuple (1 comma), skip current one
                        if next_comma_count == 1:
                            i += 1
                            continue
        
        cleaned_parts.append(current_part)
        i += 1
    
    return " -> ".join(cleaned_parts)




def parse_geometry_to_coordinates(geometry_str: str, enable_geocoding: bool = False,
                                geocoding_cache: Optional[Dict] = None,
                                data_folder: Optional[str] = None) -> str:
    """
    Parse complex geometry information to coordinate strings.
    Based on the provided geometry parsing logic.
    
    Args:
        geometry_str: Geometry string from node data
        enable_geocoding: Whether to enable geocoding for context
        geocoding_cache: Cache for geocoding results
        data_folder: Data folder path for geocoding
        
    Returns:
        Optimized geometry string with coordinates
    """
    optimized_geometry = ""
    
    try:
        if geometry_str and geometry_str != 'nan' and geometry_str.lower() != 'none':
            # Try to parse the geometry
            if 'POINT' in geometry_str.upper():
                # Extract coordinates from POINT
                coords = re.findall(r'[-+]?\d*\.?\d+', geometry_str)
                if len(coords) >= 2:
                    # Limit to 8 decimal places
                    lon = f"{float(coords[0]):.8f}"
                    lat = f"{float(coords[1]):.8f}"
                    optimized_geometry = f"POINT(lon: {lon}, lat: {lat})"

            elif 'MULTILINESTRING' in geometry_str.upper():
                # Extract start and end points from first and last linestrings
                coords = re.findall(r'[-+]?\d*\.?\d+', geometry_str)
                if len(coords) >= 4:
                    # Limit to 8 decimal places
                    start_lon = f"{float(coords[0]):.8f}"
                    start_lat = f"{float(coords[1]):.8f}"
                    end_lon = f"{float(coords[-2]):.8f}"
                    end_lat = f"{float(coords[-1]):.8f}"
                    optimized_geometry = f"MULTILINE(lon: {start_lon}, lat: {start_lat} to lon: {end_lon}, lat: {end_lat})"
                else:
                    optimized_geometry = "MULTILINE(complex)"

            elif 'LINESTRING' in geometry_str.upper():
                # Extract start and end points from LINESTRING
                coords = re.findall(r'[-+]?\d*\.?\d+', geometry_str)
                if len(coords) >= 4:
                    # Limit to 8 decimal places
                    start_lon = f"{float(coords[0]):.8f}"
                    start_lat = f"{float(coords[1]):.8f}"
                    end_lon = f"{float(coords[-2]):.8f}"
                    end_lat = f"{float(coords[-1]):.8f}"
                    optimized_geometry = f"LINE(lon: {start_lon}, lat: {start_lat} to lon: {end_lon}, lat: {end_lat})"

            elif 'MULTIPOLYGON' in geometry_str.upper():
                # Try to calculate centroid using shapely if available
                try:
                    from shapely import wkt
                    from shapely.geometry import MultiPolygon
                    geom = wkt.loads(geometry_str)
                    if isinstance(geom, MultiPolygon):
                        centroid = geom.centroid
                        lon = f"{centroid.x:.8f}"
                        lat = f"{centroid.y:.8f}"
                        optimized_geometry = f"MULTI_CENTROID(lon: {lon}, lat: {lat})"
                    else:
                        raise Exception("Not a multipolygon")
                except:
                    # Fallback: extract first coordinate from first polygon
                    coords = re.findall(r'[-+]?\d*\.?\d+', geometry_str)
                    if len(coords) >= 2:
                        lon = f"{float(coords[0]):.8f}"
                        lat = f"{float(coords[1]):.8f}"
                        optimized_geometry = f"MULTI_APPROX(lon: {lon}, lat: {lat})"
                    else:
                        optimized_geometry = "MULTIPOLYGON(complex)"

            elif 'POLYGON' in geometry_str.upper():
                # Try to calculate centroid using shapely if available
                try:
                    from shapely import wkt
                    from shapely.geometry import Polygon
                    geom = wkt.loads(geometry_str)
                    if isinstance(geom, Polygon):
                        centroid = geom.centroid
                        lon = f"{centroid.x:.8f}"
                        lat = f"{centroid.y:.8f}"
                        optimized_geometry = f"CENTROID(lon: {lon}, lat: {lat})"
                    else:
                        raise Exception("Not a polygon")
                except:
                    # Fallback: extract first coordinate
                    coords = re.findall(r'[-+]?\d*\.?\d+', geometry_str)
                    if len(coords) >= 2:
                        lon = f"{float(coords[0]):.8f}"
                        lat = f"{float(coords[1]):.8f}"
                        optimized_geometry = f"APPROX_CENTER(lon: {lon}, lat: {lat})"

            elif 'GEOMETRYCOLLECTION' in geometry_str.upper():
                # Handle geometry collections by extracting first coordinate
                coords = re.findall(r'[-+]?\d*\.?\d+', geometry_str)
                if len(coords) >= 2:
                    lon = f"{float(coords[0]):.8f}"
                    lat = f"{float(coords[1]):.8f}"
                    optimized_geometry = f"COLLECTION(lon: {lon}, lat: {lat})"
                else:
                    optimized_geometry = "COLLECTION(complex)"

            else:
                # For other geometries, try to extract first coordinate
                coords = re.findall(r'[-+]?\d*\.?\d+', geometry_str)
                if len(coords) >= 2:
                    lon = f"{float(coords[0]):.8f}"
                    lat = f"{float(coords[1]):.8f}"
                    optimized_geometry = f"POINT(lon: {lon}, lat: {lat})"

    except Exception as e:
        # If geometry parsing fails, use simplified representation
        coords = re.findall(r'[-+]?\d*\.?\d+', geometry_str)
        if len(coords) >= 2:
            lon = f"{float(coords[0]):.8f}"
            lat = f"{float(coords[1]):.8f}"
            optimized_geometry = f"POINT(lon: {lon}, lat: {lat})"
        else:
            optimized_geometry = "GEOM(unparseable)"
    
    return optimized_geometry


def extract_unique_nodes_from_jsonl(jsonl_file_path: str, data_folder: str = "singapore", nodes_gdf=None) -> Tuple[Dict[str, Dict], Set[str]]:
    """
    Extract unique nodes from source_info and target_info in JSONL data.
    Retrieves node attributes from nodes_gdf when available.
    
    Args:
        jsonl_file_path: Path to the JSONL file
        data_folder: Data folder name (e.g., 'singapore', 'newyork')
        nodes_gdf: GeoDataFrame with node information (optional, for attribute lookup)
        
    Returns:
        Tuple of (unique_nodes_dict, mapillary_node_ids)
    """
    unique_nodes = {}
    mapillary_nodes = set()
    
    # Create a helper function to get node attributes from nodes_gdf
    def get_node_attributes_from_gdf(node_id, info_dict):
        """Get node attributes from nodes_gdf if available, otherwise fallback to info_dict"""
        if nodes_gdf is not None and not nodes_gdf.empty:
            node_row = nodes_gdf[nodes_gdf['id'] == int(node_id)]
            if not node_row.empty:
                node_data = node_row.iloc[0]
                attrs = {
                    'id': node_id,
                    'name': node_data.get('name') if 'name' in node_data and pd.notna(node_data.get('name')) else info_dict.get('name'),
                    'category': node_data.get('category') if 'category' in node_data and pd.notna(node_data.get('category')) else info_dict.get('category'),
                    'type': node_data.get('type') if 'type' in node_data and pd.notna(node_data.get('type')) else info_dict.get('type'),
                    'geometry': info_dict.get('geometry'),  # Keep geometry from info_dict
                    'osm_id': node_data.get('osm_id') if 'osm_id' in node_data and pd.notna(node_data.get('osm_id')) else info_dict.get('osm_id'),
                    'source': 'osm'
                }
                
                # Add location-specific fields from nodes_gdf based on data_folder
                if data_folder == "newyork":
                    if 'neighborhood' in node_data and pd.notna(node_data.get('neighborhood')):
                        attrs['neighborhood'] = node_data['neighborhood']
                    if 'borough' in node_data and pd.notna(node_data.get('borough')):
                        attrs['borough'] = node_data['borough']
                else:  # singapore or default
                    if 'planning_area' in node_data and pd.notna(node_data.get('planning_area')):
                        attrs['planning_area'] = node_data['planning_area']
                    if 'district' in node_data and pd.notna(node_data.get('district')):
                        attrs['district'] = node_data['district']
                
                return attrs
        
        # Fallback to info_dict if nodes_gdf not available or node not found
        return None
    
    with open(jsonl_file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                
                # Extract path_triples from path_meta
                if 'path_meta' in data and 'paths_parsed' in data['path_meta']:
                    for path_data in data['path_meta']['paths_parsed']:
                        if 'path_triples' in path_data:
                            for triple in path_data['path_triples']:
                                # Process source_info
                                if 'source_info' in triple and triple['source_info']:
                                    source_info = triple['source_info']
                                    node_id = source_info.get('id')
                                    node_id = str(node_id)
                                    if node_id and node_id not in unique_nodes:
                                        # Check if it's a Mapillary node (can be in source for reversed paths)
                                        if source_info.get('type') == 'mapillary':
                                            mapillary_nodes.add(node_id)
                                            unique_nodes[node_id] = {
                                                'id': node_id,
                                                'name': source_info.get('name') or node_id,
                                                'category': source_info.get('category') or 'mapillary_image',
                                                'type': source_info.get('type'),
                                                'geometry': source_info.get('geometry'),
                                                'osm_id': source_info.get('osm_id'),
                                                'url': source_info.get('url'),
                                                'source': 'mapillary'
                                            }
                                        else:
                                            # Try to get attributes from nodes_gdf first
                                            node_data = get_node_attributes_from_gdf(node_id, source_info)
                                            if node_data is None:
                                                # Fallback to source_info
                                                node_data = {
                                                    'id': node_id,
                                                    'name': source_info.get('name'),
                                                    'category': source_info.get('category'),
                                                    'type': source_info.get('type'),
                                                    'geometry': source_info.get('geometry'),
                                                    'osm_id': source_info.get('osm_id'),
                                                    'source': 'osm'
                                                }
                                                # Add location-specific fields based on data_folder
                                                if data_folder == "newyork":
                                                    node_data['neighborhood'] = source_info.get('neighborhood')
                                                    node_data['borough'] = source_info.get('borough')
                                                else:  # singapore or default
                                                    node_data['planning_area'] = source_info.get('planning_area')
                                                    node_data['district'] = source_info.get('district')
                                            unique_nodes[node_id] = node_data
                                
                                # Process target_info
                                if 'target_info' in triple and triple['target_info']:
                                    target_info = triple['target_info']
                                    node_id = target_info.get('id')
                                    node_id = str(node_id)
                                    if node_id and node_id not in unique_nodes:
                                        # Check if it's a Mapillary node
                                        if target_info.get('type') == 'mapillary':
                                            mapillary_nodes.add(node_id)
                                            unique_nodes[node_id] = {
                                                'id': node_id,
                                                'name': target_info.get('name') or node_id,
                                                'category': target_info.get('category') or 'mapillary_image',
                                                'type': target_info.get('type'),
                                                'geometry': target_info.get('geometry'),
                                                'osm_id': target_info.get('osm_id'),
                                                'url': target_info.get('url'),
                                                'source': 'mapillary'
                                            }
                                        else:
                                            # Try to get attributes from nodes_gdf first
                                            node_data = get_node_attributes_from_gdf(node_id, target_info)
                                            if node_data is None:
                                                # Fallback to target_info
                                                node_data = {
                                                    'id': node_id,
                                                    'name': target_info.get('name'),
                                                    'category': target_info.get('category'),
                                                    'type': target_info.get('type'),
                                                    'geometry': target_info.get('geometry'),
                                                    'osm_id': target_info.get('osm_id'),
                                                    'source': 'osm'
                                                }
                                                # Add location-specific fields based on data_folder
                                                if data_folder == "newyork":
                                                    node_data['neighborhood'] = target_info.get('neighborhood')
                                                    node_data['borough'] = target_info.get('borough')
                                                else:  # singapore or default
                                                    node_data['planning_area'] = target_info.get('planning_area')
                                                    node_data['district'] = target_info.get('district')
                                            unique_nodes[node_id] = node_data
                
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                continue
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                continue
    
    return unique_nodes, mapillary_nodes


def build_node_attributes_descriptions(node_data: Dict[str, Any], 
                                      enable_geocoding: bool = False,
                                      geocoding_cache: Optional[Dict] = None,
                                      data_folder: Optional[str] = None) -> Dict[str, str]:
    """
    Build node attribute descriptions in JSON format with name, category, coordinates string keys.
    Also includes location-specific fields based on data_folder.
    
    Args:
        node_data: Dictionary containing node information
        enable_geocoding: Whether to enable geocoding for enhanced context
        geocoding_cache: Cache for geocoding results
        data_folder: Data folder path for geocoding (e.g., 'newyork', 'singapore')
        
    Returns:
        Dictionary with 'name', 'category', 'coordinates' keys, and optionally
        'neighborhood', 'borough' (for New York) or 'planning_area', 'district' (for Singapore)
    """
    # Extract and clean name
    name = node_data.get('name', '')
    if name is None or str(name).lower() in ['nan', 'none', '']:
        name = str(node_data.get('id', 'Unknown'))
    
    # Extract and clean category
    category = node_data.get('category', '')
    if category is None or str(category).lower() in ['nan', 'none', '']:
        category = node_data.get('type', 'unknown')
    
    # Parse geometry to coordinates string
    geometry_str = str(node_data.get('geometry', ''))
    coordinates = parse_geometry_to_coordinates(
        geometry_str, enable_geocoding, geocoding_cache, data_folder
    )
    
    # Build result dictionary with basic fields
    result = {
        'name': str(name),
        'category': str(category),
        'coordinates': coordinates
    }
    
    # Add location-specific fields based on data_folder
    if data_folder == "newyork":
        neighborhood = node_data.get('neighborhood', '')
        if neighborhood and str(neighborhood).lower() not in ['nan', 'none', '']:
            result['neighborhood'] = str(neighborhood)
        borough = node_data.get('borough', '')
        if borough and str(borough).lower() not in ['nan', 'none', '']:
            result['borough'] = str(borough)
    else:  # singapore or default
        planning_area = node_data.get('planning_area', '')
        if planning_area and str(planning_area).lower() not in ['nan', 'none', '']:
            result['planning_area'] = str(planning_area)
        district = node_data.get('district', '')
        if district and str(district).lower() not in ['nan', 'none', '']:
            result['district'] = str(district)
    
    return result


def build_stringified_node_attributes(nodes_dict: Dict[str, Dict[str, Any]], 
                                     enable_geocoding: bool = False,
                                     geocoding_cache: Optional[Dict] = None,
                                     data_folder: Optional[str] = None) -> str:
    """
    Build stringified dictionary of node attributes for adding to user content.
    
    Args:
        nodes_dict: Dictionary of nodes with their data
        enable_geocoding: Whether to enable geocoding for enhanced context
        geocoding_cache: Cache for geocoding results
        data_folder: Data folder path for geocoding
        
    Returns:
        Stringified dictionary using str(dictionary)
    """
    attributes_dict = {}
    
    for node_id, node_data in nodes_dict.items():
        # Get node attributes
        attrs = build_node_attributes_descriptions(
            node_data, enable_geocoding, geocoding_cache, data_folder
        )
        
        name = attrs['name']
        category = attrs['category']
        coordinates = attrs['coordinates']
        
        # Build dictionary entry
        node_id = str(node_data.get('id', ''))
        node_entry = {
            "id": node_id,
            "category": category,
            "coordinates": coordinates
        }
        
        # Add location-specific fields if available
        if 'neighborhood' in attrs:
            node_entry['neighborhood'] = attrs['neighborhood']
        if 'borough' in attrs:
            node_entry['borough'] = attrs['borough']
        if 'planning_area' in attrs:
            node_entry['planning_area'] = attrs['planning_area']
        if 'district' in attrs:
            node_entry['district'] = attrs['district']
        
        attributes_dict[name] = node_entry
    
    # Simply stringify the dictionary
    return str(attributes_dict)


def build_place_entities_information(nodes_dict: Dict[str, Dict[str, Any]], 
                                    enable_geocoding: bool = False,
                                    geocoding_cache: Optional[Dict] = None,
                                    data_folder: Optional[str] = None) -> str:
    """
    Build the complete place entities information string for user content.
    
    Args:
        nodes_dict: Dictionary of nodes with their data
        enable_geocoding: Whether to enable geocoding for enhanced context
        geocoding_cache: Cache for geocoding results
        data_folder: Data folder path for geocoding
        
    Returns:
        Complete string: "Here are relevant place entities informations: {stringified_dict}"
    """
    stringified_dict = build_stringified_node_attributes(
        nodes_dict, enable_geocoding, geocoding_cache, data_folder
    )
    
    return f"Here are relevant place entities informations: {stringified_dict}"


def extract_relevant_nodes_for_example(training_example: Dict[str, Any], 
                                      unique_nodes: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Extract relevant nodes for a specific training example.
    
    Args:
        training_example: Single training example data
        unique_nodes: Dictionary of all unique nodes
        
    Returns:
        Dictionary of relevant nodes for this example
    """
    relevant_nodes = {}
    
    # Get path_triples from the training example
    if 'path_meta' in training_example and 'paths_parsed' in training_example['path_meta']:
        for path_data in training_example['path_meta']['paths_parsed']:
            if 'path_triples' in path_data:
                for triple in path_data['path_triples']:
                    # Add source node
                    if 'source_info' in triple and triple['source_info']:
                        source_id = str(triple['source_info'].get('id', ''))
                        if source_id in unique_nodes:
                            relevant_nodes[source_id] = unique_nodes[source_id]
                    
                    # Add target node
                    if 'target_info' in triple and triple['target_info']:
                        target_id = str(triple['target_info'].get('id', ''))
                        if target_id in unique_nodes:
                            relevant_nodes[target_id] = unique_nodes[target_id]
    
    return relevant_nodes


def enhance_user_message_content(user_content: str, image_coordinates: str, 
                                place_entities_info: str, mapillary_node_id: str,
                                cleaned_spatial_path: str = "", compass_angles: Dict[str, float] = None) -> str:
    """
    Enhance user message content by adding image coordinates, compass angle description,
    place entities information, and cleaned spatial reasoning path.
    
    Args:
        user_content: Original user message content
        image_coordinates: Image coordinates string (e.g., "(103.8440, 1.2968)")
        place_entities_info: Stringified place entities information
        mapillary_node_id: Mapillary node ID
        cleaned_spatial_path: Cleaned spatial reasoning path
        compass_angles: Dictionary mapping mapillary_id to compass_angle
        
    Returns:
        Enhanced user content
    """
    # Find the pattern where we need to add coordinates
    # The NEW pattern is: "image with id {mapillary_node_id}" (used in test2_mapillary_nodes2Copy1.py)
    # The OLD pattern was: "street image location with id {mapillary_node_id}"
    # Try both patterns for backward compatibility
    pattern_new = f"image with id {mapillary_node_id}"
    pattern_old = f"street image location with id {mapillary_node_id}"
    
    # Determine which pattern is in the content
    if pattern_new in user_content:
        pattern = pattern_new
    elif pattern_old in user_content:
        pattern = pattern_old
    else:
        pattern = None
    
    # Check if coordinates are already present in the content
    # Check for " at {image_coordinates}" (with leading space) which is what gets added
    coordinates_already_present = f" at {image_coordinates}" in user_content
    
    # Build the location description only if not already present
    if not coordinates_already_present:
        location_description = f" at {image_coordinates}"
        
        # Add compass angle description if available
        if compass_angles and mapillary_node_id in compass_angles:
            compass_angle = compass_angles[mapillary_node_id]
            direction_str = convert_bearing_to_cardinal_direction(compass_angle)
            location_description += f". When you face {direction_str}, you can see the scene in the image"
        
        if pattern:
            # Add location description after the mapillary node id
            enhanced_content = user_content.replace(
                pattern, 
                f"{pattern}{location_description}"
            )
        else:
            # If pattern not found, just add location description at the end
            enhanced_content = user_content + location_description
    else:
        # Coordinates already present, don't add them again
        enhanced_content = user_content
        
        # But still add compass angle if available and not already present
        if compass_angles and mapillary_node_id in compass_angles:
            compass_angle = compass_angles[mapillary_node_id]
            direction_str = convert_bearing_to_cardinal_direction(compass_angle)
            compass_description = f". When you face {direction_str}, you can see the scene in the image"
            
            # Only add compass description if not already present
            if "When you face" not in enhanced_content:
                if pattern and pattern in enhanced_content:
                    # Find where to insert compass description (after coordinates)
                    enhanced_content = enhanced_content.replace(
                        image_coordinates,
                        f"{image_coordinates}{compass_description}"
                    )
                else:
                    enhanced_content += compass_description
    
    # Add cleaned spatial reasoning path if available
    if cleaned_spatial_path:
        # Find the "Spatial Reasoning Path:" and replace it with cleaned version
        if "Spatial Reasoning Path:" in enhanced_content:
            # Extract the part before "Spatial Reasoning Path:"
            parts = enhanced_content.split("Spatial Reasoning Path:", 1)
            if len(parts) == 2:
                before_path = parts[0]
                # Find the part after the path (usually starts with "Based on")
                after_path_parts = parts[1].split("Based on", 1)
                if len(after_path_parts) == 2:
                    enhanced_content = f"{before_path}Spatial Reasoning Path: {cleaned_spatial_path}. Based on{after_path_parts[1]}"
                else:
                    enhanced_content = f"{before_path}Spatial Reasoning Path: {cleaned_spatial_path}"
    
    # Add place entities information at the end
    enhanced_content = f"{enhanced_content}\n{place_entities_info}"
    
    return enhanced_content


def load_and_concatenate_geodataframes(osm_file_path: str, mapillary_file_path: str) -> 'pd.DataFrame':
    """
    Load and concatenate OSM and Mapillary GeoJSON/Parquet files into a single GeoDataFrame.
    
    Args:
        osm_file_path: Path to OSM geojson or parquet file
        mapillary_file_path: Path to Mapillary geojson file
        
    Returns:
        Combined GeoDataFrame with both OSM and Mapillary data
    """
    import pandas as pd  # Import pandas at the top - used for concat and to_numeric operations
    
    try:
        import geopandas as gpd
        use_geopandas = True
    except ImportError:
        use_geopandas = False
        print("Warning: geopandas not available, using regular pandas DataFrame")
    
    combined_gdf = None
    
    # Load OSM data
    if os.path.exists(osm_file_path):
        try:
            if use_geopandas:
                # Check if it's a parquet file
                if osm_file_path.endswith('.parquet'):
                    osm_gdf = pd.read_parquet(osm_file_path)
                    # Convert to GeoDataFrame if it has geometry column
                    if 'geometry' in osm_gdf.columns:
                        try:
                            from shapely import wkt
                            osm_gdf['geometry'] = osm_gdf['geometry'].apply(lambda x: wkt.loads(x) if isinstance(x, str) else x)
                            osm_gdf = gpd.GeoDataFrame(osm_gdf, geometry='geometry')
                        except:
                            pass  # Keep as regular DataFrame if conversion fails
                    print(f"Loaded {len(osm_gdf)} OSM features from {osm_file_path} (parquet)")
                else:
                    osm_gdf = gpd.read_file(osm_file_path)
                    print(f"Loaded {len(osm_gdf)} OSM features from {osm_file_path}")
                combined_gdf = osm_gdf
            else:
                # Fallback to reading as JSON if geopandas not available
                with open(osm_file_path, 'r') as f:
                    osm_data = json.load(f)
                osm_features = []
                for feature in osm_data.get('features', []):
                    props = feature.get('properties', {})
                    props['geometry_str'] = str(feature.get('geometry', {}))
                    osm_features.append(props)
                combined_gdf = pd.DataFrame(osm_features)
                print(f"Loaded {len(combined_gdf)} OSM features from {osm_file_path}")
        except Exception as e:
            print(f"Warning: Could not load OSM file {osm_file_path}: {e}")
    
    # Load Mapillary data
    if os.path.exists(mapillary_file_path):
        try:
            if use_geopandas:
                mapillary_gdf = gpd.read_file(mapillary_file_path)
                print(f"Loaded {len(mapillary_gdf)} Mapillary features from {mapillary_file_path}")
                
                if combined_gdf is not None:
                    # Concatenate the GeoDataFrames
                    combined_gdf = pd.concat([combined_gdf, mapillary_gdf], ignore_index=True)
                else:
                    combined_gdf = mapillary_gdf
            else:
                # Fallback to reading as JSON if geopandas not available
                with open(mapillary_file_path, 'r') as f:
                    mapillary_data = json.load(f)
                mapillary_features = []
                for feature in mapillary_data.get('features', []):
                    props = feature.get('properties', {})
                    props['geometry_str'] = str(feature.get('geometry', {}))
                    mapillary_features.append(props)
                mapillary_df = pd.DataFrame(mapillary_features)
                print(f"Loaded {len(mapillary_df)} Mapillary features from {mapillary_file_path}")
                
                if combined_gdf is not None:
                    combined_gdf = pd.concat([combined_gdf, mapillary_df], ignore_index=True)
                else:
                    combined_gdf = mapillary_df
        except Exception as e:
            print(f"Warning: Could not load Mapillary file {mapillary_file_path}: {e}")
    
    if combined_gdf is not None:
        print(f"Combined GeoDataFrame has {len(combined_gdf)} total features")
        
        # Convert 'id' column to integers for consistency (for nodes)
        if 'id' in combined_gdf.columns:
            try:
                # Convert to numeric, handling any non-convertible values
                combined_gdf['id'] = pd.to_numeric(combined_gdf['id'], errors='coerce')
                # Drop rows where id conversion failed (NaN values)
                original_len = len(combined_gdf)
                combined_gdf = combined_gdf.dropna(subset=['id'])
                # Convert to integer type
                combined_gdf['id'] = combined_gdf['id'].astype(int)
                
                if len(combined_gdf) < original_len:
                    print(f"⚠️ Dropped {original_len - len(combined_gdf)} rows with non-numeric IDs")
                print(f"✅ Converted {len(combined_gdf)} node IDs to integers")
            except Exception as e:
                print(f"Warning: Could not convert IDs to integers: {e}")
        
        # Convert 'source' and 'target' columns to integers for consistency (for edges)
        for col in ['source', 'target']:
            if col in combined_gdf.columns:
                try:
                    combined_gdf[col] = pd.to_numeric(combined_gdf[col], errors='coerce')
                    original_len = len(combined_gdf)
                    combined_gdf = combined_gdf.dropna(subset=[col])
                    combined_gdf[col] = combined_gdf[col].astype(int)
                    
                    if len(combined_gdf) < original_len:
                        print(f"⚠️ Dropped {original_len - len(combined_gdf)} rows with non-numeric {col} IDs")
                    print(f"✅ Converted {len(combined_gdf)} edge {col} IDs to integers")
                except Exception as e:
                    print(f"Warning: Could not convert {col} IDs to integers: {e}")
        
        return combined_gdf
    else:
        print("Warning: No data loaded, returning empty DataFrame")
        return pd.DataFrame()


def create_nodes_gdf_from_geojson_files(data_dir: str) -> 'pd.DataFrame':
    """
    Create a combined nodes GeoDataFrame from OSM and Mapillary geojson files.
    Applies filtering similar to clean_graph_attributes3.py to ensure data quality.
    
    Args:
        data_dir: Path to data directory containing the geojson files
        
    Returns:
        Combined GeoDataFrame with both OSM and Mapillary nodes
    """
    # Try different file naming conventions
    # Convention 1: Singapore-style (nodes_with_districts.geojson)
    osm_nodes_path = os.path.join(data_dir, 'nodes_with_districts.geojson')
    mapillary_nodes_path = os.path.join(data_dir, 'nodes_mapillary_with_districts.geojson')
    
    # # Convention 2: Generic style (nodes_all_add2.geojson)
    # if not os.path.exists(osm_nodes_path):
    #     osm_nodes_path = os.path.join(data_dir, 'nodes_all_add2.geojson')
    
    # if not os.path.exists(mapillary_nodes_path):
    #     mapillary_nodes_path = os.path.join(data_dir, 'nodes_mapillary.geojson')
    
    combined_gdf = load_and_concatenate_geodataframes(osm_nodes_path, mapillary_nodes_path)
    
    # Apply filtering similar to clean_graph_attributes3.py
    if combined_gdf is not None and len(combined_gdf) > 0:
        # Fill category from type if category is missing
        if 'category' in combined_gdf.columns and 'type' in combined_gdf.columns:
            combined_gdf['category'] = combined_gdf['category'].fillna(combined_gdf['type'])
        
        # Filter out nodes with null names (if name column exists)
        if 'name' in combined_gdf.columns:
            original_len = len(combined_gdf)
            combined_gdf = combined_gdf[(combined_gdf['name'].isnull()==False) | (combined_gdf['type'] == 'mapillary')]
            if len(combined_gdf) < original_len:
                print(f"⚠️ Filtered out {original_len - len(combined_gdf)} nodes with null names")
            
            # Filter out specific unwanted node types
            combined_gdf = combined_gdf[
                (combined_gdf['name'] != 'bike parking') & 
                (combined_gdf['name'] != 'seating')
            ]
            if len(combined_gdf) < original_len:
                print(f"⚠️ Filtered out additional nodes with unwanted names")
    
    return combined_gdf


def create_edges_gdf_from_geojson_files(data_dir: str) -> 'pd.DataFrame':
    """
    Create a combined edges GeoDataFrame from OSM and Mapillary geojson files.
    
    Args:
        data_dir: Path to data directory containing the geojson files
        
    Returns:
        Combined GeoDataFrame with both OSM and Mapillary edges
    """
    # Try different file naming conventions
    # Convention 1: Singapore-style (edges.geojson)
    osm_edges_path = os.path.join(data_dir, 'edges.geojson')
    
    # Convention 2: Generic style (edges_all_add1.geojson or parquet)
    if not os.path.exists(osm_edges_path):
        # Try parquet first
        # osm_edges_path = os.path.join(data_dir, 'edges_all_add1.parquet')
        # if not os.path.exists(osm_edges_path):
        #     # Fall back to geojson
        osm_edges_path = os.path.join(data_dir, 'edges_all_add1.geojson')
    
    mapillary_edges_path = os.path.join(data_dir, 'edges_mapillary.geojson')
    
    return load_and_concatenate_geodataframes(osm_edges_path, mapillary_edges_path)


def get_formatted_path_hash(data: dict) -> str:
    """
    Create a hash from formatted path string to identify duplicate paths.
    
    Args:
        data: JSONL item data
        
    Returns:
        Hash string of the formatted path, or None if path not found
    """
    try:
        if 'path_meta' not in data or 'paths_parsed' not in data['path_meta']:
            return None
        
        for path_data in data['path_meta']['paths_parsed']:
            # Try formatted_path_enhanced first, then formatted_path
            if 'formatted_path_enhanced' in path_data and path_data['formatted_path_enhanced']:
                return hash(path_data['formatted_path_enhanced'])
            elif 'formatted_path' in path_data and path_data['formatted_path']:
                return hash(path_data['formatted_path'])
        
        return None
    except Exception as e:
        print(f"Warning: Error extracting path hash: {e}")
        return None


def extract_image_coordinates(image_coordinates: str, mapillary_node: str, nodes_gdf=None) -> tuple:
    """
    Extract image coordinates from JSONL data or fallback to nodes_gdf.
    
    Args:
        image_coordinates: Image coordinates string from JSONL (e.g., "(103.8440, 1.2968)")
        mapillary_node: Mapillary node ID
        nodes_gdf: Optional GeoDataFrame to extract coordinates from if JSONL value is missing
        
    Returns:
        Tuple of (lat, lon) or None if coordinates cannot be extracted
    """
    image_coords = None
    
    # Try to parse from image_coordinates field
    if image_coordinates and image_coordinates not in ['', 'unknown coordinates', 'None']:
        try:
            # Extract coordinates from string like "(103.8440, 1.2968)"
            coords_str = image_coordinates.strip('()')
            lon, lat = map(float, coords_str.split(', '))
            image_coords = (lat, lon)  # Return as (lat, lon) tuple
            return image_coords
        except Exception as e:
            print(f"⚠️ Failed to parse image_coordinates from JSONL: {e}")
    
    # Fallback: Extract coordinates from Mapillary node in nodes_gdf
    if image_coords is None and mapillary_node and nodes_gdf is not None:
        try:
            mapillary_id_int = int(mapillary_node)
            mapillary_row = nodes_gdf[nodes_gdf['id'] == mapillary_id_int]
            
            if not mapillary_row.empty:
                row = mapillary_row.iloc[0]
                
                # Try to extract coordinates from geometry
                if hasattr(row, 'geometry') and row.geometry is not None:
                    try:
                        # Use centroid for all geometry types
                        if hasattr(row.geometry, 'centroid'):
                            centroid = row.geometry.centroid
                            lon = centroid.x
                            lat = centroid.y
                        elif hasattr(row.geometry, 'x') and hasattr(row.geometry, 'y'):
                            # For Point geometries
                            lon = row.geometry.x
                            lat = row.geometry.y
                        else:
                            raise ValueError("Cannot extract coordinates from geometry")
                        
                        image_coords = (lat, lon)
                        # Only print if successfully extracted from fallback
                        # print(f"✅ Extracted image coordinates from nodes_gdf geometry for {mapillary_id_int}: ({lat:.6f}, {lon:.6f})")
                        return image_coords
                    except Exception:
                        pass  # Silently try next method
                
                # Try lat/lon columns if geometry extraction failed
                if 'lat' in row and 'lon' in row and row['lat'] is not None and row['lon'] is not None:
                    lat = float(row['lat'])
                    lon = float(row['lon'])
                    image_coords = (lat, lon)
                    # Only print if successfully extracted from fallback
                    # print(f"✅ Extracted image coordinates from nodes_gdf lat/lon for {mapillary_id_int}: ({lat:.6f}, {lon:.6f})")
                    return image_coords
        
        except Exception:
            pass  # Silently fail - caller will handle missing coordinates
    
    # Don't print warning here - caller will skip the item and track it
    return image_coords


def create_training_data(jsonl_file_path: str, output_file_path: str, 
                        enhance_subgraphs: bool = True, 
                        compass_file_path: str = None,
                        data_folder: str = "singapore",
                        resume_from_line: int = 0,
                        extend_graph: bool = True) -> int:
    """
    Create enhanced training data with modified messages plus image_path, graph_path, and summarization.
    Also optionally enhance subgraphs with Mapillary nodes and text attributes.
    
    Args:
        jsonl_file_path: Path to input JSONL file
        output_file_path: Path to output training_data.jsonl file
        enhance_subgraphs: Whether to enhance subgraphs (if False, skip enhancement entirely)
        compass_file_path: Path to compass angles JSONL file (default: auto-constructed from data_folder)
        data_folder: Data folder name (e.g., 'singapore', 'newyork', 'beijing')
        resume_from_line: Line number to resume subgraph enhancement from (0-based)
        extend_graph: If True, add Mapillary node and path nodes to extend graph. If False, only create node_text attributes (requires enhance_subgraphs=True)
        
    Returns:
        Number of training examples processed
    """
    print(f"Creating enhanced training data from: {jsonl_file_path}")
    print(f"Data folder: {data_folder}")
    
    # Auto-construct compass file path if not provided
    if compass_file_path is None:
        compass_file_path = os.path.join(DATA_ROOT, 'geo', 'SR', 'osm_data', data_folder, 'mapillary_results_cleaned.jsonl')
        print(f"Using auto-constructed compass file path: {compass_file_path}")
    
    # Load compass angles
    print("Loading compass angles...")
    compass_angles = load_compass_angles(compass_file_path)
    
    # Load nodes and edges GeoDataFrames from geojson files (always load for coordinate extraction)
    data_dir = os.path.join(DATA_ROOT, 'geo', 'SR', 'osm_data', data_folder)
    print(f"Loading nodes and edges GeoDataFrames from geojson files ({data_dir})...")
    nodes_gdf = create_nodes_gdf_from_geojson_files(data_dir)
    edges_gdf = create_edges_gdf_from_geojson_files(data_dir)
    print(f"Loaded nodes GeoDataFrame with {len(nodes_gdf)} nodes")
    print(f"Loaded edges GeoDataFrame with {len(edges_gdf)} edges")
    
    # Enhance subgraphs if requested
    successful_graph_paths = set()
    if enhance_subgraphs:
        if extend_graph:
            print("Enhancing subgraphs with Mapillary nodes and text attributes...")
        else:
            print("Creating node_text attributes for subgraphs (without extending graph size)...")
        enhanced_count, successful_graph_paths = batch_enhance_subgraphs_from_jsonl(
            jsonl_file_path=jsonl_file_path,
            nodes_gdf=nodes_gdf,
            edges_gdf=edges_gdf,
            resume_from_line=resume_from_line,
            data_folder=data_folder,
            extend_graph=extend_graph
        )
        print(f"Enhanced {enhanced_count} subgraphs")
        print(f"Will only include {len(successful_graph_paths)} samples with successful subgraph enhancements in output JSONL")
    
    # Extract unique nodes for building place entities info
    print("Extracting unique nodes from JSONL for place entities info...")
    unique_nodes, mapillary_nodes = extract_unique_nodes_from_jsonl(jsonl_file_path, data_folder, nodes_gdf)
    print(f"Extracted {len(unique_nodes)} unique nodes")
    
    training_count = 0
    skipped_count = 0
    coordinates_corrected = 0  # Track how many unknown coordinates were fixed
    skipped_no_coordinates = 0  # Track items skipped due to missing coordinates
    skipped_duplicates = 0  # Track items skipped due to duplicate paths
    skipped_no_enhanced_subgraph = 0  # Track items skipped due to failed subgraph enhancement
    seen_path_hashes = set()  # Track unique path hashes to avoid duplicates
    
    with open(jsonl_file_path, 'r', encoding='utf-8') as infile, \
         open(output_file_path, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            try:
                data = json.loads(line.strip())
                
                # Skip items with invalid source_name (all digits)
                if not has_valid_source_name(data):
                    skipped_count += 1
                    if skipped_count % 100 == 0:  # Log every 100th skip to avoid spam
                        print(f"Skipped {skipped_count} items with invalid source_name (all digits)")
                    continue
                
                # Check for duplicate paths
                path_hash = get_formatted_path_hash(data)
                if path_hash is not None:
                    if path_hash in seen_path_hashes:
                        skipped_duplicates += 1
                        if skipped_duplicates % 100 == 0:  # Log every 100th duplicate to avoid spam
                            print(f"Skipped {skipped_duplicates} duplicate paths")
                        continue
                    else:
                        seen_path_hashes.add(path_hash)
                
                # Extract required fields
                messages = data.get('messages', [])
                image_path = data.get('image_path', '')
                graph_path = data.get('graph_path', '')
                summarization = data.get('summarization', '')
                image_coordinates_raw = data.get('image_coordinates', '')
                mapillary_node = data.get('mapillary_node', '')
                
                # Modify summarization to replace target node ID with target node name
                if summarization and 'path_meta' in data and 'paths_parsed' in data['path_meta']:
                    paths_parsed = data['path_meta']['paths_parsed']
                    if paths_parsed and len(paths_parsed) > 0:
                        last_path = paths_parsed[-1]
                        if 'path_triples' in last_path and len(last_path['path_triples']) > 0:
                            last_triple = last_path['path_triples'][-1]
                            if 'target_info' in last_triple and last_triple['target_info']:
                                target_id = str(last_triple['target_info'].get('id', ''))
                                target_name = last_triple['target_info'].get('name', '')
                                # Replace the target ID with target name in summarization
                                # Check that target_name is not NaN/None and is a valid string
                                if target_id and target_name and not pd.isna(target_name) and str(target_name).strip():
                                    target_name_str = str(target_name).strip()
                                    summarization = summarization.replace(target_id, target_name_str)
                
                # Extract and normalize image coordinates (with fallback to nodes_gdf)
                coords_tuple = extract_image_coordinates(image_coordinates_raw, mapillary_node, nodes_gdf)
                if coords_tuple:
                    lat, lon = coords_tuple
                    image_coordinates = f"({lon:.4f}, {lat:.4f})"  # Format as string for user message
                    
                    # Track if we corrected unknown coordinates
                    if image_coordinates_raw in ['', 'unknown coordinates', 'None']:
                        coordinates_corrected += 1
                else:
                    # Skip this item if we cannot find valid coordinates
                    skipped_no_coordinates += 1
                    if skipped_no_coordinates % 100 == 0:
                        print(f"Skipped {skipped_no_coordinates} items due to missing coordinates")
                    continue
                
                # Normalize both image_path and graph_path to server directory structure
                normalized_image_path = normalize_path_to_server(image_path)
                normalized_graph_path = normalize_path_to_server(graph_path)
                
                # Skip this sample if subgraph enhancement failed (if enhance_subgraphs was enabled)
                if enhance_subgraphs and successful_graph_paths and normalized_graph_path not in successful_graph_paths:
                    skipped_no_enhanced_subgraph += 1
                    if skipped_no_enhanced_subgraph % 10 == 0:
                        print(f"Skipped {skipped_no_enhanced_subgraph} samples with failed subgraph enhancements")
                    continue
                
                # Extract and clean spatial reasoning path from user content
                cleaned_spatial_path = ""
                if messages:
                    for message in messages:
                        if message.get('role') == 'user':
                            user_content = message.get('content', '')
                            # Find "Spatial Reasoning Path:" and extract the path
                            if "Spatial Reasoning Path:" in user_content:
                                parts = user_content.split("Spatial Reasoning Path:", 1)
                                if len(parts) == 2:
                                    # Extract the path part (until "Based on" or end)
                                    path_part = parts[1].split("Based on", 1)[0].strip()
                                    # Remove the trailing period if present
                                    if path_part.endswith('.'):
                                        path_part = path_part[:-1]
                                    # Clean the path
                                    cleaned_spatial_path = clean_formatted_path_enhanced(path_part)
                            break
                
                if not messages:
                    continue
                
                # Extract relevant nodes for this example
                relevant_nodes = {}
                if 'path_meta' in data and 'paths_parsed' in data['path_meta']:
                    for path_data in data['path_meta']['paths_parsed']:
                        if 'path_triples' in path_data:
                            for triple in path_data['path_triples']:
                                # Add source node
                                if 'source_info' in triple and triple['source_info']:
                                    source_id = str(triple['source_info'].get('id', ''))
                                    if source_id in unique_nodes:
                                        relevant_nodes[source_id] = unique_nodes[source_id]
                                
                                # Add target node
                                if 'target_info' in triple and triple['target_info']:
                                    target_id = str(triple['target_info'].get('id', ''))
                                    if target_id in unique_nodes:
                                        relevant_nodes[target_id] = unique_nodes[target_id]
                
                # Build place entities information
                place_entities_info = ""
                if relevant_nodes:
                    place_entities_info = build_place_entities_information(relevant_nodes)
                
                # Enhance messages (keep original format, just enhance user content)
                enhanced_messages = []
                for message in messages:
                    if message.get('role') == 'user':
                        # Enhance user content but keep all other fields
                        enhanced_message = message.copy()
                        original_content = message.get('content', '')
                        enhanced_content = enhance_user_message_content(
                            original_content,
                            image_coordinates,
                            place_entities_info,
                            mapillary_node,
                            cleaned_spatial_path,
                            compass_angles
                        )
                        enhanced_message['content'] = enhanced_content
                        enhanced_messages.append(enhanced_message)
                    else:
                        # Keep other messages exactly as is
                        enhanced_messages.append(message)
                
                # Create training example with normalized paths, renamed keys, and corrected coordinates
                training_example = {
                    'messages': enhanced_messages,
                    'images': normalized_image_path,  # Renamed from image_path
                    'graphs': normalized_graph_path,  # Renamed from graph_path
                    'summarization': summarization,
                    'image_coordinates': image_coordinates,  # Save corrected coordinates (not "unknown coordinates")
                    'mapillary_node': mapillary_node  # Keep mapillary_node for reference
                }
                
                # Write to output file
                outfile.write(json.dumps(training_example, ensure_ascii=False) + '\n')
                training_count += 1
                
                if training_count % 100 == 0:
                    print(f"Processed {training_count} training examples...")
                
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                continue
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                continue
    
    print(f"Created {training_count} enhanced training examples in: {output_file_path}")
    if skipped_count > 0:
        print(f"Skipped {skipped_count} items with invalid source_name (all digits)")
    if skipped_duplicates > 0:
        print(f"🔄 Skipped {skipped_duplicates} duplicate paths (based on formatted_path)")
    if skipped_no_coordinates > 0:
        print(f"Skipped {skipped_no_coordinates} items due to missing/invalid coordinates")
    if skipped_no_enhanced_subgraph > 0:
        print(f"⚠️ Skipped {skipped_no_enhanced_subgraph} items due to failed subgraph enhancement")
    if coordinates_corrected > 0:
        print(f"✅ Corrected {coordinates_corrected} unknown/missing coordinates using nodes_gdf")
    print(f"Processing summary: {training_count} processed, {skipped_count} skipped (invalid name), {skipped_duplicates} skipped (duplicates), {skipped_no_coordinates} skipped (no coordinates), {skipped_no_enhanced_subgraph} skipped (failed subgraph), {coordinates_corrected} coordinates corrected")
    print(f"✅ Unique paths retained: {len(seen_path_hashes)}")
    return training_count




def example_usage():
    # Example usage
    input_file = "data/geo/SR/osm_data/singapore/reasoning_path_mapillary_swift_qa3_no_intersection_nodes.jsonl"
    output_file = "data/geo/SR/osm_data/singapore/training_data.jsonl"
    
    # Create the enhanced training dataset
    training_count = create_training_data(
        jsonl_file_path=input_file,
        output_file_path=output_file
    )
    
    # # Export to DataFrames
    # training_df = export_to_dataframe(dataset)
    # nodes_df = create_nodes_dataframe(dataset)
    
    # # Analyze statistics
    # stats = analyze_dataset_statistics(dataset)
    # print("Dataset Statistics:")
    # for key, value in stats.items():
    #     print(f"{key}: {value}")
    
    # # Show DataFrame info
    # print(f"\nTraining DataFrame shape: {training_df.shape}")
    # print(f"Nodes DataFrame shape: {nodes_df.shape}")
    
    # # Example of using node_attributes_descriptions for a single node
    # if dataset['nodes']:
    #     sample_node_id = list(dataset['nodes'].keys())[0]
    #     sample_node = dataset['nodes'][sample_node_id]
    #     attributes = build_node_attributes_descriptions(sample_node)
    #     print(f"\nSample node attributes for {sample_node_id}:")
    #     print(json.dumps(attributes, indent=2))
    
    return training_count


def enhance_graph_with_node_attributes(graph, unique_nodes: Dict[str, Dict[str, Any]]) -> None:
    """
    Enhance NetworkX graph nodes with processed node attributes.
    
    Args:
        graph: NetworkX graph object
        unique_nodes: Dictionary of unique nodes with their data
    """
    for node_id in graph.nodes():
        node_id_str = str(node_id)
        if node_id_str in unique_nodes:
            node_data = unique_nodes[node_id_str]
            
            # Build node attributes
            attrs = build_node_attributes_descriptions(node_data)
            
            # Add processed attributes to the graph node
            graph.nodes[node_id]['processed_name'] = attrs['name']
            graph.nodes[node_id]['processed_category'] = attrs['category'] 
            graph.nodes[node_id]['processed_coordinates'] = attrs['coordinates']
            graph.nodes[node_id]['original_geometry'] = node_data.get('geometry', '')
            graph.nodes[node_id]['osm_id'] = node_data.get('osm_id', '')
            graph.nodes[node_id]['node_source'] = node_data.get('source', 'unknown')


def save_enhanced_graph(graph, unique_nodes: Dict[str, Dict[str, Any]], output_path: str) -> None:
    """
    Save NetworkX graph with enhanced node attributes.
    
    Args:
        graph: NetworkX graph object
        unique_nodes: Dictionary of unique nodes with their data
        output_path: Path to save the enhanced graph
    """
    import pickle
    import os
    
    # Enhance the graph with node attributes
    enhance_graph_with_node_attributes(graph, unique_nodes)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the enhanced graph
    with open(output_path, 'wb') as f:
        pickle.dump(graph, f)
    
    print(f"Enhanced graph saved to: {output_path}")


def load_and_enhance_existing_graphs(jsonl_file_path: str, graph_directory: str, data_folder: str = "singapore", nodes_gdf=None) -> int:
    """
    Load existing graph.pkl files and enhance them with node attributes.
    
    Args:
        jsonl_file_path: Path to JSONL file containing the training data
        graph_directory: Directory containing graph.pkl files to enhance
        data_folder: Data folder name (e.g., 'singapore', 'newyork')
        nodes_gdf: Optional GeoDataFrame with node information for attribute lookup
        
    Returns:
        Number of graphs enhanced
    """
    import pickle
    import os
    from glob import glob
    
    print(f"Loading node data from: {jsonl_file_path}")
    
    # Extract unique nodes
    unique_nodes, mapillary_nodes = extract_unique_nodes_from_jsonl(jsonl_file_path, data_folder, nodes_gdf)
    print(f"Extracted {len(unique_nodes)} unique nodes")
    
    # Find all .pkl files in the directory
    pkl_files = glob(os.path.join(graph_directory, "**/*.pkl"), recursive=True)
    print(f"Found {len(pkl_files)} .pkl files to enhance")
    
    enhanced_count = 0
    
    for pkl_file in pkl_files:
        try:
            print(f"Processing: {pkl_file}")
            
            # Load the graph
            with open(pkl_file, 'rb') as f:
                graph = pickle.load(f)
            
            # Check if it's a NetworkX graph
            if hasattr(graph, 'nodes') and hasattr(graph, 'edges'):
                # Enhance with node attributes
                enhance_graph_with_node_attributes(graph, unique_nodes)
                
                # Save the enhanced graph in new directory with same filename
                pkl_dir = os.path.dirname(pkl_file)
                pkl_filename = os.path.basename(pkl_file)
                enhanced_dir = os.path.join(pkl_dir, ENHANCED_SUBGRAPH_DIR_NAME)
                os.makedirs(enhanced_dir, exist_ok=True)
                enhanced_pkl_file = os.path.join(enhanced_dir, pkl_filename)
                
                with open(enhanced_pkl_file, 'wb') as f:
                    pickle.dump(graph, f)
                
                enhanced_count += 1
                print(f"✅ Enhanced graph saved as: {enhanced_pkl_file}")
            else:
                print(f"⚠️ Not a NetworkX graph, skipping: {pkl_file}")
                
        except Exception as e:
            print(f"❌ Error processing {pkl_file}: {e}")
            continue
    
    print(f"Enhanced {enhanced_count} graphs with node attributes")
    return enhanced_count


def add_mapillary_node_with_neighbors(graph, mapillary_id: str, nodes_gdf, hop_distance: int = 2, image_coords=None):
    """
    Add a Mapillary node and its neighbors to the subgraph.
    
    Args:
        graph: NetworkX graph object
        mapillary_id: Mapillary node ID to add (will be converted to int)
        nodes_gdf: GeoDataFrame containing node information
        hop_distance: Number of hops to include neighbors (default: 2)
        image_coords: Optional tuple of (lat, lon) for creating Mapillary node if not in nodes_gdf
        
    Returns:
        Updated graph with Mapillary node and neighbors
    """
    import networkx as nx
    
    # Convert mapillary_id to integer for consistency
    try:
        mapillary_id_int = int(mapillary_id)
    except (ValueError, TypeError):
        print(f"⚠️ Cannot convert Mapillary ID {mapillary_id} to integer")
        return graph
    
    # Try to find the Mapillary node in nodes_gdf (search by integer ID)
    mapillary_row = nodes_gdf[nodes_gdf['id'] == mapillary_id_int]
    
    if mapillary_row.empty:
        # ✅ FIX: Mapillary nodes (street-view images) are NOT in OSM nodes_gdf!
        # Create the node from image_coords instead
        print(f"⚠️ Mapillary node {mapillary_id_int} not found in nodes_gdf")
        
        if image_coords is None:
            print(f"❌ Cannot add Mapillary node: no image_coords provided")
            return graph
        
        # Create Mapillary node from image coordinates
        lat, lon = image_coords
        mapillary_node_data = {
            'id': mapillary_id_int,
            'category': 'mapillary',
            'lat': lat,
            'lon': lon,
            'geometry': f'POINT({lon} {lat})'
        }
        print(f"✅ Creating Mapillary node {mapillary_id_int} from image coordinates ({lat:.6f}, {lon:.6f})")
    else:
        mapillary_node_data = mapillary_row.iloc[0].to_dict()
        print(f"✅ Found Mapillary node {mapillary_id_int} in nodes_gdf")
    
    # Add the Mapillary node to the graph if not already present (use integer ID)
    if mapillary_id_int not in graph.nodes():
        graph.add_node(mapillary_id_int, **mapillary_node_data)
        print(f"✅ Added Mapillary node {mapillary_id_int} to graph")
    
    # Find nodes within hop_distance of existing graph nodes
    # This assumes the original graph represents the spatial connectivity
    existing_nodes = set(graph.nodes())
    nodes_to_add = set()
    
    # For each existing node, find its neighbors up to hop_distance
    for existing_node in existing_nodes:
        try:
            # Use NetworkX to find nodes within hop distance
            if existing_node in graph:
                neighbors = nx.single_source_shortest_path_length(graph, existing_node, cutoff=hop_distance)
                nodes_to_add.update(neighbors.keys())
        except:
            continue
    
    # Add any missing nodes from nodes_gdf (IDs should already be integers)
    added_count = 0
    for node_id in nodes_to_add:
        if node_id not in graph.nodes():
            # Find node in GDF by integer ID
            node_row = nodes_gdf[nodes_gdf['id'] == node_id]
            
            if not node_row.empty:
                node_data = node_row.iloc[0]
                graph.add_node(node_id, **node_data.to_dict())
                added_count += 1
    
    print(f"✅ Added {added_count} nodes within {hop_distance} hops")
    return graph


def add_mapillary_node_with_direct_edge(graph, mapillary_id: str, nodes_gdf, image_coords=None):
    """
    Add a Mapillary node to the graph and connect it with a direct edge to the nearest existing node.
    This is a minimal extension that only adds the Mapillary node itself, not neighbors.
    
    Args:
        graph: NetworkX graph object
        mapillary_id: Mapillary node ID to add (will be converted to int)
        nodes_gdf: GeoDataFrame containing node information
        image_coords: Optional tuple of (lat, lon) for creating Mapillary node if not in nodes_gdf
        
    Returns:
        Updated graph with Mapillary node and direct edge to nearest node
    """
    import networkx as nx
    from geopy.distance import geodesic
    
    # Convert mapillary_id to integer for consistency
    try:
        mapillary_id_int = int(mapillary_id)
    except (ValueError, TypeError):
        print(f"⚠️ Cannot convert Mapillary ID {mapillary_id} to integer")
        return graph
    
    # Check if Mapillary node already exists
    if mapillary_id_int in graph.nodes():
        print(f"ℹ️ Mapillary node {mapillary_id_int} already exists in graph")
        return graph
    
    # Try to find the Mapillary node in nodes_gdf (search by integer ID)
    mapillary_row = nodes_gdf[nodes_gdf['id'] == mapillary_id_int]
    
    if mapillary_row.empty:
        # Mapillary nodes (street-view images) are NOT in OSM nodes_gdf!
        # Create the node from image_coords instead
        print(f"⚠️ Mapillary node {mapillary_id_int} not found in nodes_gdf")
        
        if image_coords is None:
            print(f"❌ Cannot add Mapillary node: no image_coords provided")
            return graph
        
        # Create Mapillary node from image coordinates
        lat, lon = image_coords
        mapillary_node_data = {
            'id': mapillary_id_int,
            'category': 'mapillary',
            'lat': lat,
            'lon': lon,
            'geometry': f'POINT({lon} {lat})'
        }
        mapillary_coords = (lat, lon)
        print(f"✅ Creating Mapillary node {mapillary_id_int} from image coordinates ({lat:.6f}, {lon:.6f})")
    else:
        mapillary_node_data = mapillary_row.iloc[0].to_dict()
        # Extract coordinates from geometry
        geom = mapillary_row.iloc[0].geometry
        if hasattr(geom, 'x') and hasattr(geom, 'y'):
            mapillary_coords = (geom.y, geom.x)  # (lat, lon)
        else:
            if image_coords:
                mapillary_coords = image_coords
            else:
                print(f"⚠️ Cannot extract coordinates for Mapillary node {mapillary_id_int}")
                return graph
        print(f"✅ Found Mapillary node {mapillary_id_int} in nodes_gdf")
    
    # Add the Mapillary node to the graph
    graph.add_node(mapillary_id_int, **mapillary_node_data)
    print(f"✅ Added Mapillary node {mapillary_id_int} to graph")
    
    # Find the nearest existing node in the graph
    nearest_node = None
    min_distance = float('inf')
    
    for node_id in graph.nodes():
        if node_id == mapillary_id_int:
            continue
        
        # Get node coordinates from nodes_gdf
        node_row = nodes_gdf[nodes_gdf['id'] == node_id]
        if not node_row.empty:
            node_geom = node_row.iloc[0].geometry
            if hasattr(node_geom, 'x') and hasattr(node_geom, 'y'):
                node_coords = (node_geom.y, node_geom.x)  # (lat, lon)
            else:
                # Try to get from node attributes
                node_attrs = graph.nodes[node_id]
                if 'lat' in node_attrs and 'lon' in node_attrs:
                    node_coords = (node_attrs['lat'], node_attrs['lon'])
                else:
                    continue
        else:
            # Try to get from node attributes
            node_attrs = graph.nodes[node_id]
            if 'lat' in node_attrs and 'lon' in node_attrs:
                node_coords = (node_attrs['lat'], node_attrs['lon'])
            else:
                continue
        
        # Calculate distance using geodesic
        try:
            distance = geodesic(mapillary_coords, node_coords).meters
            if distance < min_distance:
                min_distance = distance
                nearest_node = node_id
        except Exception as e:
            print(f"⚠️ Error calculating distance to node {node_id}: {e}")
            continue
    
    # Add direct edge to nearest node
    if nearest_node is not None:
        # Calculate edge weight (distance in meters)
        edge_weight = min_distance
        
        # Add edge with weight attribute
        graph.add_edge(mapillary_id_int, nearest_node, 
                      weight=edge_weight,
                      type='mapillary_connection',
                      distance=edge_weight)
        print(f"✅ Added direct edge from Mapillary node {mapillary_id_int} to nearest node {nearest_node} (distance: {min_distance:.2f}m)")
    else:
        print(f"⚠️ Warning: Could not find nearest node to connect Mapillary node {mapillary_id_int}")
    
    return graph


def extract_coordinates_from_geometry(geometry):
    """
    Extract (lon, lat) coordinates from geometry object.
    
    Handles Point, LineString, Polygon, and other geometry types by using centroid.
    For LineString geometries, uses the centroid of the line.
    
    Args:
        geometry: Shapely geometry object
        
    Returns:
        Tuple of (lon, lat) or None if extraction fails
    """
    import re
    if geometry is None:
        return None

    try:
        # Check if geometry is empty or invalid
        if hasattr(geometry, 'is_empty') and geometry.is_empty:
            return None

        # Method 1: Try to get centroid (works for Point, LineString, Polygon, etc.)
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


def build_node_id_to_row_lookup(nodes_gdf):
    """
    Build a fast lookup dictionary mapping node_id to row data.
    This should be built once and reused for multiple graphs.
    
    Args:
        nodes_gdf: GeoDataFrame containing node information
        
    Returns:
        Dictionary mapping node_id (as string) -> row data (pandas Series)
    """
    node_id_to_row = {}
    for idx, row in nodes_gdf.iterrows():
        node_id_str = str(row.get('id', ''))
        if node_id_str:
            # Store both string and integer versions for flexibility
            node_id_to_row[node_id_str] = row
            try:
                node_id_int = int(row.get('id'))
                if str(node_id_int) != node_id_str:
                    node_id_to_row[str(node_id_int)] = row
            except (ValueError, TypeError):
                pass
    return node_id_to_row


def create_node_text_attribute(graph, nodes_gdf, image_coords=None, data_folder="singapore", node_id_to_row=None,mapillary_id=None):
    """
    Create 'node_text' and 'coords' attributes for all nodes in the graph.
    Removes nodes that are not found in nodes_gdf.
    
    Optimized version using dictionary indexing for fast node lookups.
    Based on clean_graph_attributes3.py implementation.
    
    Args:
        graph: NetworkX graph object (node IDs should be integers)
        nodes_gdf: GeoDataFrame containing node information
        image_coords: Optional tuple of (lat, lon) for image location (deprecated, kept for compatibility)
        data_folder: Data folder name (e.g., 'singapore', 'newyork')
        node_id_to_row: Optional pre-built lookup dictionary (if None, will build it)
    """
    import pandas as pd
    import networkx as nx
    
    # Build lookup dictionary if not provided (for backward compatibility)
    if node_id_to_row is None:
        node_id_to_row = build_node_id_to_row_lookup(nodes_gdf)
    
    # Collect nodes to remove (nodes not found in nodes_gdf)
    nodes_to_remove = []
    
    # Process all nodes
    for node_id in list(graph.nodes()):  # Use list() to avoid modification during iteration
        node_id_str = str(node_id)
        
        # Fast lookup using dictionary - get row directly
        row = node_id_to_row.get(node_id_str)
        
        if row is not None:

            # Start with basic info
            node_name = 'mapillary image' if row.get('type') == 'mapillary' else row.get('name', 'unknown')
            node_text = f"Name: {node_name}, "
            node_text += f"ID: {row.get('id', 'unknown')}, "
            node_text += f"Category: {row.get('category', row.get('type', 'unknown'))}, "
            node_text += f"Address: {row.get('address', 'unknown')}, "

            # Add location-specific fields based on data_folder
            if data_folder == "newyork":
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

            # Add rich attributes (optional, for completeness)
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
                if value is not None and pd.notna(value) and str(value).strip() != "" and str(value).lower() not in [
                    'unknown', 'none', 'null']:
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
            # Node not found in GeoDataFrame - mark for removal
            # if node_id != mapillary_id:
                nodes_to_remove.append(node_id)

    # Remove nodes that were not found in nodes_gdf
    # nodes_to_remove = []
    if nodes_to_remove:
        graph.remove_nodes_from(nodes_to_remove)
        print(f"⚠️ Removed {len(nodes_to_remove)} nodes not found in nodes_gdf")


def enhance_subgraph_with_mapillary_and_text(pkl_file_path: str, mapillary_id: str, 
                                           nodes_gdf=None, edges_gdf=None, image_coords=None, path_nodes=None, 
                                           output_path: str = None, data_folder: str = "singapore",
                                           node_id_to_row=None, extend_graph: bool = True) -> bool:
    """
    Load subgraph pkl, optionally add Mapillary node and neighbors, and create node_text attributes.
    For large subgraphs (>200 nodes), downsize by keeping only path nodes and their 3-hop neighbors.
    
    Args:
        pkl_file_path: Path to the subgraph pkl file
        mapillary_id: Mapillary node ID to add
        nodes_gdf: GeoDataFrame containing node information
        edges_gdf: Optional GeoDataFrame containing edge information
        image_coords: Optional tuple of (lat, lon) for image location
        path_nodes: Optional set of nodes that appear in the path (for downsizing)
        output_path: Optional path to save enhanced graph (default: overwrite original)
        data_folder: Data folder name (e.g., 'singapore', 'newyork')
        node_id_to_row: Optional pre-built lookup dictionary (if None, will build it)
        extend_graph: If True, add Mapillary node and path nodes to extend graph. If False, only create node_text attributes.
    """
    import pickle
    import os
    
    print(f"Loading subgraph from: {pkl_file_path}")
    
    try:
        # Load the graph
        with open(pkl_file_path, 'rb') as f:
            graph = pickle.load(f)

        print(f"Original graph has {len(graph.nodes())} nodes and {len(graph.edges())} edges")

        # Convert all node IDs to integers for consistency
        import networkx as nx

        node_mapping = {}
        nodes_converted = 0

        for node in graph.nodes():
            try:
                # Try to convert to integer
                node_int = int(node)
                if node != node_int:  # Only add to mapping if conversion needed
                    node_mapping[node] = node_int
                    nodes_converted += 1
            except (ValueError, TypeError):
                # If conversion fails, keep original (should not happen)
                print(f"⚠️ Warning: Cannot convert node {node} to integer, keeping as-is")

        # Relabel nodes if any conversions were needed
        if node_mapping:
            graph = nx.relabel_nodes(graph, node_mapping, copy=False)
            print(f"✅ Converted {nodes_converted} node IDs to integers")
        else:
            print(f"✅ All node IDs already integers")

        print(f"Graph after ID conversion: {len(graph.nodes())} nodes and {len(graph.edges())} edges")

        # Convert mapillary_id to integer for consistency
        try:
            mapillary_id_int = int(mapillary_id)
        except (ValueError, TypeError):
            print(f"⚠️ Cannot convert Mapillary ID {mapillary_id} to integer")
            return False

        # Graph extension: add path nodes and Mapillary node (only if extend_graph=True)
        if extend_graph:
            # Downsize large subgraphs (>200 nodes) by keeping only path nodes + 3-hop neighbors
            # if path_nodes and len(graph.nodes()) > 200:
            #     print(f"Large subgraph detected ({len(graph.nodes())} nodes). Downsizing to path nodes + 3-hop neighbors...")
            #     graph = downsize_large_subgraph(graph, path_nodes, max_hops=2)

            # Add missing path nodes with their 2-hop neighbors (respecting 200 node limit)
            # Final verification: Check if all path nodes are in the subgraph
            if path_nodes:
                missing_path_nodes = path_nodes - set(graph.nodes())
                if missing_path_nodes:
                    print(f"⚠️ WARNING: {len(missing_path_nodes)} path nodes still missing from subgraph!")
                    print(f"Missing nodes: {missing_path_nodes}")
                    # Don't fail, just warn (they might be missing due to 200 node limit)
                else:
                    print(f"✅ Verified: All {len(path_nodes)} path nodes are present in the enhanced subgraph")
            if path_nodes:
                max_graph_nodes = get_max_graph_nodes_limit(data_folder)
                print(f"Applying graph pruning limit of {max_graph_nodes} nodes for data folder '{data_folder}'")
                graph = add_missing_path_nodes_to_subgraph(
                    graph,
                    path_nodes,
                    nodes_gdf,
                    edges_gdf,
                    max_total_nodes=max_graph_nodes
                )

            # Check if Mapillary node already exists in the subgraph (check both original and integer version)
            mapillary_exists = str(mapillary_id) in graph.nodes() or mapillary_id_int in graph.nodes()
            print(f"Mapillary node {mapillary_id_int} {'already exists' if mapillary_exists else 'not found'} in subgraph")

            # Only add Mapillary node and neighbors if it doesn't exist
            if not mapillary_exists:
                print(f"Adding Mapillary node {mapillary_id_int} with 2-hop neighbors...")
                graph = add_mapillary_node_with_neighbors(graph, mapillary_id, nodes_gdf, hop_distance=2, image_coords=image_coords)
                print(f"Graph after adding Mapillary node: {len(graph.nodes())} nodes and {len(graph.edges())} edges")
            else:
                print(f"Skipping Mapillary node addition - already exists in subgraph")
        else:
            print("⏭️ Skipping graph extension (extend_graph=False) - will only create node_text attributes")
            # Check if Mapillary node exists in original graph
            mapillary_exists_in_original = str(mapillary_id) in graph.nodes() or mapillary_id_int in graph.nodes()
            if mapillary_exists_in_original:
                print(f"ℹ️ Mapillary node {mapillary_id_int} exists in original graph (will be included in node_text)")
            else:
                print(f"ℹ️ Mapillary node {mapillary_id_int} not found in original graph - adding it with direct edge to nearest node")
                # Add Mapillary node with direct edge to nearest node (minimal extension)
                graph = add_mapillary_node_with_direct_edge(graph, mapillary_id, nodes_gdf, image_coords)

        # Create node_text attributes for all nodes (always do this)
        print("Creating node_text attributes for all nodes...")
        create_node_text_attribute(graph, nodes_gdf, image_coords, data_folder, node_id_to_row,mapillary_id)

        print(f"Final enhanced graph has {len(graph.nodes())} nodes and {len(graph.edges())} edges")

        # Final verification: Check if Mapillary node is in the subgraph before saving
        if extend_graph:
            # When extending graph, Mapillary node MUST exist after enhancement
            if mapillary_id_int not in graph.nodes():
                print(f"⚠️ WARNING: Mapillary node {mapillary_id_int} still not in subgraph after enhancement!")
                print(f"❌ Skipping save - subgraph enhancement failed")
                print(f"   Likely cause: Mapillary node not found in nodes_gdf")
                print(f"   This sample will be skipped (not saved to enhanced directory or output JSONL)")
                # Don't save this subgraph - return False to indicate failure
                return False
            else:
                print(f"✅ Verified: Mapillary node {mapillary_id_int} is present in the enhanced subgraph")
        else:
            # When not extending graph, we don't require Mapillary node to exist
            # (it's optional - node_text will be created for whatever nodes exist)
            print(f"⏭️ Mapillary node verification skipped (extend_graph=False) - proceeding to save graph with node_text attributes")

        # Save the enhanced graph to fixed output directory
        pkl_filename = os.path.basename(pkl_file_path)
        fixed_output_dir = os.path.join(OUTPUT_ROOT, 'multiview_three_pairs', 'subgraphs')
        os.makedirs(fixed_output_dir, exist_ok=True)
        output_file = os.path.join(fixed_output_dir, pkl_filename)

        with open(output_file, 'wb') as f:
            pickle.dump(graph, f)

        print(f"✅ Enhanced subgraph saved to: {output_file}")
        return True  # Success
        
    except Exception as e:
        print(f"❌ Error enhancing subgraph {pkl_file_path}: {e}")
        return False  # Failure


def normalize_path_to_server(path: str) -> str:
    """
    Normalize any path to use DATA_ROOT or OUTPUT_ROOT as the base directory.
    
    Converts paths like:
    - "/root/lanyun-fs/UrbanKG/data/..." -> "{DATA_ROOT}/..."
    - "./data/geo/SR/..." -> "{DATA_ROOT}/geo/SR/..."
    - "data/geo/SR/..." -> "{DATA_ROOT}/geo/SR/..."
    
    Args:
        path: Original path from JSONL
        
    Returns:
        Normalized path for server system
    """
    if not path:
        return path
    
    # Remove leading "./" if present
    if path.startswith("./"):
        path = path[2:]
    
    # Replace /root/lanyun-fs/ with DATA_ROOT
    if "/root/lanyun-fs/" in path:
        if "/UrbanKG/data/" in path:
            rel_path = path.split("/UrbanKG/data/")[-1]
            path = os.path.join(DATA_ROOT, rel_path)
        else:
            path = path.replace("/root/lanyun-fs/", "")
            if not path.startswith(DATA_ROOT) and not path.startswith(OUTPUT_ROOT):
                path = os.path.join(DATA_ROOT, path)
    
    # If path starts with "data/geo/SR/" or "data/", prepend DATA_ROOT
    if path.startswith("data/"):
        return os.path.join(DATA_ROOT, path[5:])  # Skip "data/" prefix
    
    # Extract the relative path from "data/geo/SR/..." onwards if it's in the middle
    if "data/geo/SR/" in path:
        relative_part = path.split("data/geo/SR/", 1)[1]
        normalized_path = os.path.join(DATA_ROOT, "geo", "SR", relative_part)
        return normalized_path
    
    # Handle old /home/xingtong/ paths
    if "/home/xingtong/" in path:
        if "/UrbanKG/data/" in path:
            rel_path = path.split("/UrbanKG/data/")[-1]
            path = os.path.join(DATA_ROOT, rel_path)
        elif "/ms_swift/mydata/" in path:
            rel_path = path.split("/ms_swift/mydata/")[-1]
            path = os.path.join(OUTPUT_ROOT, rel_path)
    
    # If it already starts with DATA_ROOT or OUTPUT_ROOT, return as-is
    if path.startswith(DATA_ROOT) or path.startswith(OUTPUT_ROOT):
        return path
    
    # Default: return as-is if no pattern matches
    return path


def normalize_graph_path(graph_path: str) -> str:
    """
    Normalize graph_path from JSONL to local directory structure.
    Wrapper around normalize_path_to_server for backward compatibility.
    
    Args:
        graph_path: Original graph path from JSONL
        
    Returns:
        Normalized path for local system
    """
    return normalize_path_to_server(graph_path)

def get_path_nodes_from_data(data: dict) -> set:
    """
    Extract all nodes that appear in paths from JSONL data. 
    Checks both 'path_meta.paths_parsed.path' and 'all_paths.paths.nodes'.
    Returns integer node IDs.
    """
    path_nodes = set()
    
    try:
        # Extract from path_meta.paths_parsed.path
        if 'path_meta' in data and 'paths_parsed' in data['path_meta']:
            for path_data in data['path_meta']['paths_parsed']:
                if 'path' in path_data:
                    for node in path_data['path']:
                        # Convert to integer for consistency with graph node IDs
                        try:
                            path_nodes.add(int(node))
                        except (ValueError, TypeError):
                            # If conversion fails, skip this node
                            print(f"Warning: Cannot convert path node {node} to integer")
                            continue
        
        # Extract from all_paths.paths.nodes (the comprehensive list)
        if 'all_paths' in data and 'paths' in data['all_paths']:
            for path_data in data['all_paths']['paths']:
                if 'nodes' in path_data:
                    for node in path_data['nodes']:
                        # Convert to integer for consistency with graph node IDs
                        try:
                            path_nodes.add(int(node))
                        except (ValueError, TypeError):
                            # If conversion fails, skip this node
                            print(f"Warning: Cannot convert all_paths node {node} to integer")
                            continue
    
    except Exception as e:
        print(f"Warning: Error extracting path nodes: {e}")
    
    return path_nodes

def add_missing_path_nodes_to_subgraph(graph, path_nodes: set, nodes_gdf, edges_gdf=None, 
                                       max_total_nodes: int = 200) -> 'networkx.Graph':
    """
    Add missing path nodes to subgraph with their 2-hop neighbors.
    Respects the maximum node count limit.
    
    Args:
        graph: NetworkX graph
        path_nodes: Set of integer node IDs that should be in the path
        nodes_gdf: GeoDataFrame containing node information
        edges_gdf: Optional GeoDataFrame containing edge information
        max_total_nodes: Maximum total nodes allowed in the graph
        
    Returns:
        Updated NetworkX graph with path nodes added
    """
    import networkx as nx
    
    # Find missing path nodes
    missing_path_nodes = path_nodes - set(graph.nodes())
    
    if not missing_path_nodes:
        print(f"✅ All {len(path_nodes)} path nodes already exist in subgraph")
        return graph
    
    print(f"Found {len(missing_path_nodes)} missing path nodes: {missing_path_nodes}")
    
    # Calculate available space for new nodes
    current_node_count = len(graph.nodes())
    available_space = max_total_nodes - current_node_count
    
    if available_space <= 0:
        print(f"⚠️ Graph already at max capacity ({current_node_count} nodes). Cannot add missing path nodes.")
        return graph
    
    print(f"Available space for new nodes: {available_space} (current: {current_node_count}, max: {max_total_nodes})")
    
    # Build a temporary full graph from edges_gdf to find neighbors
    # This is needed to find 2-hop neighbors of missing nodes
    temp_full_graph = None
    if edges_gdf is not None and not edges_gdf.empty:
        try:
            temp_full_graph = nx.Graph()
            # Add edges from edges_gdf
            for _, edge in edges_gdf.iterrows():
                source = edge.get('source')
                target = edge.get('target')
                if source is not None and target is not None:
                    temp_full_graph.add_edge(int(source), int(target))
            print(f"Built temporary graph with {len(temp_full_graph.nodes())} nodes for neighbor lookup")
        except Exception as e:
            print(f"Warning: Could not build temporary graph from edges: {e}")
    
    # Collect nodes to add: missing path nodes + their 2-hop neighbors
    nodes_to_add = set(missing_path_nodes)
    
    # Add 2-hop neighbors if we have the full graph
    if temp_full_graph:
        for missing_node in missing_path_nodes:
            if missing_node in temp_full_graph.nodes():
                try:
                    # Get 2-hop neighbors
                    neighbors = nx.single_source_shortest_path_length(
                        temp_full_graph, missing_node, cutoff=2
                    )
                    nodes_to_add.update(neighbors.keys())
                except:
                    # If node not in temp graph, just add the node itself
                    continue
    
    # Limit to available space
    nodes_to_add = list(nodes_to_add)[:available_space]
    
    print(f"Adding {len(nodes_to_add)} nodes (including 2-hop neighbors)")
    
    # Add nodes to graph with their attributes from nodes_gdf
    added_count = 0
    for node_id in nodes_to_add:
        if node_id not in graph.nodes():
            # Find node in GDF
            node_row = nodes_gdf[nodes_gdf['id'] == node_id]
            
            if not node_row.empty:
                node_data = node_row.iloc[0]
                graph.add_node(node_id, **node_data.to_dict())
                added_count += 1
            else:
                # If not in GDF, add with minimal attributes
                graph.add_node(node_id, id=node_id)
                added_count += 1
    
    print(f"✅ Added {added_count} nodes to subgraph (now {len(graph.nodes())} total)")
    
    # Add edges between the new nodes from edges_gdf
    if edges_gdf is not None and not edges_gdf.empty:
        edges_added = 0
        nodes_in_graph = set(graph.nodes())
        
        for _, edge in edges_gdf.iterrows():
            source = edge.get('source')
            target = edge.get('target')
            
            if source is not None and target is not None:
                source_int = int(source)
                target_int = int(target)
                
                # Only add edge if both nodes are in the subgraph
                if source_int in nodes_in_graph and target_int in nodes_in_graph:
                    if not graph.has_edge(source_int, target_int):
                        # Add edge with all attributes from edges_gdf
                        edge_attrs = edge.to_dict()
                        graph.add_edge(source_int, target_int, **edge_attrs)
                        edges_added += 1
        
        print(f"✅ Added {edges_added} edges to connect the new nodes")
    
    # Verify all critical path nodes are now in graph
    still_missing = path_nodes - set(graph.nodes())
    if still_missing:
        print(f"⚠️ Warning: {len(still_missing)} path nodes still missing due to space constraints: {still_missing}")
    else:
        print(f"✅ All {len(path_nodes)} path nodes now in subgraph")
    
    return graph


def downsize_large_subgraph(graph, path_nodes: set, max_hops: int = 2) -> 'networkx.Graph':
    """
    Downsize large subgraphs by keeping only path nodes and their k-hop neighbors.
    
    Args:
        graph: NetworkX graph
        path_nodes: Set of nodes that appear in the path
        max_hops: Maximum number of hops to include around path nodes
        
    Returns:
        Downsized NetworkX graph
    """
    import networkx as nx
    
    nodes_to_keep = set()
    
    # Add all path nodes
    for node in path_nodes:
        if node in graph.nodes():
            nodes_to_keep.add(node)
    
    # Add k-hop neighbors of path nodes
    for node in path_nodes:
        if node in graph.nodes():
            try:
                # Get k-hop neighbors
                for hop in range(1, max_hops + 1):
                    hop_neighbors = set()
                    if hop == 1:
                        hop_neighbors = set(graph.neighbors(node))
                    else:
                        # Get neighbors of previous hop neighbors
                        prev_hop_nodes = [n for n in nodes_to_keep 
                                        if nx.shortest_path_length(graph, node, n) == hop - 1]
                        for prev_node in prev_hop_nodes:
                            hop_neighbors.update(graph.neighbors(prev_node))
                    
                    # Filter to only nodes at exactly this hop distance
                    for neighbor in hop_neighbors:
                        try:
                            if nx.shortest_path_length(graph, node, neighbor) == hop:
                                nodes_to_keep.add(neighbor)
                        except nx.NetworkXNoPath:
                            continue
            except Exception as e:
                print(f"Warning: Error getting neighbors for node {node}: {e}")
                continue
    
    # Create subgraph with selected nodes
    downsized_graph = graph.subgraph(nodes_to_keep).copy()
    print(f"Downsized graph from {len(graph.nodes())} to {len(downsized_graph.nodes())} nodes")
    
    return downsized_graph

def batch_enhance_subgraphs_from_jsonl(jsonl_file_path: str, nodes_gdf, edges_gdf=None,
                                      subgraph_directory: str = None, resume_from_line: int = 0,
                                      data_folder: str = "singapore", extend_graph: bool = True) -> tuple[int, set]:
    """
    Batch enhance subgraphs based on training data from JSONL file.
    Includes resume functionality, deduplication, and subgraph downsizing.
    
    Args:
        jsonl_file_path: Path to JSONL file containing training data
        nodes_gdf: GeoDataFrame containing node information
        edges_gdf: Optional GeoDataFrame containing edge information
        subgraph_directory: Optional directory containing subgraphs (extracted from JSONL if None)
        resume_from_line: Line number to resume from (0-based)
        data_folder: Data folder name (e.g., 'singapore', 'newyork')
        extend_graph: If True, add Mapillary node and path nodes to extend graph. If False, only create node_text attributes.
        
    Returns:
        Number of subgraphs enhanced
    """
    import json
    import os
    
    print(f"Processing training data from: {jsonl_file_path}")
    if resume_from_line > 0:
        print(f"Resuming from line {resume_from_line + 1}")
    
    # Build node lookup dictionary once (reused for all subgraphs)
    print("Building node lookup dictionary (one-time operation)...")
    node_id_to_row = build_node_id_to_row_lookup(nodes_gdf)
    print(f"✅ Built lookup dictionary with {len(node_id_to_row)} node entries")
    
    enhanced_count = 0
    skipped_count = 0
    processed_subgraphs = set()  # Track processed subgraph files to avoid duplicates
    successful_graph_paths = set()  # Track successfully enhanced graph paths (normalized)
    
    with open(jsonl_file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            # Skip lines before resume point
            if line_num <= resume_from_line:
                continue
                
            # try:
            data = json.loads(line.strip())

            # Skip items with invalid source_name (all digits) early to save time
            if not has_valid_source_name(data):
                skipped_count += 1
                if skipped_count % 100 == 0:
                    print(f"Skipped {skipped_count} subgraph enhancements for invalid source_name entries")
                continue

            # Extract required information
            mapillary_node = data.get('mapillary_node', '')
            graph_path = data.get('graph_path', '')
            image_coordinates = data.get('image_coordinates', '')

            if not mapillary_node or not graph_path:
                continue

            # Normalize graph_path to local directory structure
            normalized_graph_path = normalize_graph_path(graph_path)

            # Use provided subgraph directory or extract from normalized graph_path
            if subgraph_directory:
                pkl_file = os.path.join(subgraph_directory, os.path.basename(normalized_graph_path))
            else:
                pkl_file = normalized_graph_path

            # Skip if subgraph already processed (deduplication)
            pkl_basename = os.path.basename(pkl_file)
            if pkl_basename in processed_subgraphs:
                continue

            # Check if enhanced version already exists in fixed output directory (resume functionality)
            fixed_output_dir = os.path.join(OUTPUT_ROOT, 'multiview_three_pairs', 'subgraphs')
            enhanced_pkl = os.path.join(fixed_output_dir, pkl_basename)
            if os.path.exists(enhanced_pkl):
                print(f"⏭️ Skipping already processed subgraph: {pkl_basename}")
                processed_subgraphs.add(pkl_basename)
                # Mark as successful since it already exists
                successful_graph_paths.add(normalized_graph_path)
                continue

            if os.path.exists(pkl_file):
                print(f"Enhancing subgraph {line_num}: {pkl_file}")

                # Extract image coordinates using helper function (tries JSONL first, then nodes_gdf)
                image_coords = extract_image_coordinates(image_coordinates, mapillary_node, nodes_gdf)

                # Get path nodes for potential downsizing
                path_nodes = get_path_nodes_from_data(data)

                success = enhance_subgraph_with_mapillary_and_text(
                    pkl_file, mapillary_node, nodes_gdf, edges_gdf, image_coords, path_nodes, None, data_folder, node_id_to_row, extend_graph
                )
                if success:
                    enhanced_count += 1
                    processed_subgraphs.add(pkl_basename)
                    # Track successful graph path (use normalized path for matching)
                    successful_graph_paths.add(normalized_graph_path)
                else:
                    print(f"⚠️ Skipping sample - subgraph enhancement failed for {pkl_basename}")
            else:
                print(f"⚠️ Subgraph file not found: {pkl_file}")
                
            # except json.JSONDecodeError as e:
            #     print(f"Error parsing line {line_num}: {e}")
            #     continue
            # except Exception as e:
            #     print(f"Error processing line {line_num}: {e}")
            #     continue
    
    print(f"Subgraph enhancement summary: {enhanced_count} enhanced, {skipped_count} skipped (invalid source_name)")
    print(f"Processed {len(processed_subgraphs)} unique subgraphs")
    return enhanced_count, successful_graph_paths


# Uncomment the line below to run the example
# training_count = example_usage()
# print(f"Successfully created {training_count} training examples!")

