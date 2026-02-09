import json
import time
import argparse
import re

import torch
import pandas as pd
import networkx as nx
import geopandas as gpd
import os
from tqdm import tqdm
import pickle
from datetime import datetime
import numpy as np
from itertools import islice, combinations
import math
import base64
import traceback
from io import BytesIO
from PIL import Image
from openai import OpenAI

# Get base paths from environment variables
DATA_ROOT = os.getenv('URBANKG_DATA_ROOT', './data')
OUTPUT_ROOT = os.getenv('URBANKG_OUTPUT_ROOT', './output')

# Try to import cuGraph for GPU acceleration
try:
    import cudf
    import cugraph

    CUGRAPH_AVAILABLE = True
    print("✅ cuGraph available - GPU acceleration enabled")
except ImportError:
    CUGRAPH_AVAILABLE = False
    print("⚠️ cuGraph not available - using CPU only")

# os.environ['DASHSCOPE_API_KEY'] = 'sk-16789ed96f4f4bfbb49e8adeb96167b7'
# os.environ['DASHSCOPE_API_KEY'] = 'sk-c4588c45d82145e4837648dfbc806f4d'
os.environ['NEWAPI_API_KEY'] = 'sk-aobTmuAHUBM0fR6FzZwmkCwrWRfyLWJ5Uy9Kk0L2X0TB2adQ'

# Global cache for nodes dictionary to avoid repeated building
_nodes_dict_cache = None
_nodes_gdf_hash = None


def get_nodes_dict(nodes_gdf):
    """
    Get or create nodes dictionary with caching to avoid repeated building.
    
    Args:
        nodes_gdf: GeoDataFrame containing node information
        
    Returns:
        dict: Dictionary mapping node IDs to node data
    """
    global _nodes_dict_cache, _nodes_gdf_hash
    
    # Create a hash of the nodes_gdf to detect changes
    current_hash = hash(str(nodes_gdf.shape) + str(nodes_gdf['id'].sum()))
    
    # Return cached version if available and nodes_gdf hasn't changed
    if _nodes_dict_cache is not None and _nodes_gdf_hash == current_hash:
        return _nodes_dict_cache
    
    # Build new nodes dictionary
    print("🔧 Building nodes dictionary cache...")
    _nodes_dict_cache = {row['id']: row for _, row in nodes_gdf.iterrows()}
    _nodes_gdf_hash = current_hash
    print(f"✅ Nodes dictionary cache built with {len(_nodes_dict_cache)} nodes")
    
    return _nodes_dict_cache


def clear_nodes_dict_cache():
    """Clear the nodes dictionary cache."""
    global _nodes_dict_cache, _nodes_gdf_hash
    _nodes_dict_cache = None
    _nodes_gdf_hash = None
    print("🗑️ Nodes dictionary cache cleared")


def extract_captions_from_enhanced_format(caption_text: str):
    """
    Extract image captions and summarizations from the enhanced caption format.
    (Copied from construct_train_jsonl.py)

    Args:
        caption_text (str): Text containing image descriptions and summarization

    Returns:
        tuple: (image_captions, summarization) where:
            - image_captions is a list of descriptions for each image
            - summarization is the summarization text (string)
    """
    image_captions = []
    summarization = ""
    image_caption = ""

    # Primary format: **Image N:** ... **Summarization:**
    if "**Summarization:**" in caption_text:
        # Split at the summarization section
        parts = caption_text.split("**Summarization:**")
        image_caption = parts[0].strip()
        summarization = parts[1].strip() if len(parts) > 1 else ""
    else:
        print("⚠️ No summarization section found - skipping this item")
        # Return None values to signal that this item should be skipped
        return None, None

    #     # Extract individual image captions from the main content
    #     # Pattern: **Image N:** followed by content until next **Image N:** or end
    #     # image_pattern = r"\*\*Image (\d+):\*\*\s*(.*?)(?=\*\*Image \d+:\*\*|$)"
    #     # image_matches = re.findall(image_pattern, main_content, re.DOTALL)
    #     #
    #     # for image_num, content in image_matches:
    #     #     # Clean up the content - remove extra whitespace and newlines
    #     #     cleaned_content = re.sub(r'\s+', ' ', content.strip())
    #     #     image_captions.append(cleaned_content)
    #
    # # Fallback format: ### Summarization:
    # elif "### Summarization:" in caption_text:
    #     # Split at the summarization section
    #     parts = caption_text.split("### Summarization:")
    #     main_content = parts[0].strip()
    #     summarization = parts[1].strip() if len(parts) > 1 else ""
    #
    #     # Extract individual image captions from the main content
    #     image_pattern = r"### Image (\d+):\s*(.*?)(?=### Image \d+:|$)"
    #     image_matches = re.findall(image_pattern, main_content, re.DOTALL)
    #
    #     for image_num, content in image_matches:
    #         cleaned_content = re.sub(r'\s+', ' ', content.strip())
    #         image_captions.append(cleaned_content)
    #
    # # Legacy format with DETAILED PART and SUMMARIZATION PART for each image
    # elif "**DETAILED PART:**" in caption_text or "**SUMMARIZATION PART:**" in caption_text:
    #     image_pattern = r"### Image (\d+):(.*?)(?=### Image \d+:|$)"
    #     image_matches = re.findall(image_pattern, caption_text, re.DOTALL)
    #
    #     summarizations = []
    #
    #     for image_num, content in image_matches:
    #         content = content.strip()
    #
    #         # Extract detailed part
    #         detailed_match = re.search(r"\*\*DETAILED PART:\*\*\s*(.*?)(?=\*\*SUMMARIZATION PART:\*\*|$)",
    #                                    content, re.DOTALL)
    #         if detailed_match:
    #             detailed_text = detailed_match.group(1).strip()
    #             cleaned_detailed = re.sub(r'\s+', ' ', detailed_text)
    #             image_captions.append(cleaned_detailed)
    #         else:
    #             # If no DETAILED PART marker, look for content before SUMMARIZATION PART
    #             summ_index = content.find("**SUMMARIZATION PART:**")
    #             if summ_index != -1:
    #                 detailed_text = content[:summ_index].strip()
    #                 cleaned_detailed = re.sub(r'\s+', ' ', detailed_text)
    #                 image_captions.append(cleaned_detailed)
    #             else:
    #                 # No markers found, use entire content as detailed
    #                 cleaned_content = re.sub(r'\s+', ' ', content)
    #                 image_captions.append(cleaned_content)
    #
    #         # Extract summarization part for this image
    #         summarization_match = re.search(r"\*\*SUMMARIZATION PART:\*\*\s*(.*?)$", content, re.DOTALL)
    #         if summarization_match:
    #             summarization_text = summarization_match.group(1).strip()
    #             # Clean up any incomplete text at the end
    #             if summarization_text and not summarization_text.endswith('.'):
    #                 if summarization_text.endswith('**SUMMAR') or summarization_text.endswith('**SUM'):
    #                     sentences = summarization_text.split('.')
    #                     if len(sentences) > 1:
    #                         summarization_text = '.'.join(sentences[:-1]) + '.'
    #             summarizations.append(f"Image {image_num}: {summarization_text}")
    #         else:
    #             summarizations.append("")
    #
    #     # Combine all individual summarizations
    #     summarization = "\n\n".join([s for s in summarizations if s])
    #
    # # Additional fallback: try to extract from any **Image N:** pattern without summarization section
    # if not image_captions:
    #     # Look for **Image N:** patterns anywhere in the text
    #     image_pattern = r"\*\*Image (\d+):\*\*\s*(.*?)(?=\*\*Image \d+:\*\*|\*\*Summarization:\*\*|$)"
    #     image_matches = re.findall(image_pattern, caption_text, re.DOTALL)
    #
    #     for image_num, content in image_matches:
    #         # Stop at summarization if found
    #         if "**Summarization:**" in content:
    #             content = content.split("**Summarization:**")[0]
    #         cleaned_content = re.sub(r'\s+', ' ', content.strip())
    #         if cleaned_content:
    #             image_captions.append(cleaned_content)
    #
    #     # If we found image content but no explicit summarization, try to extract it
    #     if image_captions and not summarization:
    #         if "**Summarization:**" in caption_text:
    #             summarization = caption_text.split("**Summarization:**")[1].strip()
    #
    # # Clean up summarization text
    # if summarization:
    #     # Remove excessive whitespace and normalize formatting
    #     summarization = re.sub(r'\s+', ' ', summarization)
    #     # Remove any bullet point formatting for cleaner text
    #     summarization = re.sub(r'^\s*[-•]\s*', '', summarization, flags=re.MULTILINE)

    return image_caption, summarization


def save_progress_checkpoint(output_dir, checkpoint_data):
    """
    Save progress checkpoint to allow resuming from where processing was paused.
    
    Args:
        output_dir: Output directory where checkpoint will be saved
        checkpoint_data: Dictionary containing progress information
    """
    checkpoint_file = os.path.join(output_dir, "progress_checkpoint.json")
    try:
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, indent=2)
        print(f"💾 Progress checkpoint saved: {checkpoint_file}")
    except Exception as e:
        print(f"⚠️ Failed to save checkpoint: {e}")


def load_progress_checkpoint(output_dir):
    """
    Load progress checkpoint to resume processing from where it was paused.
    
    Args:
        output_dir: Output directory where checkpoint is saved
        
    Returns:
        Dictionary containing progress information or None if no checkpoint exists
    """
    checkpoint_file = os.path.join(output_dir, "progress_checkpoint.json")
    if not os.path.exists(checkpoint_file):
        return None
    
    try:
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            checkpoint_data = json.load(f)
        print(f"📂 Progress checkpoint loaded: {checkpoint_file}")
        return checkpoint_data
    except Exception as e:
        print(f"⚠️ Failed to load checkpoint: {e}")
        return None


def clear_progress_checkpoint(output_dir):
    """
    Clear progress checkpoint file.
    
    Args:
        output_dir: Output directory where checkpoint is saved
    """
    checkpoint_file = os.path.join(output_dir, "progress_checkpoint.json")
    if os.path.exists(checkpoint_file):
        try:
            os.remove(checkpoint_file)
            print(f"🗑️ Progress checkpoint cleared: {checkpoint_file}")
        except Exception as e:
            print(f"⚠️ Failed to clear checkpoint: {e}")


def get_processed_s2cell_ids_from_checkpoint(checkpoint_data):
    """
    Extract processed S2 cell IDs from checkpoint data.
    
    Args:
        checkpoint_data: Dictionary containing checkpoint information
        
    Returns:
        Set of processed S2 cell IDs
    """
    if not checkpoint_data:
        return set()
    
    processed_ids = set()
    
    # Get processed IDs from checkpoint
    if 'processed_s2cell_ids' in checkpoint_data:
        processed_ids.update(checkpoint_data['processed_s2cell_ids'])
    
    # Also get processed IDs from last completed batch
    if 'last_completed_batch' in checkpoint_data:
        last_batch = checkpoint_data['last_completed_batch']
        if 'batch_results' in last_batch:
            for s2cell_id in last_batch['batch_results'].keys():
                processed_ids.add(s2cell_id)
    
    return processed_ids


def calculate_bearing(lat1, lon1, lat2, lon2):
    """
    Calculate bearing from point 1 to point 2 in degrees using spherical trigonometry.

    Args:
        lat1, lon1: Latitude and longitude of starting point (in degrees)
        lat2, lon2: Latitude and longitude of ending point (in degrees)

    Returns:
        bearing: Bearing in degrees (0-360), where 0° is North, 90° is East, etc.
    """
    import math

    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # Calculate difference in longitude
    dlon = lon2 - lon1

    # Calculate bearing using spherical trigonometry formula
    # y = sin(Δlong).cos(lat2)
    # x = cos(lat1).sin(lat2) − sin(lat1).cos(lat2).cos(Δlong)
    y = math.sin(dlon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)

    # Calculate bearing in radians using atan2 (handles all quadrants correctly)
    bearing = math.atan2(y, x)

    # Convert from radians to degrees
    bearing = math.degrees(bearing)

    # Normalize to 0-360 degrees (atan2 returns -180 to +180)
    bearing = (bearing + 360) % 360

    return bearing


def bearing_to_direction(bearing):
    """
    Convert bearing degrees to 16-point compass direction.

    Args:
        bearing: Bearing in degrees (0-360)

    Returns:
        direction: Compass direction string (N, NE, E, SE, S, SW, W, NW, etc.)
    """
    bearing = bearing % 360  # Ensure bearing is in 0-360 range

    # 16-point compass directions
    # Each direction covers 22.5 degrees (360/16 = 22.5)
    directions = [
        "North",  # "N"   - 0° ± 11.25°    (348.75° - 11.25°)
        "North-Northeast",  # "NNE" - 22.5° ± 11.25° (11.25° - 33.75°)
        "Northeast",  # "NE"  - 45° ± 11.25°   (33.75° - 56.25°)
        "East-Northeast",  # "ENE" - 67.5° ± 11.25° (56.25° - 78.75°)
        "East",  # "E"   - 90° ± 11.25°   (78.75° - 101.25°)
        "East-Southeast",  # "ESE" - 112.5° ± 11.25° (101.25° - 123.75°)
        "Southeast",  # "SE"  - 135° ± 11.25°  (123.75° - 146.25°)
        "South-Southeast",  # "SSE" - 157.5° ± 11.25° (146.25° - 168.75°)
        "South",  # "S"   - 180° ± 11.25°  (168.75° - 191.25°)
        "South-Southwest",  # "SSW" - 202.5° ± 11.25° (191.25° - 213.75°)
        "Southwest",  # "SW"  - 225° ± 11.25°  (213.75° - 236.25°)
        "West-Southwest",  # "WSW" - 247.5° ± 11.25° (236.25° - 258.75°)
        "West",  # "W"   - 270° ± 11.25°  (258.75° - 281.25°)
        "West-Northwest",  # "WNW" - 292.5° ± 11.25° (281.25° - 303.75°)
        "Northwest",  # "NW"  - 315° ± 11.25°  (303.75° - 326.25°)
        "North-Northwest"  # "NNW" - 337.5° ± 11.25° (326.25° - 348.75°)
    ]

    # Calculate index: divide bearing by 22.5 and round to nearest integer
    # This maps each 22.5° segment to the corresponding direction
    index = round(bearing / 22.5) % 16

    return directions[index]


def safe_extract_coords(coords):
    if isinstance(coords[0], (int, float)) and isinstance(coords[1], (int, float)):
        return coords[1], coords[0]  # Return (lat, lon)
    elif isinstance(coords[0], tuple) and isinstance(coords[1], tuple):
        # Handle ((lon1, lat1), (lon2, lat2)) format from LineString
        return coords[0][1], coords[0][0]


def get_direction_from_closest_center(node_info, center_node_ids, all_nodes_info):
    """
    Calculate direction and distance from center node (mapillary node) to this node.
    Uses nearest point on LineString for accurate distance/direction calculation.
    """
    # Get the center node (should only be one - the mapillary node)
    if not center_node_ids or center_node_ids[0] not in all_nodes_info:
        return ""
    
    center_id = center_node_ids[0]
    center_info = all_nodes_info[center_id]
    
    try:
        from shapely.ops import nearest_points
        from shapely.geometry import Point
        from geopy.distance import geodesic
        
        # Get center point geometry (mapillary node - should be a Point)
        center_coords = center_info['coordinates']
        if isinstance(center_coords[0], tuple):
            # LineString format (unlikely)
            center_lon = (center_coords[0][0] + center_coords[1][0]) / 2
            center_lat = (center_coords[0][1] + center_coords[1][1]) / 2
        else:
            # Point format (typical for mapillary node)
            center_lon, center_lat = center_coords[0], center_coords[1]
        
        if center_lon is None or center_lat is None:
            return ""
        
        center_point = Point(center_lon, center_lat)
        
        # Get this node's geometry
        node_geom_type = node_info.get('geometry_type', 'Point')
        node_coords = node_info['coordinates']
        
        if not node_coords:
            return ""
        
        # Determine the target point based on geometry type
        if node_geom_type in ['LineString', 'MultiLineString']:
            # For LineString, we need to get the actual geometry object
            # Since we don't have direct access to geometry, use nearest_points with coordinate approximation
            if isinstance(node_coords[0], tuple) and len(node_coords[0]) == 2:
                # LineString format: ((lon1, lat1), (lon2, lat2))
                # Create a LineString from the coordinates
                from shapely.geometry import LineString
                first_lon, first_lat = node_coords[0]
                last_lon, last_lat = node_coords[1]
                node_line = LineString([(first_lon, first_lat), (last_lon, last_lat)])
                
                # Find nearest point on the line to the center
                nearest_point_on_node = nearest_points(center_point, node_line)[1]
                target_lon, target_lat = nearest_point_on_node.x, nearest_point_on_node.y
            else:
                return ""
        else:
            # For Point/Polygon, use the coordinates directly
            if isinstance(node_coords[0], tuple):
                # Unlikely format
                target_lon = (node_coords[0][0] + node_coords[1][0]) / 2
                target_lat = (node_coords[0][1] + node_coords[1][1]) / 2
            else:
                target_lon, target_lat = node_coords[0], node_coords[1]
        
        if target_lon is None or target_lat is None:
            return ""
        
        # Calculate distance using geopy (more accurate for geographic coordinates)
        distance_m = geodesic((center_lat, center_lon), (target_lat, target_lon)).meters
        
        if distance_m < 0.1:  # Very close or same location
            return ""
        
        # Calculate bearing from center to target
        bearing = calculate_bearing(center_lat, center_lon, target_lat, target_lon)
        direction = bearing_to_direction(bearing)
        
        return f", {direction}, {int(distance_m)}m from image location"
        
    except Exception as e:
        # Fallback to simple calculation if shapely is not available
        coords = node_info['coordinates']
        if not coords or (hasattr(coords, '__len__') and len(coords) == 0):
            return ""
        
        # Simple midpoint calculation as fallback
        if isinstance(coords[0], tuple) and len(coords[0]) == 2:
            node_lon = (coords[0][0] + coords[1][0]) / 2
            node_lat = (coords[0][1] + coords[1][1]) / 2
        elif len(coords) == 2 and not isinstance(coords[0], tuple):
            node_lon, node_lat = coords[0], coords[1]
        else:
            return ""
        
        center_coords = all_nodes_info[center_id]['coordinates']
        if isinstance(center_coords[0], tuple):
            center_lon = (center_coords[0][0] + center_coords[1][0]) / 2
            center_lat = (center_coords[0][1] + center_coords[1][1]) / 2
        else:
            center_lon, center_lat = center_coords[0], center_coords[1]
        
        if node_lon is None or node_lat is None or center_lon is None or center_lat is None:
            return ""
        
        distance_m = degrees_to_meters(node_lat, node_lon, center_lat, center_lon)
        if distance_m:
            bearing = calculate_bearing(center_lat, center_lon, node_lat, node_lon)
            direction = bearing_to_direction(bearing)
            return f", {direction}, {int(distance_m)}m from image location"

        return ""


class GraphProcessor:
    """
    Graph processing class with GPU acceleration support for subgraphs
    """

    def __init__(self, use_gpu=True):
        self.use_gpu = use_gpu and CUGRAPH_AVAILABLE
        self.nx_graph = None
        self.cu_graph = None
        self.edge_df = None

        if self.use_gpu:
            print("🚀 Using GPU acceleration with cuGraph")
        else:
            print("🐌 Using CPU with NetworkX")

        self.nx_to_cu_mapping = {}  # NetworkX node ID -> cuGraph node ID
        self.cu_to_nx_mapping = {}  # cuGraph node ID -> NetworkX node ID
        self.needs_mapping = False

    def _create_node_mapping(self):
        """Create mapping between NetworkX and cuGraph node IDs"""
        try:
            nodes = list(self.nx_graph.nodes())

            print(f"📊 Graph analysis:")
            print(f"   Total nodes: {len(nodes)}")
            if nodes:
                print(f"   Node ID range: {min(nodes)} to {max(nodes)}")

            # Check if nodes are already consecutive integers starting from 0
            if all(isinstance(node, int) for node in nodes):
                min_node = min(nodes)
                max_node = max(nodes)
                expected_consecutive = list(range(min_node, max_node + 1))

                # If nodes are consecutive integers starting from 0, no mapping needed
                if (min_node == 0 and
                        max_node == len(nodes) - 1 and
                        set(nodes) == set(expected_consecutive)):
                    self.needs_mapping = False
                    print("✅ Node IDs are already cuGraph compatible (consecutive 0 to N-1)")
                    return

            # Always create mapping for safety - handles gaps, non-integers, etc.
            self.needs_mapping = True

            # Clear any existing mappings
            self.nx_to_cu_mapping.clear()
            self.cu_to_nx_mapping.clear()

            # Create mapping: NetworkX node -> cuGraph index (0, 1, 2, ...)
            for i, nx_node in enumerate(sorted(nodes)):  # Sort for consistency
                self.nx_to_cu_mapping[nx_node] = i
                self.cu_to_nx_mapping[i] = nx_node

            print(f"🔄 Created node mapping for {len(nodes)} nodes")

        except Exception as e:
            print(f"❌ Failed to create node mapping: {e}")
            self.needs_mapping = False

    def _convert_to_cugraph(self):
        """Convert NetworkX graph to cuGraph"""
        try:
            print("🔄 Starting cuGraph conversion...")

            # Create node mapping first
            self._create_node_mapping()

            # Extract edges
            edges = list(self.nx_graph.edges())

            # Handle empty graph gracefully
            if not edges:
                print("⚠️ Graph has no edges - disabling GPU acceleration")
                self.use_gpu = False
                self.cu_graph = None
                return

            print(f"📊 Processing {len(edges)} edges...")

            # Map edges if needed
            if self.needs_mapping:
                mapped_edges = []
                for u, v in edges:
                    cu_u = self.nx_to_cu_mapping.get(u)
                    cu_v = self.nx_to_cu_mapping.get(v)
                    if cu_u is not None and cu_v is not None:
                        mapped_edges.append((cu_u, cu_v))
                edges = mapped_edges

            if not edges:
                print("⚠️ No valid edges after mapping - disabling GPU acceleration")
                self.use_gpu = False
                self.cu_graph = None
                return

            # Create edge DataFrame
            self.edge_df = cudf.DataFrame({
                'src': [e[0] for e in edges],
                'dst': [e[1] for e in edges]
            })

            # Create cuGraph
            self.cu_graph = cugraph.Graph()
            self.cu_graph.from_cudf_edgelist(
                self.edge_df,
                source='src',
                destination='dst'
            )

            print("✅ Successfully converted to cuGraph")

        except Exception as e:
            print(f"❌ Failed to convert to cuGraph: {e}")
            self.use_gpu = False
            self.cu_graph = None

    def _map_nx_to_cu_node(self, nx_node):
        """Convert NetworkX node ID to cuGraph node ID"""
        if not self.needs_mapping:
            return nx_node
        return self.nx_to_cu_mapping.get(nx_node)

    def _map_cu_to_nx_node(self, cu_node):
        """Convert cuGraph node ID to NetworkX node ID"""
        if not self.needs_mapping:
            return cu_node
        return self.cu_to_nx_mapping.get(cu_node)


def degrees_to_meters(lat1, lon1, lat2, lon2):
    """
    Convert coordinate differences to meters using Haversine formula
    """
    try:
        # Convert latitude and longitude from degrees to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        c = 2 * math.asin(math.sqrt(a))

        # Radius of Earth in meters
        r = 6371000

        return c * r
    except:
        return None


def check_s2cell_image_node_distances(s2cell_data, nodes_gdf, max_distance_km=2):
    """
    Check if the maximum distance between any two image nodes in an S2 cell exceeds the threshold.

    Args:
        s2cell_data: S2 cell data containing image node information
        nodes_gdf: GeoDataFrame containing node information with coordinates
        max_distance_km: Maximum allowed distance in kilometers (default: 1.0)

    Returns:
        tuple: (should_skip, max_distance_km, node_pairs_info)
            - should_skip: True if S2 cell should be skipped due to distance
            - max_distance_km: The actual maximum distance found
            - node_pairs_info: Information about the node pairs and their distances
    """
    # Extract image node IDs
    image_node_ids = extract_image_node_ids(s2cell_data)

    if len(image_node_ids) <= 1:
        # Only one or no image nodes, no distance check needed
        return False, 0.0, {}

    # Get coordinates for all image nodes
    node_coordinates = {}
    for node_id in image_node_ids:
        node_row = nodes_gdf[nodes_gdf['id'] == node_id]
        if not node_row.empty:
            node_data = node_row.iloc[0]
            if node_data.geometry.geom_type == 'Point':
                lon, lat = node_data.geometry.x, node_data.geometry.y
                node_coordinates[node_id] = (lat, lon)

    if len(node_coordinates) <= 1:
        # No valid coordinates found or only one node with coordinates
        return False, 0.0, {}

    # Calculate distances between all pairs of image nodes
    max_distance_m = 0.0
    max_distance_pair = None
    all_distances = {}

    from itertools import combinations

    for node1_id, node2_id in combinations(node_coordinates.keys(), 2):
        lat1, lon1 = node_coordinates[node1_id]
        lat2, lon2 = node_coordinates[node2_id]

        # Calculate distance in meters
        distance_m = degrees_to_meters(lat1, lon1, lat2, lon2)
        distance_km = distance_m / 1000.0

        all_distances[f"{node1_id}-{node2_id}"] = {
            'distance_m': distance_m,
            'distance_km': distance_km,
            'coordinates': {
                'node1': (lat1, lon1),
                'node2': (lat2, lon2)
            }
        }
        print('distance_m', distance_m)

        if distance_m > max_distance_m:
            max_distance_m = distance_m
            max_distance_pair = (node1_id, node2_id)

    max_distance_km_actual = max_distance_m / 1000.0
    should_skip = max_distance_km_actual > max_distance_km

    node_pairs_info = {
        'total_image_nodes': len(image_node_ids),
        'nodes_with_coordinates': len(node_coordinates),
        'max_distance_km': max_distance_km_actual,
        'max_distance_pair': max_distance_pair,
        'all_distances': all_distances,
        'exceeds_threshold': should_skip
    }

    return should_skip, max_distance_km_actual, node_pairs_info


def use_qwen_vl_for_captioning(prompt_text, image_paths, model_name="qwen-vl-max-latest", api_key=None):
    """
    Generate captions for images using Alibaba DashScope API's Qwen VL models.

    Args:
        prompt_text: The caption prompt
        image_paths: List of paths to the images
        model_name: Qwen VL model to use (default: "qwen-vl-max-latest")
        api_key: DashScope API key (defaults to DASHSCOPE_API_KEY environment variable)

    Returns:
        Generated caption and list of valid image paths
    """
    # Get API key from environment if not provided
    if api_key is None:
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError(
                "No API key provided. Set DASHSCOPE_API_KEY environment variable or pass api_key parameter.")

    # Initialize the content list for the user message
    user_content = []
    valid_image_paths = []

    # Process each image in the list
    for image_path in image_paths:
        # Adjust image path format
        image_path = image_path.replace('jpeg', 'jpg')
        # Note: Image paths should be absolute or relative to DATA_ROOT
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            continue

        try:
            # Verify we can open the image
            with Image.open(image_path) as img:
                img_size = img.size
                print(f"Successfully opened image {image_path} with size {img_size}")
                valid_image_paths.append(image_path)

                # Convert image to base64
                buffered = BytesIO()
                img.save(buffered, format=img.format if img.format else "JPEG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                img_base64_uri = f"data:image/{img.format.lower() if img.format else 'jpeg'};base64,{img_base64}"

                # Add image to content list
                user_content.append({"type": "image_url", "image_url": {"url": img_base64_uri}})

        except Exception as img_error:
            print(f"Warning: Failed to open image {image_path}: {str(img_error)}")
            continue

    # If no valid images were processed, return an error
    if not user_content:
        raise ValueError("No valid images found in the provided image_paths list")

    # Add the prompt text at the end of content list
    user_content.append({"type": "text", "text": prompt_text})

    # Initialize OpenAI client with DashScope compatible API
    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    # Prepare messages with images and prompt
    messages = [
        {"role": "system", "content": [
            {"type": "text",
             "text": "You are a helpful assistant that generates detailed captions for street images with spatial context."}
        ]},
        {"role": "user", "content": user_content},
    ]

    # Print diagnostics
    print(f"Processing {len(valid_image_paths)} valid images out of {len(image_paths)} total")
    print(f"Prompt length: {len(prompt_text)} characters")
    print(f"Using model: {model_name}")

    try:
        # Call the API
        completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.2,
            max_tokens=1200,  # Increase token limit for detailed captions
        )

        # Extract the generated text
        output_text = completion.choices[0].message.content
        return output_text, valid_image_paths

    except Exception as e:
        print(f"Failed to generate caption: {str(e)}")
        traceback.print_exc()
        # Return a placeholder instead of raising an exception
        return f"[Error generating caption: {str(e)}...]", valid_image_paths


def use_qwen_vl_for_single_image_captioning(prompt_text, image_path, model_name="qwen-vl-max-latest", api_key=None):
    """
    Generate caption for a single image using Alibaba DashScope API's Qwen VL models.

    Args:
        prompt_text: The caption prompt
        image_path: Path to the single image
        model_name: Qwen VL model to use (default: "qwen-vl-max-latest")
        api_key: DashScope API key (defaults to DASHSCOPE_API_KEY environment variable)

    Returns:
        Generated caption and valid image path
    """
    # Get API key from environment if not provided
    if api_key is None:
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError(
                "No API key provided. Set DASHSCOPE_API_KEY environment variable or pass api_key parameter.")

    # Initialize the content list for the user message
    user_content = []
    valid_image_path = None

    # Process the single image
    image_path = image_path.replace('jpeg', 'jpg')
    if not os.path.exists(image_path):
        print(f"Warning: Image not found: {image_path}")
        return None, None

    try:
        # Verify we can open the image
        with Image.open(image_path) as img:
            img_size = img.size
            print(f"Successfully opened image {image_path} with size {img_size}")
            valid_image_path = image_path

            # Convert image to base64
            buffered = BytesIO()
            img.save(buffered, format=img.format if img.format else "JPEG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            img_base64_uri = f"data:image/{img.format.lower() if img.format else 'jpeg'};base64,{img_base64}"

            # Add image to content list
            user_content.append({"type": "image_url", "image_url": {"url": img_base64_uri}})

    except Exception as img_error:
        print(f"Warning: Failed to open image {image_path}: {str(img_error)}")
        return None, None

    # If no valid image was processed, return an error
    if not user_content:
        raise ValueError("No valid image found")

    # Add the prompt text at the end of content list
    user_content.append({"type": "text", "text": prompt_text})

    # Initialize OpenAI client with DashScope compatible API
    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    # Prepare messages with image and prompt
    messages = [
        {"role": "system", "content": [
            {"type": "text",
             "text": "You are a helpful assistant that generates detailed captions for street images with spatial context."}
        ]},
        {"role": "user", "content": user_content},
    ]

    # Print diagnostics
    print(f"Processing 1 image: {valid_image_path}")
    print(f"Prompt length: {len(prompt_text)} characters")
    print(f"Using model: {model_name}")

    try:
        # Call the API
        completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.2,
            max_tokens=1300,  # Increase token limit for detailed captions
        )

        # Extract the generated text
        output_text = completion.choices[0].message.content
        return output_text, valid_image_path

    except Exception as e:
        print(f"Failed to generate caption: {str(e)}")
        traceback.print_exc()
        # Return a placeholder instead of raising an exception
        return f"[Error generating caption: {str(e)}...]", valid_image_path


def use_yinli_for_single_image_captioning(prompt_text, image_path, model_name="gpt-4.1", api_key=None):
    """
    Generate caption for a single image using yinli.one API.

    Args:
        prompt_text: The caption prompt
        image_path: Path to the single image
        model_name: Model to use (default: "gpt-4.1")
        api_key: Yinli API key (defaults to NEWAPI_API_KEY environment variable)

    Returns:
        Generated caption and valid image path
    """
    # Get API key from environment if not provided
    if api_key is None:
        api_key = os.getenv("NEWAPI_API_KEY")
        if not api_key:
            raise ValueError(
                "No API key provided. Set NEWAPI_API_KEY environment variable or pass api_key parameter.")

    # Initialize the content list for the user message
    user_content = []
    valid_image_path = None

    # Process the single image
    image_path = image_path.replace('jpeg', 'jpg')
    if not os.path.exists(image_path):
        print(f"Warning: Image not found: {image_path}")
        return None, None

    try:
        # Verify we can open the image
        with Image.open(image_path) as img:
            img_size = img.size
            print(f"Successfully opened image {image_path} with size {img_size}")
            valid_image_path = image_path

            # Convert image to base64
            buffered = BytesIO()
            img.save(buffered, format=img.format if img.format else "JPEG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            img_base64_uri = f"data:image/{img.format.lower() if img.format else 'jpeg'};base64,{img_base64}"

            # Add image to content list
            user_content.append({"type": "image_url", "image_url": {"url": img_base64_uri}})

    except Exception as img_error:
        print(f"Warning: Failed to open image {image_path}: {str(img_error)}")
        return None, None

    # If no valid image was processed, return an error
    if not user_content:
        raise ValueError("No valid image found")

    # Add the prompt text at the end of content list
    user_content.append({"type": "text", "text": prompt_text})

    # Initialize OpenAI client with yinli.one API
    client = OpenAI(
        api_key=api_key,
        base_url="https://yinli.one/v1",
    )

    # Prepare messages with image and prompt
    messages = [
        {"role": "user", "content": user_content},
    ]

    # Print diagnostics
    print(f"Processing 1 image: {valid_image_path}")
    print(f"Prompt length: {len(prompt_text)} characters")
    print(f"Using model: {model_name}")

    try:
        # Call the API
        completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=300,
        )

        # Extract the generated text
        output_text = completion.choices[0].message.content
        return output_text, valid_image_path

    except Exception as e:
        print(f"Failed to generate caption: {str(e)}")
        traceback.print_exc()
        # Return a placeholder instead of raising an exception
        return f"[Error generating caption: {str(e)}...]", valid_image_path


def generate_captions_for_individual_images(prompt_text, image_paths, api_key=None, api_provider='qwen', delay_between_calls=2):
    """
    Generate captions for each image individually using the same prompt.
    
    Args:
        prompt_text: The caption prompt to use for all images
        image_paths: List of image paths
        api_key: API key for caption generation
        api_provider: API provider for caption generation ('qwen' or 'yinli')
        delay_between_calls: Delay between API calls in seconds (default: 2)
    
    Returns:
        Dictionary mapping image paths to their captions
    """
    captions = {}
    valid_image_paths = []
    
    print(f"🖼️ Generating individual captions for {len(image_paths)} images...")
    
    for i, image_path in enumerate(image_paths[:3]):
        print(f"\n📸 Processing image {i+1}/{len(image_paths)}: {os.path.basename(image_path)}")
        
        try:
            if api_provider == 'qwen':
                caption, valid_path = use_qwen_vl_for_single_image_captioning(
                    prompt_text=prompt_text,
                    image_path=image_path,
                    api_key=api_key
                )
            elif api_provider == 'yinli':
                caption, valid_path = use_yinli_for_single_image_captioning(
                    prompt_text=prompt_text,
                    image_path=image_path,
                    api_key=api_key
                )
            else:
                raise ValueError(f"Unsupported API provider: {api_provider}")
            
            if caption and valid_path:
                captions[valid_path] = caption
                valid_image_paths.append(valid_path)
                print(f"✅ Generated caption for {os.path.basename(image_path)}")
                print(f"   Caption length: {len(caption)} characters")
            else:
                print(f"⚠️ Failed to generate caption for {os.path.basename(image_path)}")
                
        except Exception as e:
            print(f"❌ Error generating caption for {os.path.basename(image_path)}: {e}")
            captions[image_path] = f"[Caption generation failed: {str(e)}]"
        
        # Add delay between calls to avoid rate limiting
        if i < len(image_paths) - 1:  # Don't delay after the last call
            print(f"⏳ Waiting {delay_between_calls} seconds before next call...")
            time.sleep(delay_between_calls)
    
    print(f"\n🎉 Successfully generated captions for {len(valid_image_paths)}/{len(image_paths)} images")
    return captions, valid_image_paths


def get_optimized_coordinates(node_info):
    """Get optimized coordinate representation based on geometry type and actual geometry data"""
    coords = node_info['coordinates']
    geom_type = node_info.get('geometry_type', 'unknown')

    # Try to get the actual geometry string if available
    geometry_str = ""
    if 'geometry' in node_info:
        geometry_str = str(node_info.get('geometry', ''))

    # If we have actual geometry data, use optimized parsing
    if geometry_str and geometry_str != 'nan' and geometry_str.lower() != 'none':
        try:
            import re
            if 'POINT' in geometry_str.upper():
                # Extract coordinates from POINT
                geom_coords = re.findall(r'[-+]?\d*\.?\d+', geometry_str)
                if len(geom_coords) >= 2:
                    return f"({geom_coords[0]}, {geom_coords[1]})"

            elif 'MULTILINESTRING' in geometry_str.upper():
                # Extract start and end points from first and last linestrings
                geom_coords = re.findall(r'[-+]?\d*\.?\d+', geometry_str)
                if len(geom_coords) >= 4:
                    return f"from ({geom_coords[0]}, {geom_coords[1]}) to ({geom_coords[-2]}, {geom_coords[-1]})"
                else:
                    return f"multiline (complex geometry)"

            elif 'LINESTRING' in geometry_str.upper():
                # Extract start and end points from LINESTRING
                geom_coords = re.findall(r'[-+]?\d*\.?\d+', geometry_str)
                if len(geom_coords) >= 4:
                    return f"from ({geom_coords[0]}, {geom_coords[1]}) to ({geom_coords[-2]}, {geom_coords[-1]})"

            elif 'MULTIPOLYGON' in geometry_str.upper():
                # Try to calculate centroid using shapely if available
                try:
                    from shapely import wkt
                    from shapely.geometry import MultiPolygon
                    geom = wkt.loads(geometry_str)
                    if isinstance(geom, MultiPolygon):
                        centroid = geom.centroid
                        return f"centered at ({centroid.x:.6f}, {centroid.y:.6f})"
                    else:
                        raise Exception("Not a multipolygon")
                except:
                    # Fallback: extract first coordinate from first polygon
                    geom_coords = re.findall(r'[-+]?\d*\.?\d+', geometry_str)
                    if len(geom_coords) >= 2:
                        return f"approx center ({geom_coords[0]}, {geom_coords[1]})"
                    else:
                        return f"multipolygon (complex)"

            elif 'POLYGON' in geometry_str.upper():
                # Try to calculate centroid using shapely if available
                try:
                    from shapely import wkt
                    from shapely.geometry import Polygon
                    geom = wkt.loads(geometry_str)
                    if isinstance(geom, Polygon):
                        centroid = geom.centroid
                        return f"centered at ({centroid.x:.6f}, {centroid.y:.6f})"
                    else:
                        raise Exception("Not a polygon")
                except:
                    # Fallback: extract first coordinate
                    geom_coords = re.findall(r'[-+]?\d*\.?\d+', geometry_str)
                    if len(geom_coords) >= 2:
                        return f"approx center ({geom_coords[0]}, {geom_coords[1]})"

            elif 'GEOMETRYCOLLECTION' in geometry_str.upper():
                # Handle geometry collections by extracting first coordinate
                geom_coords = re.findall(r'[-+]?\d*\.?\d+', geometry_str)
                if len(geom_coords) >= 2:
                    return f"collection at ({geom_coords[0]}, {geom_coords[1]})"
                else:
                    return f"collection (complex)"

        except Exception as e:
            # If geometry parsing fails, fall back to basic coordinates
            pass
    #
    # # Fallback to basic coordinate handling if geometry parsing fails or unavailable
    # if coords[0] is None or coords[1] is None:
    #     return "(coordinates unavailable)"
    #
    # if geom_type == 'Point':
    #     return f"({coords[0]:.6f}, {coords[1]:.6f})"
    # elif geom_type in ['LineString', 'MultiLineString']:
    #     return f"line at ({coords[0]:.6f}, {coords[1]:.6f})"
    # elif geom_type in ['Polygon', 'MultiPolygon']:
    #     return f"area at ({coords[0]:.6f}, {coords[1]:.6f})"
    # else:
    #     return f"({coords[0]:.6f}, {coords[1]:.6f})"


class SpatialContextQAGenerator:
    """
    Spatial Q&A generator for creating network descriptions
    """

    def __init__(self, model_name="Qwen/Qwen2.5-7B", device=None):
        self.model_name = model_name
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')

    def _extract_node_info(self, node_id, nodes_gdf):
        """Extract key information about a node from GeoDataFrame."""
        try:
            node = nodes_gdf[nodes_gdf['id'] == node_id].iloc[0]
            name = node.get('name', 'unnamed location')
            node_type = node.get('type', 'unknown type')
            category = node.get('category', None)

            # Extract additional address information
            street = node.get('street', None)
            housenumber = node.get('housenumber', None)
            postcode = node.get('postcode', None)
            building_use = node.get('building use', None)
            historic_district = node.get('historic district', None)
            architect = node.get('architect', None)
            address = node.get('address', None)
            building = node.get('building', None)
            historic = node.get('historic', None)
            turn_count = node.get('turn_count', None)
            
            # Extract geographic/administrative area information
            planning_area = node.get('planning_area', None)
            district = node.get('district', None)
            city = node.get('city', None)
            country = node.get('country', None)
            # Extract physical properties
            width = node.get('width', None)
            length = node.get('length_meters', None)

            # Extract directional properties
            street_direction = node.get('street_direction', None)
            direction = node.get('direction', None)
            # segment_position = node.get('segment_position', None)

            # Get coordinates (handle different geometry types)
            if node.geometry.geom_type == 'Point':
                coordinates = (node.geometry.x, node.geometry.y)
            elif node.geometry.geom_type == 'LineString':
                # Extract first and last coordinates from LineString
                coords_list = list(node.geometry.coords)
                if len(coords_list) >= 2:
                    first_coord = coords_list[0]  # First coordinate (start point)
                    last_coord = coords_list[-1]  # Last coordinate (end point)
                    # Store both as a tuple of tuples, or use first for main coordinate
                    coordinates = (first_coord, last_coord)  # or could store both: (first_coord, last_coord)
                # else:
                #     # Fallback to centroid if insufficient coordinates
                #     coordinates = (node.geometry.centroid.x, node.geometry.centroid.y)

            elif node.geometry.geom_type == 'MultiLineString':
                # Extract first coordinate from first linestring and last coordinate from last linestring
                linestrings = list(node.geometry.geoms)
                if linestrings:
                    first_linestring = linestrings[0]
                    last_linestring = linestrings[-1]

                    first_coords = list(first_linestring.coords)
                    last_coords = list(last_linestring.coords)

                    if first_coords and last_coords:
                        first_coord = first_coords[0]  # Start of first linestring
                        last_coord = last_coords[-1]  # End of last linestring
                        coordinates = (first_coord, last_coord)  # or store both: (first_coord, last_coord)
                #     else:
                #         # Fallback to centroid
                #         coordinates = (node.geometry.centroid.x, node.geometry.centroid.y)
                # else:
                #     coordinates = (node.geometry.centroid.x, node.geometry.centroid.y)

            elif node.geometry.geom_type in ['Polygon', 'MultiPolygon']:
                coordinates = (node.geometry.centroid.x, node.geometry.centroid.y)
            else:
                coordinates = (None, None)

            return {
                'id': node_id,
                'name': name if not pd.isna(name) else None,
                'type': node_type if not pd.isna(node_type) else None,
                'category': category if not pd.isna(category) else None,
                'street': street if not pd.isna(street) else None,
                'housenumber': housenumber if not pd.isna(housenumber) else None,
                'postcode': postcode if not pd.isna(postcode) else None,
                'address': address if not pd.isna(address) else None,
                'width': width if not pd.isna(width) else None,
                'length': length if not pd.isna(length) else None,
                'turn_count': turn_count if not pd.isna(turn_count) else None,
                'building use': building_use if not pd.isna(building_use) else None,  # Match key name with space
                'historic district': historic_district if not pd.isna(historic_district) else None,  # Match key name with space
                'architect': architect if not pd.isna(architect) else None,
                'historic': historic if not pd.isna(historic) else None,
                'building': building if not pd.isna(building) else None,
                'planning_area': planning_area if not pd.isna(planning_area) else None,
                'district': district if not pd.isna(district) else None,
                'city': city if not pd.isna(city) else None,
                'country': country if not pd.isna(country) else None,

                # 'segment_position': segment_position if not pd.isna(segment_position) else None,
                'coordinates': coordinates,
                'geometry_type': node.geometry.geom_type,
                'geometry': node.geometry
            }
        except (IndexError, KeyError) as e:
            return {
                'id': node_id,
                'name': None,
                'type': None,
                'category': None,
                'street': None,
                'housenumber': None,
                'postcode': None,
                'address': None,
                'width': None,
                'length': None,
                'street_direction': None,
                'direction': None,
                'segment_position': None,
                'coordinates': (None, None),
                'geometry_type': None,
            }

    def _get_edge_relationship_description(self, edge_type, center_info, neighbor_info, edge_data=None):
        """Generate a natural language description of the edge relationship between nodes."""
        if pd.isna(edge_type) or edge_type is None:
            return "is near"

        edge_type = str(edge_type).lower()

        # Get geometry types for both nodes
        center_geom_type = center_info.get('geometry_type', 'Unknown')
        neighbor_geom_type = neighbor_info.get('geometry_type', 'Unknown')
        center_name = center_info.get('name', 'Unknown')
        neighbor_name = neighbor_info.get('name', 'Unknown')

        if center_name == neighbor_name:
            return "continues as"

        # Handle crossing types
        elif edge_type == 'pedestrian-crossing':
            return "has a pedestrian crossing to"
        elif edge_type == 'complex-roundabout':
            return "connects via a complex roundabout to"
        elif edge_type == 'highway-t-junction':
            return "meets at a highway T-junction with"
        elif edge_type == 'major-complex-junction':
            return "meets at a major complex junction with"
        elif edge_type == 'major-roundabout':
            return "connects via a major roundabout to"
        elif edge_type == 'service-connection':
            return "has a service connection to"
        elif edge_type == 'street-change':
            return "connects to"
        elif edge_type == 'continuation':
            return "continues as"
        elif edge_type == 'major-road-end':
            return "ends at"
        elif edge_type == 'simple-roundabout':
            return "connects via a simple roundabout to"
        elif edge_type == 'path-connection':
            return "has a path connecting to"
        elif edge_type == 'three-way':
            return "meets at a three-way intersection with"
        elif edge_type == 'four-way':
            return "meets at a four-way intersection with"
        elif edge_type == 'complex-intersection':
            return "has a complex intersection with"
        elif edge_type == 'bridge-intersection':
            return "intersects with"
        elif edge_type == 'overlap':
            return "overlaps with"
        elif edge_type == 'bridge-continuation':
            return "continues across a bridge to"
        elif edge_type == 'highway-crossing':
            return "crosses"
        elif edge_type == 'highway-interchange':
            return "connects through a highway interchange to"
        elif edge_type == 'major-t-junction':
            return "meets at a major T-junction with"
        elif edge_type == 'three-way-bend':
            return "connects to"
        elif edge_type == 'highway-ramp':
            return "has a highway ramp connecting to"
        elif edge_type == 'side-connection':
            return "has a side connection to"
        elif edge_type == 'driveway-connection':
            return "connects to"
        elif 'crossing' in edge_type and edge_type != 'boundary_crossing':
            return "crosses"
        elif 'boundary' in edge_type:
            if center_geom_type in ['LineString', 'MultiLineString']:
                if neighbor_geom_type in ['Polygon', 'MultiPolygon']:
                    return "bounds"
                else:
                    return "is bounded by"
        elif edge_type == 'nearest':
            return "is near to"
        elif 'intersects' in edge_type:
            if center_geom_type in ['LineString', 'MultiLineString']:
                if neighbor_geom_type in ['Polygon', 'MultiPolygon']:
                    return "passes through"
                else:
                    return "intersects with"
            elif center_geom_type in ['Polygon', 'MultiPolygon']:
                if neighbor_geom_type in ['LineString', 'MultiLineString']:
                    return "is very near to"
                else:
                    return "overlaps with"
            else:
                return "intersects with"
        elif edge_type == 'contains':
            return "contains"
        elif edge_type == 'within':
            return "is within"
        elif edge_type == 'connects':
            return "connects to"
        else:
            return "is connected to"

    def _create_network_description(self, G, center_node_ids, nodes_gdf, edges_gdf, max_hops=2):
        """
        Create a network description using the spatial context QA generator approach
        """

        def get_crossing_info(edge_data):
            """Get crossing coordinates and details if edge involves a crossing"""
            crossing_info = ""
            edge_type = str(edge_data.get('type', '')).lower()

            if 'crossing' in edge_type and 'crossing_id' in edge_data:
                crossing_id = edge_data['crossing_id']
                if not pd.isna(crossing_id):
                    # Try to find crossing coordinates in nodes_gdf
                    crossing_row = nodes_gdf[nodes_gdf['id'] == crossing_id]
                    if not crossing_row.empty:
                        crossing_node = crossing_row.iloc[0]
                        if crossing_node.geometry.geom_type == 'Point':
                            crossing_lon, crossing_lat = crossing_node.geometry.x, crossing_node.geometry.y
                            crossing_info = f" at crossing point ({crossing_lon:.6f}, {crossing_lat:.6f})"

                        # Add crossing name if available
                        crossing_name = crossing_node.get('name', '')
                        if crossing_name and not pd.isna(crossing_name) and str(crossing_name).strip() != "":
                            crossing_info += f" [{crossing_name}]"

            return crossing_info

        def get_all_mentioned_node_ids():
            """Get all node IDs that are mentioned in the network structure"""
            mentioned_ids = set()

            # Add center nodes
            mentioned_ids.update(center_node_ids)

            # Add all nodes from all_nodes_info that appear in the network
            mentioned_ids.update(all_nodes_info.keys())

            return mentioned_ids

        def get_node_description_for_connection(node_name):
            """Get simplified node description for nodes mentioned in connections"""
            # Find all nodes with this name in the subgraph
            matching_nodes = []
            for node_id, node_info in all_nodes_info.items():
                if node_info['name'] == node_name:
                    matching_nodes.append((node_id, node_info))

            if not matching_nodes:
                return ""

            # If single node, return simplified attributes
            if len(matching_nodes) == 1:
                node_id, node_info = matching_nodes[0]
                node_type = node_info['category'] if node_info.get('category') else node_info['type']

                # Skip if unknown type or None
                if not node_type or str(node_type).lower() in ['unknown', 'unknown type', 'none']:
                    return ""

                geom_type = node_info.get('geometry_type', 'unknown')

                # For connections, keep it simple - just show key identifying attributes
                attrs = []

                # Add meaningful attributes (only the most important ones)
                for attr in ['building use', 'historic district', 'architect',
                                 'building', 'historic', 'housenumber', 'street', 'address','postcode',
                                 'planning_area', 'district', 'city', 'country']:
                    if (attr in node_info and
                            node_info[attr] is not None and
                            not pd.isna(node_info[attr]) and
                            str(node_info[attr]).strip() != "" and
                            str(node_info[attr]).lower() not in ['unknown', 'none', 'null']):
                        attrs.append(f"{attr}={node_info[attr]}")

                # Only show attributes if we have meaningful ones, otherwise keep it minimal
                attr_str = f", {', '.join(attrs)}" if attrs else ""
                return f" [{node_type}, {geom_type}{attr_str}]"

            # Multiple nodes - return very simple summary
            else:
                node_types = set()
                geom_types = set()
                for _, node_info in matching_nodes:
                    node_type = node_info['category'] if node_info.get('category') else node_info['type']
                    if node_type and str(node_type).lower() not in ['unknown', 'unknown type', 'none']:
                        node_types.add(node_type)
                    geom_types.add(node_info.get('geometry_type', 'unknown'))

                if not node_types:  # All nodes have unknown type
                    return ""

                type_str = ', '.join(node_types)
                geom_str = ', '.join(geom_types)
                return f" [{type_str}, {geom_str}, {len(matching_nodes)} instances]"

        if not isinstance(center_node_ids, list):
            center_node_ids = [center_node_ids]

        # Collect all nodes and edges in the subgraph
        all_nodes_info = {}
        all_edges = []

        # Create node lookup for efficient access (use cached version)
        nodes_dict = get_nodes_dict(nodes_gdf)
        # Filter to only include nodes in the subgraph
        nodes_dict = {node_id: node_data for node_id, node_data in nodes_dict.items() if node_id in G.nodes()}

        # Add information for all nodes in the subgraph
        for node_id in G.nodes():
            if node_id in nodes_dict:
                node_info = self._extract_node_info(node_id, nodes_gdf)
                all_nodes_info[node_id] = node_info

        processed_edge_pairs = set()

        # Process all edges in the subgraph
        for u_id, v_id in G.edges():
            # Create a unique identifier for this edge pair
            edge_pair = tuple(sorted([u_id, v_id]))

            if edge_pair in processed_edge_pairs:
                continue
            processed_edge_pairs.add(edge_pair)

            # Find edge information in the edges GeoDataFrame
            forward_edges = edges_gdf[
                (edges_gdf['id1'] == u_id) & (edges_gdf['id2'] == v_id)
                ]
            reverse_edges = edges_gdf[
                (edges_gdf['id2'] == u_id) & (edges_gdf['id1'] == v_id)
                ]

            # Process forward direction edges
            if not forward_edges.empty and u_id in all_nodes_info and v_id in all_nodes_info:
                for k in range(len(forward_edges)):
                    row = forward_edges.iloc[k]
                    full_edge_data = row.to_dict()
                    edge_type = row.get('type', None)

                    relationship = self._get_edge_relationship_description(
                        edge_type, all_nodes_info[u_id], all_nodes_info[v_id], full_edge_data
                    )

                    edge_data = {
                        'from_id': u_id,
                        'to_id': v_id,
                        'relationship': relationship,
                        'is_reversed': False
                    }

                    # Add original edge attributes
                    for col in full_edge_data:
                        if col not in edge_data:
                            edge_data[col] = full_edge_data[col]

                    all_edges.append(edge_data)

            # Process reverse direction edges
            if not reverse_edges.empty and u_id in all_nodes_info and v_id in all_nodes_info:
                for k in range(len(reverse_edges)):
                    row = reverse_edges.iloc[k]
                    full_edge_data = row.to_dict()
                    edge_type = row.get('type', None)

                    if edge_type == 'contains':
                        edge_type = 'within'
                    elif edge_type == 'nearest':
                        edge_type = 'connects'

                    if "crossing" in str(edge_type):
                        edge_type = "crosses"

                    relationship = self._get_edge_relationship_description(
                        edge_type, all_nodes_info[u_id], all_nodes_info[v_id], full_edge_data
                    )

                    edge_data = {
                        'from_id': u_id,
                        'to_id': v_id,
                        'relationship': relationship,
                        'is_reversed': True
                    }

                    for col in full_edge_data:
                        if col not in edge_data:
                            edge_data[col] = full_edge_data[col]

                    all_edges.append(edge_data)

        description = f"This network contains {len(all_nodes_info)} locations, with {len(center_node_ids)} center nodes and {len(all_edges)} connections:\n\n"

        # Step 1: Display center nodes (image locations) - keep detailed
        description += "Center nodes (image locations) "
        for node_id in center_node_ids:
            if node_id in all_nodes_info:
                node_info = all_nodes_info[node_id]
                # Skip nodes with None names
                # if node_info['name'] is None:
                #     continue
                    
                name = "node_info['name']"
                coord_str = get_optimized_coordinates(node_info)
                node_type = node_info['category'] if node_info.get('category') else node_info['type']
                geom_type = node_info.get('geometry_type', 'unknown geometry')

                attrs = [f"ID: {node_id}", f"type: {node_type}", f"geometry: {geom_type}"]

                # Add meaningful attributes for center nodes
                for attr in ['building use', 'historic district', 'architect',
                                 'building', 'historic', 'housenumber', 'street', 'address','postcode',
                                 'planning_area', 'district', 'city', 'country']:
                    if (attr in node_info and node_info[attr] is not None and
                            not pd.isna(node_info[attr]) and str(node_info[attr]).strip() != "" and
                            str(node_info[attr]).lower() not in ['unknown', 'none', 'null']):
                        attrs.append(f"{attr}={node_info[attr]}")

                # Calculate distance from closest center node
                coords = node_info['coordinates']
                min_distance_info = None
                if coords[0] is not None and coords[1] is not None:
                    min_distance = float('inf')
                    closest_center = None
                    node_lat, node_lon = coords[1], coords[0]

                    for center_id in center_node_ids:
                        if center_id in all_nodes_info:
                            center_coords = all_nodes_info[center_id]['coordinates']
                            if (all_nodes_info[center_id]['geometry_type'] in ['Point', 'Polygon', 'MultiPolygon'] and
                                    node_info['geometry_type'] in ['Point', 'Polygon', 'MultiPolygon'] and
                                    center_coords[0] is not None and center_coords[1] is not None):
                                center_lat, center_lon = center_coords[1], center_coords[0]
                                distance_m = degrees_to_meters(center_lat, center_lon, node_lat, node_lon)
                                if distance_m:
                                    if distance_m < min_distance:
                                        min_distance = distance_m
                                        closest_center = center_id

                    if closest_center is not None:
                        min_distance_info = f", {min_distance:.0f}m from image location {closest_center}"

                # distance_str = min_distance_info if min_distance_info else ""
                direction_info = get_direction_from_closest_center(node_info, center_node_ids, all_nodes_info)
                description += f"[{', '.join(attrs)}] at {coord_str} {direction_info}\n"

        # Step 2: Find adjacent nodes (edge-connected + nearby)
        edge_connected_nodes = set()
        for edge in all_edges:
            if edge['from_id'] in center_node_ids and edge['to_id'] not in center_node_ids:
                edge_connected_nodes.add(edge['to_id'])
            elif edge['to_id'] in center_node_ids and edge['from_id'] not in center_node_ids:
                edge_connected_nodes.add(edge['from_id'])

        nearby_nodes = set()
        for center_id in center_node_ids:
            if center_id in all_nodes_info:
                center_coords = all_nodes_info[center_id]['coordinates']
                if (all_nodes_info[center_id]['geometry_type'] in ['Point', 'MultiPolygon', 'Polygon'] and
                        center_coords[0] is not None and center_coords[1] is not None):

                    center_lat, center_lon = center_coords[1], center_coords[0]

                    for node_id, node_info in all_nodes_info.items():
                        if (node_id != center_id and node_id not in center_node_ids and
                                node_info['coordinates'][0] is not None and node_info['coordinates'][1] is not None and
                                node_info['geometry_type'] in ['Point', 'MultiPolygon', 'Polygon']):

                            node_lat, node_lon = node_info['coordinates'][1], node_info['coordinates'][0]
                            distance_meters = degrees_to_meters(center_lat, center_lon, node_lat, node_lon)
                            if distance_meters:
                                if distance_meters <= 200:
                                    nearby_nodes.add(node_id)

        adjacent_nodes = edge_connected_nodes.union(nearby_nodes)

        # Step 3: Display adjacent nodes with smart grouping for exact duplicates only - SORTED BY DISTANCE
        if adjacent_nodes:
            description += "\nNearby locations:\n"

            # FIRST: Calculate distances and create sortable list
            adjacent_nodes_with_distance = []
            for node_id in adjacent_nodes:
                if node_id in all_nodes_info:
                    node_info = all_nodes_info[node_id]
                    if node_info['name'] != None:
                        node_type = node_info['category'] if node_info.get('category') else node_info['type']

                        # Skip unknown types
                        if str(node_type).lower() not in ['unknown', 'unknown type']:
                            # Calculate minimum distance to any center node
                            min_distance = float('inf')
                            coords = node_info['coordinates']
                            if coords[0] is not None and coords[1] is not None:
                                node_lat, node_lon = coords[1], coords[0]

                                for center_id in center_node_ids:
                                    if center_id in all_nodes_info:
                                        center_coords = all_nodes_info[center_id]['coordinates']
                                        if center_coords[0] is not None and center_coords[1] is not None:
                                            center_lat, center_lon = center_coords[1], center_coords[0]
                                            distance_m = degrees_to_meters(node_lat, node_lon, center_lat, center_lon)
                                            if distance_m and distance_m < min_distance:
                                                min_distance = distance_m

                            adjacent_nodes_with_distance.append((node_id, node_info, min_distance))

            # Sort by distance (ascending)
            adjacent_nodes_with_distance.sort(key=lambda x: x[2])

            # Group nodes more selectively - only group if name, type, and street are identical
            # But maintain distance sorting within groups
            nodes_by_exact_group = {}
            for node_id, node_info, distance in adjacent_nodes_with_distance:
                node_type = node_info['category'] if node_info.get('category') else node_info['type']
                street = node_info.get('street', '')
                key = (node_info['name'], node_type, street)
                if key not in nodes_by_exact_group:
                    nodes_by_exact_group[key] = []
                nodes_by_exact_group[key].append((node_id, node_info, distance))

            # Sort groups by minimum distance within each group
            sorted_groups = []
            for key, node_items in nodes_by_exact_group.items():
                min_group_distance = min(item[2] for item in node_items)
                sorted_groups.append((key, node_items, min_group_distance))

            sorted_groups.sort(key=lambda x: x[2])  # Sort by minimum distance in group
            sorted_groups = sorted_groups[:10]
            # Display with detailed attributes, group only true duplicates
            for (name, node_type, street), node_items, min_group_dist in sorted_groups:

                if len(node_items) == 1:
                    # Single node - show full detailed attributes
                    node_id, node_info, distance = node_items[0]
                    coord_str = get_optimized_coordinates(node_info)
                    geom_type = node_info.get('geometry_type', 'unknown geometry')

                    attrs = [f"ID: {node_id}", f"type: {node_type}", f"geometry: {geom_type}"]

                    # Add detailed attributes
                    for attr in ['building use', 'historic district', 'architect',
                                 'building', 'historic', 'housenumber', 'street', 'address','postcode',
                                 'planning_area', 'district', 'city', 'country']:
                        if (attr in node_info and node_info[attr] is not None and
                                not pd.isna(node_info[attr]) and str(node_info[attr]).strip() != "" and
                                str(node_info[attr]).lower() not in ['unknown', 'none', 'null']):
                            attrs.append(f"{attr}={node_info[attr]}")

                    direction_info = get_direction_from_closest_center(node_info, center_node_ids, all_nodes_info)
                    description += f"- {name} [{', '.join(attrs)}] at {coord_str} {direction_info}\n"
                else:
                    # Multiple nodes with identical name, type, and street - show count with IDs and sample attributes
                    # Sort within group by distance and use closest one as sample
                    node_items.sort(key=lambda x: x[2])
                    sample_node_id, sample_node_info, sample_distance = node_items[0]

                    geom_types = set(node_info.get('geometry_type', 'unknown') for _, node_info, _ in node_items)
                    geom_str = ', '.join(geom_types)

                    attrs = [f"type: {node_type}", f"geometry: {geom_str}"]
                    if street:
                        attrs.append(f"street={street}")

                    # Add sample attributes from closest node
                    sample_attrs = []
                    for attr in ['housenumber', 'address', 'width']:
                        if (attr in sample_node_info and sample_node_info[attr] is not None and
                                not pd.isna(sample_node_info[attr]) and str(sample_node_info[attr]).strip() != ""):
                            sample_attrs.append(f"{attr}={sample_node_info[attr]}")

                    if sample_attrs:
                        attrs.extend(sample_attrs[:2])  # Add up to 2 sample attributes

                    # Get node IDs for display (sorted by distance)
                    node_ids = [str(node_id) for node_id, _, _ in node_items]
                    ids_str = ', '.join(node_ids[:5])  # Show first 5 IDs
                    if len(node_items) > 5:
                        ids_str += '...'

                    direction_info = get_direction_from_closest_center(sample_node_info, center_node_ids,
                                                                       all_nodes_info)
                    description += f"- {name} [{', '.join(attrs)}] ({len(node_items)} instances: {ids_str}) {direction_info}\n"

        # Step 4: Find neighbor nodes (connected to adjacent nodes) - SORTED BY DISTANCE
        neighbor_nodes = set()
        for edge in all_edges:
            if (edge['from_id'] in adjacent_nodes and
                    edge['to_id'] not in center_node_ids and
                    edge['to_id'] not in adjacent_nodes):
                neighbor_nodes.add(edge['to_id'])
            elif (edge['to_id'] in adjacent_nodes and
                  edge['from_id'] not in center_node_ids and
                  edge['from_id'] not in adjacent_nodes):
                neighbor_nodes.add(edge['from_id'])

        # Display neighbor nodes with detailed attributes - SORTED BY DISTANCE
        if neighbor_nodes:
            description += "\nLocations connected to nearby areas:\n"

            # Calculate distances and sort by distance first, then by attribute richness
            neighbor_nodes_with_info = []
            for node_id in neighbor_nodes:
                if node_id in all_nodes_info:
                    node_info = all_nodes_info[node_id]
                    node_type = node_info['category'] if node_info.get('category') else node_info['type']

                    if str(node_type).lower() not in ['unknown', 'unknown type']:
                        # Calculate minimum distance to any center node
                        min_distance = float('inf')
                        coords = node_info['coordinates']
                        if coords[0] is not None and coords[1] is not None:
                            node_lat, node_lon = coords[1], coords[0]

                            for center_id in center_node_ids:
                                if center_id in all_nodes_info:
                                    center_coords = all_nodes_info[center_id]['coordinates']
                                    if center_coords[0] is not None and center_coords[1] is not None:
                                        center_lat, center_lon = center_coords[1], center_coords[0]
                                        distance_m = degrees_to_meters(node_lat, node_lon, center_lat, center_lon)
                                        if distance_m and distance_m < min_distance:
                                            min_distance = distance_m

                        # Count meaningful attributes
                        attr_count = sum(1 for attr in ['building use', 'historic district', 'architect',
                                 'building', 'historic', 'housenumber', 'street', 'address','postcode',
                                 'planning_area', 'district', 'city', 'country']
                                         if (attr in node_info and node_info[attr] is not None and
                                             not pd.isna(node_info[attr]) and str(node_info[attr]).strip() != ""))
                        neighbor_nodes_with_info.append((node_id, node_info, attr_count, min_distance))

            # Sort by distance first (ascending), then by attribute count (descending) as secondary
            neighbor_nodes_with_info.sort(key=lambda x: (x[3], -x[2]))

            # Show top 15 by distance
            for node_id, node_info, attr_count, distance in neighbor_nodes_with_info[:12]:
                name = node_info['name']
                # Skip nodes with None names
                if name is None:
                    continue
                    
                coord_str = get_optimized_coordinates(node_info)
                node_type = node_info['category'] if node_info.get('category') else node_info['type']
                geom_type = node_info.get('geometry_type', 'unknown geometry')

                attrs = [f"ID: {node_id}", f"type: {node_type}", f"geometry: {geom_type}"]

                # Add detailed attributes
                for attr in ['building use', 'historic district', 'architect',
                                 'building', 'historic', 'housenumber', 'street', 'address','postcode',
                                 'planning_area', 'district', 'city', 'country']:
                    if (attr in node_info and node_info[attr] is not None and
                            not pd.isna(node_info[attr]) and str(node_info[attr]).strip() != "" and
                            str(node_info[attr]).lower() not in ['unknown', 'none', 'null']):
                        attrs.append(f"{attr}={node_info[attr]}")

                direction_info = get_direction_from_closest_center(node_info, center_node_ids, all_nodes_info)
                description += f"- {name} [{', '.join(attrs)}] at {coord_str} {direction_info}\n"

            if len(neighbor_nodes_with_info) > 13:
                description += f"... and {len(neighbor_nodes_with_info) - 13} more connected locations\n"

        # Step 5: Display connections with detailed attributes
        edge_pairs = {}
        for edge in all_edges:
            from_id = edge['from_id']
            to_id = edge['to_id']

            if from_id not in all_nodes_info or to_id not in all_nodes_info:
                continue

            from_node = all_nodes_info[from_id]['name']
            to_node = all_nodes_info[to_id]['name']
            edge_key = (from_node, to_node)

            if edge_key not in edge_pairs:
                edge_pairs[edge_key] = []
            edge_pairs[edge_key].append(edge)

        description += "\nConnections in the network:\n"

        # Helper function for detailed edge attributes (but without intersection names for crossings)
        def get_detailed_edge_attributes(rel_edges, relationship):
            """Build detailed edge attribute descriptions"""
            edge_attrs = []

            # Collect edge types
            edge_types = set()
            for edge in rel_edges:
                edge_type = edge.get('type')
                if edge_type and not pd.isna(edge_type):
                    edge_types.add(edge_type)

            if edge_types:
                edge_attrs.append(f"type: {', '.join(sorted(edge_types))}")

            # Direction information - prioritize edge bearing for "nearest" type
            directions = []
            for edge in rel_edges:
                # For "nearest" type edges, prioritize bearing from edge attributes
                if any('nearest' in str(et).lower() for et in edge_types):
                    edge_bearing = edge.get('bearing')
                    if edge_bearing and not pd.isna(edge_bearing):
                        dir_text = bearing_to_direction(edge_bearing)
                        directions.append(dir_text)
                else:
                    # For other edge types, use existing bearing logic
                    if not pd.isna(edge.get('bearing')):
                        dir_text = bearing_to_direction(edge['bearing'])
                        directions.append(dir_text)

            if directions:
                dir_counts = {}
                for dir in directions:
                    dir_counts[dir] = dir_counts.get(dir, 0) + 1

                if len(dir_counts) == 1:
                    edge_attrs.append(f"direction: {next(iter(dir_counts))}")
                else:
                    most_common_dir = max(dir_counts.items(), key=lambda x: x[1])[0]
                    if dir_counts[most_common_dir] / len(directions) > 0.7:
                        edge_attrs.append(f"primarily {most_common_dir}")
                    else:
                        edge_attrs.append("multiple directions")

            # Distance information - prioritize edge distance for "nearest" type
            distances = []
            for edge in rel_edges:
                # For "nearest" type edges, check for distance attribute in edge data
                if any('nearest' in str(et).lower() for et in edge_types):
                    # Check for distance in edge attributes
                    edge_distance = edge.get('distance')
                    if edge_distance and not pd.isna(edge_distance):
                        distances.append(edge_distance)
                else:
                    # For other edge types, use the existing distance logic
                    if not pd.isna(edge.get('distance')):
                        distances.append(edge['distance'])

            if distances:
                avg_dist = sum(distances) / len(distances)
                if max(distances) - min(distances) < 50:  # Very similar distances
                    if avg_dist < 1000:
                        edge_attrs.append(f"~{int(avg_dist)} meters")
                    else:
                        edge_attrs.append(f"~{round(avg_dist / 1000, 1)} km")
                else:
                    # Show range for varied distances
                    if avg_dist < 1000:
                        edge_attrs.append(f"{int(min(distances))}-{int(max(distances))} meters")
                    else:
                        edge_attrs.append(f"{round(min(distances) / 1000, 1)}-{round(max(distances) / 1000, 1)} km")

            return " [" + ", ".join(edge_attrs) + "]" if edge_attrs else ""

        # Simplified crossing info - no intersection names
        def get_edge_attributes_string(edge_data):
            """Extract non-null edge attributes and format them as a string"""
            edge_attrs = []
            # crossing_id is now processed, not excluded
            exclude_attrs = {'from_id', 'to_id', 'relationship', 'is_reversed', 'id1', 'id2', 'geometry',
                             'name_similarity', 'x', 'y', 'point_geometry', 'nearest_line_point'}

            for attr, value in edge_data.items():
                if (attr not in exclude_attrs and
                        value is not None and
                        not pd.isna(value) and
                        str(value).strip() != "" and
                        str(value).lower() not in ['unknown', 'none', 'null']):
                    if attr == 'bearing':
                        value = bearing_to_direction(int(value))
                        attr = 'direction'
                    elif attr == 'crossing_id':
                        # Get crossing coordinates from nodes_gdf
                        try:
                            crossing_row = nodes_gdf[nodes_gdf['id'] == value]
                            if not crossing_row.empty:
                                crossing_node = crossing_row.iloc[0]
                                if crossing_node.geometry.geom_type == 'Point':
                                    crossing_lon, crossing_lat = crossing_node.geometry.x, crossing_node.geometry.y
                                    value = f"at ({crossing_lon:.6f}, {crossing_lat:.6f})"
                                    attr = 'crossing'
                                else:
                                    continue  # Skip if not a point geometry
                            else:
                                continue  # Skip if crossing not found
                        except:
                            continue  # Skip if any error occurs
                    edge_attrs.append(f"{attr}={value}")

            return "[" + ", ".join(edge_attrs) + "]" if edge_attrs else ""

        def get_min_distance_to_centers(node_ids):
            """Get minimum distance from any of the node IDs to any center node"""
            min_distance = float('inf')
            for node_id in node_ids:
                if node_id in all_nodes_info:
                    node_info = all_nodes_info[node_id]
                    coords = node_info['coordinates']
                    if coords[0] is not None and coords[1] is not None:
                        node_lat, node_lon = coords[1], coords[0]

                        # Calculate distance to each center node
                        for center_id in center_node_ids:
                            if center_id in all_nodes_info:
                                center_coords = all_nodes_info[center_id]['coordinates']
                                if center_coords[0] is not None and center_coords[1] is not None:
                                    center_lat, center_lon = center_coords[1], center_coords[0]
                                    distance_m = degrees_to_meters(node_lat, node_lon, center_lat, center_lon)
                                    if distance_m and distance_m < min_distance:
                                        min_distance = distance_m

            return min_distance if min_distance != float('inf') else float('inf')

        all_mentioned_node_ids = get_all_mentioned_node_ids()
        # Create a sorted list of all nodes by distance for reference
        all_nodes_by_distance = []
        for node_id in all_mentioned_node_ids:
            if node_id in all_nodes_info and node_id not in center_node_ids:
                distance = get_min_distance_to_centers([node_id])
                all_nodes_by_distance.append((node_id, distance))

        # Sort nodes by distance
        all_nodes_by_distance.sort(key=lambda x: x[1])

        # Create a mapping of node_id to distance rank for sorting connections
        node_distance_map = {node_id: distance for node_id, distance in all_nodes_by_distance}

        # Categorize connections
        center_to_center = []
        center_to_neighbor = []
        neighbor_to_neighbor = []

        for (from_node, to_node), edges in edge_pairs.items():
            from_node_ids = [edge['from_id'] for edge in edges]
            to_node_ids = [edge['to_id'] for edge in edges]

            involves_center_from = any(node_id in center_node_ids for node_id in from_node_ids)
            involves_center_to = any(node_id in center_node_ids for node_id in to_node_ids)

            if involves_center_from and involves_center_to:
                center_to_center.append((from_node, to_node, edges))
            elif involves_center_from or involves_center_to:
                # Ensure center node is always the from_node for consistent display
                if involves_center_from:
                    # Center is already from_node - keep as is
                    center_node = from_node
                    neighbor_node = to_node
                    non_center_ids = to_node_ids
                else:
                    # Center is to_node - swap the order
                    center_node = to_node
                    neighbor_node = from_node
                    non_center_ids = from_node_ids
                
                # Add distance info for sorting - get minimum distance of non-center nodes
                min_dist = min((node_distance_map.get(nid, float('inf')) for nid in non_center_ids),
                               default=float('inf'))
                center_to_neighbor.append((center_node, neighbor_node, edges, min_dist))
            else:
                # ENHANCED: Include connection if either node is mentioned in network structure
                from_is_mentioned = any(node_id in all_mentioned_node_ids for node_id in from_node_ids)
                to_is_mentioned = any(node_id in all_mentioned_node_ids for node_id in to_node_ids)
                if from_is_mentioned or to_is_mentioned:
                    # Calculate minimum distance for sorting - use the closer node
                    from_min_dist = min((node_distance_map.get(nid, float('inf')) for nid in from_node_ids),
                                        default=float('inf'))
                    to_min_dist = min((node_distance_map.get(nid, float('inf')) for nid in to_node_ids),
                                      default=float('inf'))
                    min_dist = min(from_min_dist, to_min_dist)
                    neighbor_to_neighbor.append((from_node, to_node, edges, min_dist))

        # Sort connections by distance
        center_to_neighbor.sort(key=lambda x: x[3])  # Sort by distance
        neighbor_to_neighbor.sort(key=lambda x: x[3])  # Sort by distance
        #
        # # Display connections with detailed attributes
        # if center_to_center:
        #     description += "*Connections between image locations:*\n"
        #     for from_node, to_node, edges in center_to_center:
        #         # Skip if either node has None or empty name
        #         if not from_node or not to_node or from_node == 'None' or to_node == 'None':
        #             continue
        #
        #         from_node_desc = get_node_description_for_connection(from_node)
        #         to_node_desc = get_node_description_for_connection(to_node)
        #
        #         edges_by_relationship = {}
        #         for edge in edges:
        #             rel = edge['relationship']
        #             if rel not in edges_by_relationship:
        #                 edges_by_relationship[rel] = []
        #             edges_by_relationship[rel].append(edge)
        #
        #         for relationship, rel_edges in edges_by_relationship.items():
        #             # Build optimized description
        #             distances = []
        #             for edge in rel_edges:
        #                 edge_distance = edge.get('distance')
        #                 if edge_distance and not pd.isna(edge_distance):
        #                     distances.append(edge_distance)
        #
        #             distance_text = ""
        #             if distances:
        #                 avg_dist = sum(distances) / len(distances)
        #                 if avg_dist < 1:
        #                     distance_text = " within 1 meter"
        #                 elif avg_dist < 1000:
        #                     distance_text = f" within {int(avg_dist)} meters"
        #                 else:
        #                     distance_text = f" within {round(avg_dist / 1000, 1)} km"
        #
        #             crossing_info = get_crossing_info(rel_edges[0])
        #
        #             description += f"- {from_node}{from_node_desc} {relationship} {to_node}{to_node_desc}{distance_text}{crossing_info}\n"

        if center_to_neighbor:
            print('-----------------len(center_to_neighbor)', len(center_to_neighbor))
            description += "*Connections from image locations to nearby areas:*\n"
            for from_node, to_node, edges, min_dist in center_to_neighbor:
                # from_node is always the mapillary center node (guaranteed by our logic above)
                # Handle None name for mapillary node
                if from_node is None or from_node == 'None':
                    from_node = 'image location'
                    from_node_desc = ''
                else:
                    from_node_desc = get_node_description_for_connection(from_node)
                
                # Handle None name for neighbor node - skip connection if no meaningful name
                if to_node is None or to_node == 'None' or to_node == '':
                    continue
                    
                to_node_desc = get_node_description_for_connection(to_node)

                # ENHANCED: Only skip if neither node has any meaningful information
                has_meaningful_info = (from_node_desc or to_node_desc or
                                       any(all_nodes_info.get(edge['from_id'], {}).get('type') for edge in edges) or
                                       any(all_nodes_info.get(edge['to_id'], {}).get('type') for edge in edges))

                if not has_meaningful_info:
                    continue

                edges_by_relationship = {}
                for edge in edges:
                    rel = edge['relationship']
                    if rel not in edges_by_relationship:
                        edges_by_relationship[rel] = []
                    edges_by_relationship[rel].append(edge)

                for relationship, rel_edges in edges_by_relationship.items():
                    # Skip if relationship is None
                    if relationship is None:
                        continue
                    
                    # Build optimized description
                    # Only add distance for proximity-based relationships (not for crossing, contains, within, etc.)
                    proximity_keywords = ['near', 'connects', 'adjacent', 'close']
                    should_show_distance = any(keyword in relationship.lower() for keyword in proximity_keywords)
                    
                    distance_text = ""
                    if should_show_distance:
                        # Calculate distance directly from geometries (like calculate_nearest_direction_enhanced)
                        try:
                            from shapely.ops import nearest_points
                            from shapely.geometry import Point
                            from geopy.distance import geodesic
                            
                            # Get center (mapillary) node info - should be the first center node
                            center_node_id = center_node_ids[0] if center_node_ids else None
                            if center_node_id and center_node_id in all_nodes_info:
                                center_info = all_nodes_info[center_node_id]
                                center_coords = center_info.get('coordinates', [])
                                
                                # Get neighbor node ID from edges
                                neighbor_node_ids = []
                                for edge in rel_edges:
                                    from_id = edge.get('from_id')
                                    to_id = edge.get('to_id')
                                    # The neighbor is the one that's NOT the center
                                    if from_id != center_node_id:
                                        neighbor_node_ids.append(from_id)
                                    if to_id != center_node_id:
                                        neighbor_node_ids.append(to_id)
                                
                                if neighbor_node_ids and neighbor_node_ids[0] in all_nodes_info:
                                    neighbor_info = all_nodes_info[neighbor_node_ids[0]]
                                    neighbor_coords = neighbor_info.get('coordinates', [])
                                    
                                    # Create Point for center (mapillary is always a Point)
                                    if len(center_coords) >= 2:
                                        center_lon, center_lat = center_coords[0], center_coords[1]
                                        center_point = Point(center_lon, center_lat)
                                        
                                        # Get neighbor geometry
                                        neighbor_geom_type = neighbor_info.get('geometry_type', 'Point')
                                        
                                        if neighbor_geom_type in ['LineString', 'MultiLineString'] and isinstance(neighbor_coords[0], tuple):
                                            # For LineString, create geometry and find nearest point
                                            from shapely.geometry import LineString
                                            neighbor_line = LineString(neighbor_coords)
                                            nearest_point_on_neighbor = nearest_points(center_point, neighbor_line)[1]
                                            neighbor_lat, neighbor_lon = nearest_point_on_neighbor.y, nearest_point_on_neighbor.x
                                        elif len(neighbor_coords) >= 2:
                                            # For Point, use coordinates directly
                                            if isinstance(neighbor_coords[0], tuple):
                                                neighbor_lon = (neighbor_coords[0][0] + neighbor_coords[1][0]) / 2
                                                neighbor_lat = (neighbor_coords[0][1] + neighbor_coords[1][1]) / 2
                                            else:
                                                neighbor_lon, neighbor_lat = neighbor_coords[0], neighbor_coords[1]
                                        else:
                                            raise ValueError("Invalid neighbor coordinates")
                                        
                                        # Calculate distance using geodesic
                                        distance = geodesic((center_lat, center_lon), (neighbor_lat, neighbor_lon)).meters
                                        
                                        if distance < 1000:
                                            distance_text = f" ({int(distance)}m)"
                                        else:
                                            distance_text = f" ({distance / 1000:.1f}km)"
                        except Exception as e:
                            # Fallback to edge distance if calculation fails
                            distances = []
                            for edge in rel_edges:
                                edge_distance = edge.get('distance')
                                if edge_distance and not pd.isna(edge_distance):
                                    distances.append(edge_distance)
                            
                            if distances:
                                avg_dist = sum(distances) / len(distances)
                                if avg_dist < 1000:
                                    distance_text = f" ({int(avg_dist)}m)"
                                else:
                                    distance_text = f" ({avg_dist / 1000:.1f}km)"
                    
                    crossing_info = get_crossing_info(rel_edges[0])

                    description += f"- {from_node}{from_node_desc} {relationship} {to_node}{to_node_desc}{distance_text}{crossing_info}\n"

        if neighbor_to_neighbor:
            description += "*Connections between nearby areas:*\n"
            # ENHANCED: Show more connections and be less restrictive
            displayed_count = 0
            max_display = 17  # Increased from 15

            for from_node, to_node, edges, min_dist in neighbor_to_neighbor:
                if displayed_count >= max_display:
                    break

                # Skip if either node has None or empty name
                if not from_node or not to_node or from_node == 'None' or to_node == 'None':
                    continue

                from_node_desc = get_node_description_for_connection(from_node)
                to_node_desc = get_node_description_for_connection(to_node)

                # ENHANCED: More lenient filtering - show connection if at least one node has info
                from_has_info = (from_node_desc or
                                 any(all_nodes_info.get(edge['from_id'], {}).get('type') for edge in edges) or
                                 any(all_nodes_info.get(edge['from_id'], {}).get('category') for edge in edges))
                to_has_info = (to_node_desc or
                               any(all_nodes_info.get(edge['to_id'], {}).get('type') for edge in edges) or
                               any(all_nodes_info.get(edge['to_id'], {}).get('category') for edge in edges))

                # Show connection if at least ONE node has meaningful info
                if not (from_has_info or to_has_info):
                    continue

                edges_by_relationship = {}
                for edge in edges:
                    rel = edge['relationship']
                    if rel not in edges_by_relationship:
                        edges_by_relationship[rel] = []
                    edges_by_relationship[rel].append(edge)

                for relationship, rel_edges in edges_by_relationship.items():
                    # Skip if relationship is None
                    if relationship is None:
                        continue
                    
                    # Build optimized description
                    # Only add distance for proximity-based relationships (not for crossing, contains, within, etc.)
                    proximity_keywords = ['near', 'connects', 'adjacent', 'close']
                    should_show_distance = any(keyword in relationship.lower() for keyword in proximity_keywords)
                    
                    distance_text = ""
                    if should_show_distance:
                        distances = []
                        for edge in rel_edges:
                            edge_distance = edge.get('distance')
                            if edge_distance and not pd.isna(edge_distance):
                                distances.append(edge_distance)
                        
                        if distances:
                            avg_dist = sum(distances) / len(distances)
                            if avg_dist < 1000:
                                distance_text = f" ({int(avg_dist)}m)"
                            else:
                                distance_text = f" ({avg_dist / 1000:.1f}km)"
                    
                    crossing_info = get_crossing_info(rel_edges[0])

                    description += f"- {from_node} {relationship} {to_node}{distance_text}{crossing_info}\n"
                    displayed_count += 1

                    if displayed_count >= max_display:
                        break

                if displayed_count >= max_display:
                    break
            remaining_connections = len(neighbor_to_neighbor) - displayed_count
            if remaining_connections > 0:
                description += f"... and {remaining_connections} more connections between areas\n"

        # Add summary if no connections found
        if not (center_to_center or center_to_neighbor or neighbor_to_neighbor):
            description += "- No connections found in the network\n"
        node_categories = {
            'center_nodes': set(center_node_ids),
            'adjacent_nodes': adjacent_nodes,
            'neighbor_nodes': neighbor_nodes,
            'all_essential_nodes': set(center_node_ids) | adjacent_nodes | neighbor_nodes
        }

        return description, node_categories

    def create_downsized_graph_from_categories(self, original_G, node_categories):
        """
        Create downsized graph using pre-computed node categories
        """
        essential_nodes = node_categories['all_essential_nodes']

        # Create the downsized graph as subgraph of essential nodes
        downsized_G = original_G.subgraph(essential_nodes).copy()

        # Calculate statistics
        original_stats = {
            'total_nodes': len(original_G.nodes()),
            'total_edges': len(original_G.edges())
        }

        downsized_stats = {
            'total_nodes': len(downsized_G.nodes()),
            'total_edges': len(downsized_G.edges()),
            'center_nodes': len(node_categories['center_nodes']),
            'adjacent_nodes': len(node_categories['adjacent_nodes']),
            'neighbor_nodes': len(node_categories['neighbor_nodes'])
        }

        # Optional: Print statistics
        print(f"Graph downsizing complete:")
        print(f"Original: {original_stats['total_nodes']} nodes, {original_stats['total_edges']} edges")
        print(f"Downsized: {downsized_stats['total_nodes']} nodes, {downsized_stats['total_edges']} edges")
        print(
            f"Reduction: {((original_stats['total_nodes'] - downsized_stats['total_nodes']) / original_stats['total_nodes'] * 100):.1f}% nodes, "
            f"{((original_stats['total_edges'] - downsized_stats['total_edges']) / original_stats['total_edges'] * 100):.1f}% edges")
        print(
            f"Node breakdown: {downsized_stats['center_nodes']} center + {downsized_stats['adjacent_nodes']} adjacent + {downsized_stats['neighbor_nodes']} neighbor")

        return downsized_G, downsized_stats

    # def process_and_downsize_network(self, G, center_node_ids, nodes_gdf, edges_gdf, max_hops=2):
    #     """
    #     Complete workflow: create description, extract node categories, downsize graph, and save
    #     """
    #     # Step 1: Generate description and get node categories
    #     description, node_categories = self._create_network_description(
    #         G, center_node_ids, nodes_gdf, edges_gdf, max_hops
    #     )
    #
    #     # Step 2: Create downsized graph using the categories
    #     downsized_G, stats = self.create_downsized_graph_from_categories(G, node_categories)
    #
    #     # Step 3: Save downsized graph (example with pickle, adapt to your preferred format)
    #     import pickle
    #
    #     # Save the downsized graph
    #     graph_save_data = {
    #         'graph': downsized_G,
    #         'center_node_ids': list(node_categories['center_nodes']),
    #         'node_categories': node_categories,
    #         'description': description,
    #         'stats': stats
    #     }
    #
    #     # Example save paths (modify as needed)
    #     # pickle.dump(graph_save_data, open(f'downsized_graph_{center_node_ids[0]}.pkl', 'wb'))
    #
    #     return downsized_G, description, node_categories, stats


def get_processed_s2cell_ids(output_jsonl_file):
    """
    Get all processed S2 cell IDs from the output JSONL file.
    
    Args:
        output_jsonl_file: Path to the output JSONL file
        
    Returns:
        Set of processed S2 cell IDs or empty set if file doesn't exist or is empty
    """
    processed_s2cell_ids = set()
    
    if not os.path.exists(output_jsonl_file):
        print(f"📄 Output file doesn't exist yet: {output_jsonl_file}")
        return processed_s2cell_ids
    
    try:
        with open(output_jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        item = json.loads(line)
                        # Add all S2 cell IDs from the batch results
                        if item and len(item) > 0:
                            for s2cell_id in item.keys():
                                processed_s2cell_ids.add(s2cell_id)
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        print(f"⚠️ Error reading output file: {e}")
        return set()
    
    if processed_s2cell_ids:
        print(f"📄 Found {len(processed_s2cell_ids)} already processed S2 cells")
    else:
        print(f"📄 No processed S2 cells found in output file")
    
    return processed_s2cell_ids


def process_image_prompt_data_enhanced_with_paths_and_captioning(jsonl_file, nodes_gdf, edges_gdf, output_dir=None,
                                                                 batch_size=1, subgraph_output_dir='',
                                                                 save_subgraphs=True,
                                                                 use_gpu=True, generate_captions=True, api_key=None, api_provider='qwen',
                                                                 max_image_distance_km=1.0, custom_s2cell_data=None,
                                                                 resume_from_checkpoint=True, clear_checkpoint=False):
    """
    Enhanced processing with path analysis and caption generation
    Added filtering for S2 cells where image nodes are too far apart.

    Args:
        max_image_distance_km: Maximum allowed distance between image nodes in km (default: 1.0)
        resume_from_checkpoint: Whether to resume from checkpoint if available (default: True)
        clear_checkpoint: Whether to clear existing checkpoint and start fresh (default: False)
        api_provider: API provider for caption generation ('qwen' or 'yinli')
    """
    # Read the JSONL file or use custom data
    data_items = []
    if custom_s2cell_data is not None:
        data_items = custom_s2cell_data
        print(f"Using custom S2 cell data with {len(data_items)} items")
    elif jsonl_file:
        with open(jsonl_file, 'r') as f:
            for line in f:
                if line.strip():
                    data_items.append(json.loads(line))
        print(f"Loaded {len(data_items)} items from {jsonl_file}")
    else:
        print("❌ No data source provided (neither jsonl_file nor custom_s2cell_data)")
        return [], {}

    # Create output directories
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if subgraph_output_dir and not os.path.exists(subgraph_output_dir):
        os.makedirs(subgraph_output_dir)

    # Create captions output directory
    captions_output_dir = os.path.join(output_dir, "generated_captions") if output_dir else None
    if captions_output_dir and not os.path.exists(captions_output_dir):
        os.makedirs(captions_output_dir)

    # Create filtered output directory for logging skipped cells
    filtered_log_dir = os.path.join(output_dir, "filtering_logs") if output_dir else None
    if filtered_log_dir and not os.path.exists(filtered_log_dir):
        os.makedirs(filtered_log_dir)

    # Handle checkpoint functionality
    checkpoint_data = None
    start_batch_index = 0
    results = []
    filtering_stats = {
        'total_processed': 0,
        'skipped_distance': 0,
        'skipped_no_images': 0,
        'successfully_processed': 0,
        'skipped_cells': []
    }

    if clear_checkpoint and output_dir:
        clear_progress_checkpoint(output_dir)
        print("🗑️ Checkpoint cleared - starting fresh")

    if resume_from_checkpoint and output_dir:
        checkpoint_data = load_progress_checkpoint(output_dir)
        if checkpoint_data:
            print("🔄 Resuming from checkpoint...")
            # Restore previous state
            start_batch_index = checkpoint_data.get('last_completed_batch_index', 0) + 1
            results = checkpoint_data.get('results', [])
            filtering_stats = checkpoint_data.get('filtering_stats', filtering_stats)
            print(f"   Resuming from batch {start_batch_index}")
            print(f"   Previously processed: {len(results)} batches")
            print(f"   Filtering stats: {filtering_stats}")
        else:
            print("📄 No checkpoint found - starting from beginning")

    # Get all already processed S2 cell IDs to skip them
    output_jsonl_file = os.path.join(output_dir, "enhanced_prompts_with_paths_and_individual_captions.jsonl") if output_dir else None
    processed_s2cell_ids = get_processed_s2cell_ids(output_jsonl_file)
    
    # Also get processed IDs from checkpoint if available
    if checkpoint_data:
        checkpoint_processed_ids = get_processed_s2cell_ids_from_checkpoint(checkpoint_data)
        processed_s2cell_ids.update(checkpoint_processed_ids)
        print(f"   Total processed S2 cells (including checkpoint): {len(processed_s2cell_ids)}")
    
    # Process in batches, skipping already processed S2 cells
    batch_range = range(start_batch_index * batch_size, len(data_items), batch_size)
    for batch_start in tqdm(batch_range, desc="Processing batches with path analysis and captioning"):
        print("---------------------------------")
        print(f"-----------{batch_start}-----------")
        batch_end = min(batch_start + batch_size, len(data_items))
        batch_items = data_items[batch_start:batch_end]

        batch_results = {}

        for s2cell_data in batch_items:
            s2cell_id = s2cell_data['s2cell_id']
            
            # Skip if this S2 cell has already been processed
            if s2cell_id in processed_s2cell_ids:
                print(f"⏭️ Skipping already processed S2 cell: {s2cell_id}")
                continue
            
            filtering_stats['total_processed'] += 1

            print(f"\n🔄 Processing S2 cell with path analysis and captioning: {s2cell_id}")

            # Extract image node IDs - handle both standard and custom data structures
            if 'valid_images' in s2cell_data:
                # Custom data structure from reasoning paths
                image_node_ids = []
                for image_key, image_info in s2cell_data['valid_images'].items():
                    if 'mapillary_node_id' in image_info:
                        image_node_ids.append(int(image_info['mapillary_node_id']))
                    else:
                        # Fallback to standard format
                        image_node_ids.append(int(image_key))
            else:
                # Standard data structure
                image_node_ids = extract_image_node_ids(s2cell_data)

            if not image_node_ids:
                print("No image nodes found, skipping...")
                filtering_stats['skipped_no_images'] += 1
                continue

            print(f"Found {len(image_node_ids)} image nodes: {image_node_ids}")
            image_node_ids = [int(i) for i in  image_node_ids]
            # # NEW: Check distance between image nodes
            # should_skip, max_distance_km, distance_info = check_s2cell_image_node_distances(
            #     s2cell_data, nodes_gdf, max_image_distance_km
            # )

            # if should_skip:
            #     print(
            #         f"❌ Skipping S2 cell {s2cell_id}: Maximum distance between image nodes is {max_distance_km:.2f}km (exceeds {max_image_distance_km}km threshold)")
            #     print(f"   Furthest nodes: {distance_info['max_distance_pair']} ({max_distance_km:.2f}km apart)")
            #
            #     filtering_stats['skipped_distance'] += 1
            #     filtering_stats['skipped_cells'].append({
            #         's2cell_id': s2cell_id,
            #         'reason': 'distance_exceeded',
            #         'max_distance_km': max_distance_km,
            #         'threshold_km': max_image_distance_km,
            #         'image_node_count': len(image_node_ids),
            #         'distance_info': distance_info
            #     })
            #
            #     # Log the skipped cell details
            #     if filtered_log_dir:
            #         skip_log_file = os.path.join(filtered_log_dir, f"skipped_{s2cell_id}_distance.json")
            #         with open(skip_log_file, 'w') as f:
            #             json.dump({
            #                 's2cell_id': s2cell_id,
            #                 'skip_reason': 'distance_exceeded',
            #                 'max_distance_km': max_distance_km,
            #                 'threshold_km': max_image_distance_km,
            #                 'distance_analysis': distance_info,
            #                 'timestamp': datetime.now().isoforedges.parquetmat()
            #             }, f, indent=2)
            #
            #     continue
            # else:
            #     print(
            #         f"✅ Distance check passed: Maximum distance between image nodes is {max_distance_km:.2f}km (within {max_image_distance_km}km threshold)")

            # Determine subgraph parameters
            max_hops = determine_max_hops(len(image_node_ids))

            # Create enhanced subgraph WITH path analysis
            subgraph_data = create_spatial_subgraph_enhanced_with_paths(
                nodes_gdf=nodes_gdf,
                edges_gdf=edges_gdf,
                center_node_ids=image_node_ids,
                max_hops=max_hops
            )
            if subgraph_data:

                print(
                    f"Created subgraph with {subgraph_data['num_nodes']} nodes and {subgraph_data['num_edges']} edges")
                print(f"Path analysis: {subgraph_data.get('path_statistics', {}).get('total_paths', 0)} paths found")
            else:
                continue

            # Process paths with simplified parsing
            parsed_paths_data = None
            # if subgraph_data.get('paths_data') and subgraph_data.get('paths_data', {}).get('total_paths', 0) > 0:
            #     try:
            #         print("🔄 Processing paths with enhanced coordinate parsing...")
            #         parsed_paths_data = process_paths_with_enhanced_parsing(
            #             subgraph_data['paths_data'],
            #             nodes_gdf,
            #             edges_gdf,
            #             subgraph_data['subgraph'],
            #             max_paths_to_parse=5,
            #             use_gpu=use_gpu
            #         )
            #         print(f"✅ Successfully parsed {parsed_paths_data['successful_parses']} paths")
            #
            #     except Exception as e:
            #         print(f"⚠️ Error in path parsing: {e}")
            #         parsed_paths_data = None
            # else:
            #     print("ℹ️ No paths found for parsing")

            # Create enhanced prompt with path information

            try:
                prompt, downsized_G = create_enhanced_caption_prompt_with_paths(
                    s2cell_data,
                    subgraph_data,
                    nodes_gdf,
                    parsed_paths_data,
                )
                print("Generated enhanced prompt with path analysis")

            except Exception as e:
                print(f"Error creating enhanced prompt: {e}")
                continue

            # Collect image information - handle both standard and custom data structures
            if 'valid_images' in s2cell_data:
                images = s2cell_data['valid_images']
                image_paths = []
                for img_info in images.values():
                    if isinstance(img_info, dict) and 'image_path' in img_info:
                        image_paths.append(img_info["image_path"])
                    elif isinstance(img_info, str):
                        image_paths.append(img_info)
                
                if image_paths:
                    print('image_paths sample No.1:', image_paths[0])
                else:
                    print("No valid image paths found in custom data")
                    continue
            else:
                # Standard data structure
                images = s2cell_data['valid_images']
                image_paths = [img_info["image_path"] for img_info in images.values()]
                print('image_paths sample No.1:', image_paths[0])
            if downsized_G:
                subgraph_data['subgraph'] = downsized_G
            
            # Save subgraph with path data first (needed for Swift format)
            subgraph_file = None
            if save_subgraphs and subgraph_output_dir:
                run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{s2cell_id}"
                subgraph_file = os.path.join(subgraph_output_dir, f"{run_id}_with_paths.pkl")

                try:
                    with open(subgraph_file, 'wb') as f:
                        pickle.dump(subgraph_data['subgraph'], f)
                    print(f"Saved enhanced subgraph with paths to {subgraph_file}")
                except Exception as e:
                    print(f"Error saving subgraph: {e}")
                    subgraph_file = None
            
            # Generate captions for each image individually using Qwen VL if requested
            individual_captions = {}
            valid_image_paths = []
            if generate_captions and image_paths:
                try:
                    print(f"🤖 Generating individual captions for {len(image_paths)} images using {api_provider} API...")
                    individual_captions, valid_image_paths = generate_captions_for_individual_images(
                        prompt_text=prompt,
                        image_paths=image_paths,
                        api_key=api_key,
                        api_provider=api_provider,
                        delay_between_calls=3  # Add delay to avoid rate limiting
                    )
                    
                    # Create individual results for each image caption
                    if individual_captions and valid_image_paths:
                        for i, (image_path, individual_caption) in enumerate(individual_captions.items()):
                            # Extract image caption and summarization from individual caption
                            image_caption, summarization_text = extract_captions_from_enhanced_format(individual_caption)
                            
                            # # Use the first (and only) image caption from individual caption
                            # image_caption = image_caption_text[0] if image_caption_text else "No caption available"
                            #
                            print(f"   📝 Extracted for {os.path.basename(image_path)}:")
                            print(f"      Image caption length: {len(image_caption)} chars")
                            print(f"      Summarization length: {len(summarization_text)} chars")
                            
                            swift_message = {
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
                                "images": [image_path],
                                "graphs": [subgraph_file] if subgraph_file else [],
                                "label": 1.0
                            }
                            
                            # Create individual result for this image
                            individual_result = {
                                "prompt": prompt,
                                "s2_cell_id": s2cell_id,
                                "subgraph_path": subgraph_file,
                                "individual_captions": individual_caption,  # Full caption text for this image
                                "image_paths": image_path,  # Single image path for this result
                                "swift_format": swift_message,  # Swift format message for this image
                                "images": images,
                                "subgraph_info": {
                                    "num_nodes": subgraph_data['num_nodes'],
                                    "num_edges": subgraph_data['num_edges'],
                                    "center_nodes": subgraph_data['center_nodes'],
                                    "paths_found": subgraph_data.get('path_statistics', {}).get('total_paths', 0),
                                    "paths_parsed": parsed_paths_data['successful_parses'] if parsed_paths_data else 0
                                },
                            }
                            
                            # Save individual result immediately to JSONL file
                            if output_dir:
                                batch_file = os.path.join(output_dir, "enhanced_prompts_with_paths_and_individual_captions.jsonl")
                                with open(batch_file, 'a') as f:
                                    json.dump(individual_result, f)
                                    f.write('\n')
                                print(f"💾 Saved individual result for {os.path.basename(image_path)} to {batch_file}")
                                
                                # Save swift_message to separate training data file
                                training_data_file = os.path.join(output_dir, "image_caption_training_data.jsonl")
                                with open(training_data_file, 'a') as f:
                                    json.dump(swift_message, f)
                                    f.write('\n')
                                print(f"💾 Saved swift_message for {os.path.basename(image_path)} to {training_data_file}")
                    
                    print(f"✅ Created and saved {len(individual_captions)} individual results")

                except Exception as e:
                    print(f"⚠️ Error generating individual captions: {e}")


            filtering_stats['successfully_processed'] += 1
            print(f"✅ Successfully processed S2 cell {s2cell_id} with path analysis and individual captioning")
            print(f"   Generated {len(individual_captions)} individual captions")
            break

        
        # Save checkpoint after each batch
        if output_dir:
            current_batch_index = batch_start // batch_size
            checkpoint_data = {
                'last_completed_batch_index': current_batch_index,
                'last_completed_batch': {
                    'batch_start': batch_start,
                    'batch_end': batch_end,
                    's2cell_id': s2cell_id
                },
                'filtering_stats': filtering_stats,
                'processed_s2cell_ids': list(processed_s2cell_ids),
                'timestamp': datetime.now().isoformat(),
                'total_data_items': len(data_items),
                'batch_size': batch_size
            }
            save_progress_checkpoint(output_dir, checkpoint_data)



    # print(f"\n📊 Processing Summary:")
    # print(f"   Total S2 cells processed: {filtering_stats['total_processed']}")
    # print(f"   Successfully processed: {filtering_stats['successfully_processed']}")
    # print(f"   Skipped (no images): {filtering_stats['skipped_no_images']}")
    # print(f"   Skipped (distance > {max_image_distance_km}km): {filtering_stats['skipped_distance']}")
    # print(
    #     f"   Success rate: {filtering_stats['successfully_processed'] / filtering_stats['total_processed'] * 100:.1f}%")

    print(f"✅ Processed {len(results)} batches successfully with path analysis and captioning")
    
    # Clear checkpoint on successful completion
    if output_dir:
        clear_progress_checkpoint(output_dir)
        print("🎉 Processing completed successfully - checkpoint cleared")
    
    # Clear nodes dictionary cache to free memory
    clear_nodes_dict_cache()
    
    return results, filtering_stats


# def create_subgraph_description_text(subgraph_data):
#     """Create formatted text description of the subgraph."""
#     descriptions = subgraph_data['descriptions']
#
#     # Summary statistics
#     summary = descriptions['summary']
#     desc_text = f"""
#
# **Node Types in Network:**
# """
#
#     for node_type, count in summary['node_types'].items():
#         desc_text += f"- {node_type}: {count} locations\n"
#
#     # Connection types
#     if summary['edge_types']:
#         desc_text += "\n**Connection Types:**\n"
#         for edge_type, count in summary['edge_types'].items():
#             desc_text += f"- {edge_type}: {count} connections\n"
#
#     # Node attributes (sample of key nodes)
#     desc_text += "\n**Key Location Details:**\n"
#     center_nodes = subgraph_data['center_nodes']
#
#     for node_id in center_nodes[:10]:  # Show details for first 10 center nodes
#         node_id_str = str(node_id)
#         if node_id_str in descriptions['node_attributes']:
#             attrs = descriptions['node_attributes'][node_id_str]
#
#             # Add relevant attributes
#             for attr, value in attrs.items():
#                 if value is not None:
#                     desc_text += f"  * {attr}: {value}\n"
#
#     # Connection triples - show edges involving image nodes AND edges between their connected nodes
#     if descriptions['edge_triples']:
#         desc_text += "\n**Network Connections Involving Image Locations:**\n"
#         image_node_connections = []
#
#         # First, find all nodes that are directly connected to center nodes
#         nodes_connected_to_centers = set()
#         for triple in descriptions['edge_triples']:
#             if triple['subject_id'] in center_nodes:
#                 nodes_connected_to_centers.add(triple['object_id'])
#             elif triple['object_id'] in center_nodes:
#                 nodes_connected_to_centers.add(triple['subject_id'])
#
#         # Now collect edges that either:
#         # 1. Involve center nodes directly, OR
#         # 2. Connect nodes that are connected to center nodes
#         for triple in descriptions['edge_triples']:
#             # Check if either subject_id or object_id is an image node (center node)
#             involves_center = triple['subject_id'] in center_nodes or triple['object_id'] in center_nodes
#
#             # Check if both nodes are connected to center nodes (even if not center nodes themselves)
#             both_connected_to_centers = (triple['subject_id'] in nodes_connected_to_centers and
#                                          triple['object_id'] in nodes_connected_to_centers)
#
#             if involves_center or both_connected_to_centers:
#                 image_node_connections.append(triple)
#
#         # Show connections involving image nodes and their connected nodes
#         for triple in image_node_connections:  # Show all relevant connections
#             desc_text += f"- {triple['subject']} --[{triple['predicate']}]--> {triple['object']}\n"
#
#         # if len(image_node_connections) > 15:
#         #     desc_text += f"... and {len(image_node_connections) - 15} more connections involving image locations\n"
#
#         if not image_node_connections:
#             desc_text += "- No direct connections found between image locations\n"
#
#     return desc_text

def get_mapillary_coordinates(mapillary_id, s2cell_data):
    """
    Get coordinates for a Mapillary image ID from the s2cell data
    """
    try:
        # Search through all relationships to find the mapillary_id
        for rel_type, rel_list in s2cell_data['relationships'].items():
            for rel in rel_list:
                if 'nodes' in rel:
                    for node in rel['nodes']:
                        if node.get('mapillary_id') == mapillary_id:
                            coords = node.get('coordinates', [])
                            if len(coords) >= 2:
                                return coords[1], coords[0]  # lat, lon
        return None, None
    except Exception as e:
        print(f"Error getting coordinates for {mapillary_id}: {e}")
        return None, None


def create_relationship_description_text_with_coordinates(relationships, s2cell_data, nodes_gdf=None):
    """
    Create relationship description with coordinate information and image-to-node mapping
    """
    desc_text = "\n**Relationships Between Image Locations:**\n"

    # First, create a mapping of images to nodes with names
    image_to_node_mapping = {}
    for rel_type, rel_list in relationships.items():
        for rel in rel_list:
            if 'nodes' in rel:
                for node in rel['nodes']:
                    if 'node_id' in node and 'mapillary_id' in node:
                        node_id = node['node_id']
                        mapillary_id = node['mapillary_id']

                        # Get node name
                        node_name = get_node_name(node_id, nodes_gdf) if nodes_gdf is not None else f"Node_{node_id}"

                        # Get coordinates
                        lat, lon = get_mapillary_coordinates(mapillary_id, s2cell_data)

                        image_to_node_mapping[mapillary_id] = {
                            'node_id': node_id,
                            'node_name': node_name,
                            'coordinates': (lat, lon) if lat and lon else None
                        }

    # Show image-to-location mapping
    if image_to_node_mapping:
        desc_text += "\n*Image Location Mapping:*\n"
        for i, (mapillary_id, info) in enumerate(image_to_node_mapping.items(), 1):
            coords_str = f"({info['coordinates'][0]:.6f}, {info['coordinates'][1]:.6f})" if info[
                'coordinates'] else "(coordinates unavailable)"
            desc_text += f"- Image {i} (ID: {mapillary_id}): Located at {info['node_name']} {coords_str}\n"
        desc_text += "\n"

    # Same street relationships
    if relationships.get('same_street_textual'):
        desc_text += "*Same Street Connections:*\n"
        for street_rel in relationships['same_street_textual']:
            street_name = street_rel['street_name']
            nodes = street_rel['nodes']
            desc_text += f"- Street '{street_name}' connects {len(nodes)} image locations:\n"

            for node in nodes:
                mapillary_id = node['mapillary_id']
                node_id = node['node_id']

                # Get node name and coordinates
                node_name = get_node_name(node_id, nodes_gdf) if nodes_gdf is not None else f"Node_{node_id}"
                lat, lon = get_mapillary_coordinates(mapillary_id, s2cell_data)

                if lat and lon:
                    desc_text += f"  * Image at {node_name} - ({lat:.6f}, {lon:.6f})\n"
                else:
                    desc_text += f"  * Image at {node_name} - (coordinates unavailable)\n"

            # Add distances in meters
            if 'distances' in street_rel:
                desc_text += "  * Distances between locations:\n"
                for distance_key, distance_val in street_rel['distances'].items():
                    # Convert degree-based distance to meters
                    distance_meters = distance_val * 111000  # 1 degree ≈ 111km

                    # Try to get node names for the distance key
                    try:
                        node_ids = distance_key.split('-')
                        if len(node_ids) == 2 and nodes_gdf is not None:
                            node1_name = get_node_name(int(node_ids[0]), nodes_gdf)
                            node2_name = get_node_name(int(node_ids[1]), nodes_gdf)
                            desc_text += f"    - {node1_name} to {node2_name}: {distance_meters:.1f} meters\n"
                        else:
                            desc_text += f"    - {distance_key}: {distance_meters:.1f} meters\n"
                    except:
                        desc_text += f"    - {distance_key}: {distance_meters:.1f} meters\n"

    # POI category relationships with coordinates and node names
    if relationships.get('poi_category_groups'):
        desc_text += "\n*Same Category Locations:*\n"
        for poi_rel in relationships['poi_category_groups']:
            category = poi_rel['category_value']
            nodes = poi_rel['nodes']
            desc_text += f"- Category '{category}' includes {len(nodes)} image locations:\n"

            for node in nodes:
                mapillary_id = node['mapillary_id']
                node_id = node['node_id']

                # Get node name and coordinates
                node_name = get_node_name(node_id, nodes_gdf) if nodes_gdf is not None else f"Node_{node_id}"
                lat, lon = get_mapillary_coordinates(mapillary_id, s2cell_data)

                if lat and lon:
                    desc_text += f"  * Image at {node_name} - ({lat:.6f}, {lon:.6f})\n"
                else:
                    desc_text += f"  * Image at {node_name} - (coordinates unavailable)\n"

    # Proximity clusters
    if relationships.get('proximity_clusters'):
        desc_text += "\n*Proximity Clusters:*\n"
        for cluster in relationships['proximity_clusters']:
            cluster_id = cluster['cluster_id']
            nodes = cluster['nodes']
            desc_text += f"- Cluster {cluster_id} contains {len(nodes)} nearby image locations:\n"

            # Show coordinates and node names for cluster locations
            for node in nodes:
                mapillary_id = node['mapillary_id']
                node_id = node['node_id']

                # Get node name and coordinates
                node_name = get_node_name(node_id, nodes_gdf) if nodes_gdf is not None else f"Node_{node_id}"
                lat, lon = get_mapillary_coordinates(mapillary_id, s2cell_data)

                if lat and lon:
                    desc_text += f"  * Image at {node_name} - ({lat:.6f}, {lon:.6f})\n"
                else:
                    desc_text += f"  * Image at {node_name} - (coordinates unavailable)\n"

    # Other relationship types
    for rel_type, rel_list in relationships.items():
        if rel_type not in ['same_street_textual', 'poi_category_groups', 'proximity_clusters'] and rel_list:
            desc_text += f"\n*{rel_type.replace('_', ' ').title()}:*\n"
            desc_text += f"- Found {len(rel_list)} relationship(s) of this type\n"

            # Try to show node names for other relationship types
            for rel in rel_list[:3]:  # Show first 3 examples
                if 'nodes' in rel:
                    node_names = []
                    for node in rel['nodes']:
                        if 'node_id' in node:
                            node_id = node['node_id']
                            node_name = get_node_name(node_id,
                                                      nodes_gdf) if nodes_gdf is not None else f"Node_{node_id}"
                            node_names.append(node_name)
                    if node_names:
                        desc_text += f"  * Images at: {', '.join(node_names[:3])}\n"

    return desc_text


def create_enhanced_caption_prompt_with_paths(s2cell_data, subgraph_data, nodes_gdf, parsed_paths_data=None, ):
    """
    Create enhanced prompt with simplified path analysis
    """
    print("Creating enhanced caption prompt with path analysis...")

    # Extract basic information
    s2cell_id = s2cell_data['s2cell_id']
    # image_count = s2cell_data['metadata']['valid_images_count']

    # # 1. SUBGRAPH DESCRIPTION
    # if nodes_gdf is not None:
    # try:
    # Initialize the QA generator
    qa_generator = SpatialContextQAGenerator()

    # Get the subgraph and other required data
    subgraph = subgraph_data['subgraph']
    center_nodes = subgraph_data['center_nodes']

    # Create a minimal edges_gdf from the subgraph edges for the QA generator
    edges_data = []
    for u, v, data in subgraph.edges(data=True):
        edge_dict = {'id1': u, 'id2': v}
        edge_dict.update(data)
        edges_data.append(edge_dict)
    edges_gdf_subset = pd.DataFrame(edges_data) if edges_data else pd.DataFrame(columns=['id1', 'id2'])

    # Use the enhanced network description
    subgraph_desc, node_categories = qa_generator._create_network_description(
        subgraph, center_nodes, nodes_gdf, edges_gdf_subset
    )
    print("✅ Used SpatialContextQAGenerator for enhanced network description")
    downsized_G = None
    if len(node_categories['all_essential_nodes']) > 200:
        downsized_G, stats = qa_generator.create_downsized_graph_from_categories(subgraph, node_categories)

    #     except Exception as e:
    #         print(f"⚠️ Could not use SpatialContextQAGenerator, falling back to original: {e}")
    #         subgraph_desc = create_subgraph_description_text(subgraph_data)
    # else:
    #     # Fallback to original method
    #     print("ℹ️ nodes_gdf not provided, using original description method")
    #     subgraph_desc = create_subgraph_description_text(subgraph_data)
    # subgraph_desc = create_subgraph_description_text(subgraph_data)

    # 2. RELATIONSHIP DESCRIPTION (improved with coordinates and node names)
    # relationship_desc = create_relationship_description_text_with_coordinates(s2cell_data['relationships'], s2cell_data,
    #                                                                           nodes_gdf)
    #
    # # 3. SIMPLIFIED PATH ANALYSIS
    # path_desc = create_simplified_path_description(subgraph_data, parsed_paths_data)

    # 4. CONSTRUCT ENHANCED PROMPT
    prompt = f"""
You are an advanced vision model tasked with generating detailed captions for urban street images within a comprehensive spatial network context.

# Comprehensive Spatial Context:

## Network Structure:
{subgraph_desc}


# Enhanced Caption Generation Instructions:

You will be provided with 1 street image from the same spatial area from the Network Structure descriptions. The street image is within a spatial context represented with graph information.

For the image, generate a detailed caption. And then summarize those image features based on the understanding of spatial context. 

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
Given the spatial context where the image was taken, the wide multi-lane design and commercial storefronts are typical of Roosevelt Avenue in North Corona. The street signs visible at the corner indicate this is near the intersection with 104 Street. The density of retail establishments and the urban setting with mixed-use buildings are characteristic of this commercial district, with architectural styles common to this part of Queens."

**Instructions:**
- For the image provided, generate "**Image:**" symbol before captions
- End with a "**Summarization:**" section
- Connect individual observations to the broader spatial understanding and connectivity patterns
- Refer to the places and locations in network descriptions when describing the image
- Also consider the interactive effects of visual, acoustic, and olfactory environments on mental wellbeing and emotional responses
"""

    return prompt, downsized_G


def select_optimal_image_nodes(subgraph, image_node_ids, max_nodes=4):
    """
    Select optimal image nodes based on shortest path traversal criteria

    Args:
        subgraph: NetworkX subgraph
        image_node_ids: List of all image node IDs
        max_nodes: Maximum number of nodes to select (default: 4)

    Returns:
        List of selected image node IDs and selection info
    """
    print(f"🎯 Selecting optimal image nodes from {len(image_node_ids)} candidates")

    if len(image_node_ids) <= max_nodes:
        print(f"✅ Using all {len(image_node_ids)} image nodes (within limit)")
        return image_node_ids, {
            'selection_method': 'all_nodes',
            'total_available': len(image_node_ids),
            'total_selected': len(image_node_ids),
            'selection_criteria': f'All nodes used (≤{max_nodes})'
        }

    print(f"🔍 Need to select {max_nodes} nodes from {len(image_node_ids)} candidates")

    # Try to find the combination of 4 nodes with shortest traversal path
    best_combination = None
    best_path_length = float('inf')
    best_path = None

    # Generate all combinations of max_nodes from image_node_ids
    from itertools import combinations, permutations

    print(f"🧮 Evaluating {len(list(combinations(image_node_ids, max_nodes)))} combinations...")

    for combination in combinations(image_node_ids, max_nodes):
        # For each combination, try to find the shortest path that visits all nodes
        shortest_traversal = find_shortest_traversal_path(subgraph, list(combination))

        if shortest_traversal and shortest_traversal['total_length'] < best_path_length:
            best_combination = list(combination)
            best_path_length = shortest_traversal['total_length']
            best_path = shortest_traversal

    if best_combination:
        print(f"✅ Selected optimal combination: {best_combination}")
        print(f"📏 Best traversal path length: {best_path_length} hops")

        selection_info = {
            'selection_method': 'shortest_traversal',
            'total_available': len(image_node_ids),
            'total_selected': len(best_combination),
            'selection_criteria': f'Shortest path traversing {max_nodes} nodes',
            'best_path_length': best_path_length,
            'best_traversal_path': best_path,
            'rejected_nodes': [node for node in image_node_ids if node not in best_combination]
        }

        return best_combination, selection_info
    else:
        # Fallback: select first max_nodes nodes
        print(f"⚠️ No traversal path found, using first {max_nodes} nodes")
        selected = image_node_ids[:max_nodes]

        selection_info = {
            'selection_method': 'fallback_first_n',
            'total_available': len(image_node_ids),
            'total_selected': len(selected),
            'selection_criteria': f'Fallback: first {max_nodes} nodes',
            'rejected_nodes': image_node_ids[max_nodes:]
        }

        return selected, selection_info


def find_paths_through_image_nodes(subgraph, image_node_ids, max_paths_per_pair=3, cutoff=10, max_image_nodes=4):
    """
    Find paths that traverse through multiple image nodes within the subgraph
    Now includes logic to select optimal image nodes and find traversal paths
    """
    print(f"🛣️ Finding paths through {len(image_node_ids)} image nodes")

    # Step 1: Select optimal image nodes if needed
    selected_image_nodes, selection_info = select_optimal_image_nodes(
        subgraph, image_node_ids, max_nodes=max_image_nodes
    )

    print(f"📍 Working with {len(selected_image_nodes)} selected image nodes: {selected_image_nodes}")

    paths_data = {
        'original_image_nodes': image_node_ids,
        'selected_image_nodes': selected_image_nodes,
        'selection_info': selection_info,
        'node_pairs': [],
        'paths_by_pair': {},
        'all_paths': [],
        'traversal_path': None,
        'path_statistics': {}
    }

    # Step 2: Find traversal path through all selected nodes
    print(f"\n🔍 Finding traversal path through all {len(selected_image_nodes)} selected nodes...")
    traversal_result = find_shortest_traversal_path(subgraph, selected_image_nodes)

    if traversal_result:
        paths_data['traversal_path'] = traversal_result
        print(f"✅ Found traversal path with {traversal_result['total_length']} total hops")
        print(f"📍 Traversal order: {' → '.join(map(str, traversal_result['path']))}")

        # Add traversal segments to all_paths
        for segment in traversal_result['segments']:
            paths_data['all_paths'].append({
                'source': segment['from'],
                'target': segment['to'],
                'path': segment['path'],
                'length': len(segment['path']),
                'hops': segment['length'],
                'path_type': 'traversal_segment'
            })
    else:
        print("❌ No traversal path found through all selected nodes")

    # Step 3: Find pairwise paths between selected nodes (original functionality)
    node_pairs = list(combinations(selected_image_nodes, 2))
    paths_data['node_pairs'] = [(int(a), int(b)) for a, b in node_pairs]

    print(f"\n📊 Finding pairwise paths between {len(node_pairs)} node pairs...")

    total_paths_found = 0

    for source, target in node_pairs:
        print(f"🔍 Finding paths between {source} and {target}")

        # Find paths between this pair
        paths = find_simple_paths_generator(
            subgraph, source, target,
            cutoff=cutoff,
            max_paths=max_paths_per_pair
        )

        if paths:
            pair_key = f"{source}_{target}"
            paths_data['paths_by_pair'][pair_key] = {
                'source': source,
                'target': target,
                'paths': paths,
                'path_count': len(paths),
                'path_lengths': [len(path) for path in paths]
            }

            # Add to all paths list
            for path in paths:
                paths_data['all_paths'].append({
                    'source': source,
                    'target': target,
                    'path': path,
                    'length': len(path),
                    'hops': len(path) - 1,
                    'path_type': 'pairwise'
                })

            total_paths_found += len(paths)
            print(f"✅ Found {len(paths)} paths between {source} and {target}")
        else:
            print(f"❌ No paths found between {source} and {target}")

    # Step 4: Calculate statistics
    if paths_data['all_paths']:
        path_lengths = [p['length'] for p in paths_data['all_paths']]
        paths_data['path_statistics'] = {
            'total_paths': total_paths_found,
            'total_pairs_processed': len(node_pairs),
            'pairs_with_paths': len(paths_data['paths_by_pair']),
            'average_path_length': np.mean(path_lengths),
            'min_path_length': min(path_lengths),
            'max_path_length': max(path_lengths),
            'median_path_length': np.median(path_lengths),
            'has_traversal_path': traversal_result is not None,
            'traversal_path_length': traversal_result['total_length'] if traversal_result else None
        }
    else:
        paths_data['path_statistics'] = {
            'total_paths': 0,
            'total_pairs_processed': len(node_pairs),
            'pairs_with_paths': 0,
            'has_traversal_path': traversal_result is not None,
            'traversal_path_length': traversal_result['total_length'] if traversal_result else None,
            'error': 'No paths found between any image nodes'
        }

    # Summary
    print(f"\n📊 Path Finding Summary:")
    print(f"   • Original image nodes: {len(image_node_ids)}")
    print(f"   • Selected image nodes: {len(selected_image_nodes)}")
    print(f"   • Selection method: {selection_info['selection_method']}")
    print(f"   • Traversal path found: {'Yes' if traversal_result else 'No'}")
    if traversal_result:
        print(f"   • Traversal path length: {traversal_result['total_length']} hops")
    print(f"   • Pairwise paths found: {total_paths_found}")

    return paths_data


def process_paths_with_enhanced_parsing(paths_data, nodes_gdf, edges_gdf, subgraph,
                                        max_paths_to_parse=10, use_gpu=True):
    """
    Simplified path processing - just get the formatted enhanced paths
    """
    print(f"🔄 Processing paths with enhanced coordinate parsing")

    # Initialize GraphProcessor with the subgraph
    graph_processor = GraphProcessor(use_gpu=use_gpu)
    graph_processor.nx_graph = subgraph

    # Convert to cuGraph if GPU enabled
    if graph_processor.use_gpu:
        graph_processor._convert_to_cugraph()

    print(f"✅ Initialized GraphProcessor with {subgraph.number_of_nodes()} nodes, {subgraph.number_of_edges()} edges")

    # Try to import the enhanced parsing function
    try:
        from generate_routes import (
            parse_single_path_directions_enhanced_with_coordinates,
        )
        enhanced_parsing_available = True
        print("✅ Enhanced parsing functions available")
    except ImportError:
        print("⚠️ Enhanced parsing functions not available, using simple parsing")
        enhanced_parsing_available = False

    # Simple result structure - just store the formatted paths
    parsed_paths = {
        'formatted_paths': [],
        'total_paths': len(paths_data['all_paths']),
        'successful_parses': 0,
        'failed_parses': 0
    }

    # Sort paths by length and limit for performance
    sorted_paths = sorted(paths_data['all_paths'], key=lambda x: x['length'])
    paths_to_parse = sorted_paths[:max_paths_to_parse]

    print(f"📝 Parsing {len(paths_to_parse)} paths")

    for i, path_info in enumerate(paths_to_parse):
        path_nodes = path_info['path']
        source = path_info['source']
        target = path_info['target']

        try:
            if enhanced_parsing_available:
                # Use enhanced parsing
                parsed_result = parse_single_path_directions_enhanced_with_coordinates(
                    path_nodes, nodes_gdf, edges_gdf, graph_processor
                )

                if parsed_result.get('success', False):
                    formatted_path = parsed_result.get('formatted_path_enhanced', '')
                    if formatted_path:
                        parsed_paths['formatted_paths'].append(formatted_path)
                        parsed_paths['successful_parses'] += 1
                        print(f"    ✅ Path {i + 1}: Enhanced parsing successful")
                    else:
                        # Fallback to simple
                        simple_path = create_simple_path_description(source, target, path_nodes, nodes_gdf)
                        parsed_paths['formatted_paths'].append(simple_path)
                        parsed_paths['successful_parses'] += 1
                        print(f"    ⚠️ Path {i + 1}: Used simple fallback")
                else:
                    # Fallback to simple
                    simple_path = create_simple_path_description(source, target, path_nodes, nodes_gdf)
                    parsed_paths['formatted_paths'].append(simple_path)
                    parsed_paths['successful_parses'] += 1
                    print(f"    ⚠️ Path {i + 1}: Enhanced failed, used simple")
            else:
                # Use simple parsing
                simple_path = create_simple_path_description(source, target, path_nodes, nodes_gdf)
                parsed_paths['formatted_paths'].append(simple_path)
                parsed_paths['successful_parses'] += 1
                print(f"    ✅ Path {i + 1}: Simple parsing")

        except Exception as e:
            parsed_paths['failed_parses'] += 1
            print(f"    ❌ Path {i + 1}: Exception - {e}")

    print(
        f"📊 Parsing complete: {parsed_paths['successful_parses']} successful, {parsed_paths['failed_parses']} failed")
    return parsed_paths


def get_node_type_distribution(subgraph, nodes_gdf):
    """Get distribution of node types in the subgraph."""
    type_counts = {}
    for node_id in subgraph.nodes():
        node_row = nodes_gdf[nodes_gdf['id'] == node_id]
        if not node_row.empty:
            node_type = node_row.iloc[0].get('type', 'unknown')
            if pd.notna(node_type):
                type_counts[str(node_type)] = type_counts.get(str(node_type), 0) + 1
    return type_counts


def get_edge_type_distribution(subgraph):
    """Get distribution of edge types in the subgraph."""
    type_counts = {}
    for _, _, edge_data in subgraph.edges(data=True):
        edge_type = edge_data.get('type', 'unknown')
        if pd.notna(edge_type):
            type_counts[str(edge_type)] = type_counts.get(str(edge_type), 0) + 1
    return type_counts


def generate_subgraph_descriptions(subgraph, nodes_gdf, edges_gdf, center_node_ids):
    """
    Generate detailed descriptions of the subgraph including node attributes and edge triples.
    """
    descriptions = {
        'node_attributes': {},
        'edge_triples': [],
        'summary': {}
    }

    print("Generating subgraph descriptions...")

    # 1. NODE ATTRIBUTES
    print("  Processing node attributes...")
    for node_id in subgraph.nodes():
        node_row = nodes_gdf[nodes_gdf['id'] == node_id]
        if not node_row.empty:
            node_data = node_row.iloc[0]

            # Extract non-null attributes
            attributes = {}
            for col in node_data.index:
                if col != 'geometry' and pd.notna(node_data[col]):
                    # Convert numpy types to Python types for JSON serialization
                    value = node_data[col]
                    if isinstance(value, (np.integer, np.int64)):
                        value = int(value)
                    elif isinstance(value, (np.floating, np.float64)):
                        value = float(value)
                    else:
                        value = str(value)
                    attributes[col] = value

            descriptions['node_attributes'][str(node_id)] = attributes

    # 2. EDGE TRIPLES
    print("  Processing edge triples...")
    for edge in subgraph.edges(data=True):
        node1_id, node2_id, edge_data = edge

        # Get node names
        node1_name = get_node_name(node1_id, nodes_gdf)
        node2_name = get_node_name(node2_id, nodes_gdf)

        # Get edge type
        edge_type = edge_data.get('type', 'connected_to')

        # Create triple: (node1_name, edge_type, node2_name)
        triple = {
            'subject': node1_name,
            'predicate': edge_type,
            'object': node2_name,
            'subject_id': int(node1_id),
            'object_id': int(node2_id)
        }

        descriptions['edge_triples'].append(triple)

    # 3. SUMMARY STATISTICS
    descriptions['summary'] = {
        'total_nodes': len(subgraph.nodes()),
        'total_edges': len(subgraph.edges()),
        'center_nodes': center_node_ids,
        'node_types': get_node_type_distribution(subgraph, nodes_gdf),
        'edge_types': get_edge_type_distribution(subgraph)
    }

    print(
        f"  Generated descriptions for {len(descriptions['node_attributes'])} nodes and {len(descriptions['edge_triples'])} edges")

    return descriptions


def create_spatial_subgraph_enhanced_alternative(nodes_gdf, edges_gdf, center_node_ids, max_hops=2):
    """
    Alternative optimized approach using pandas operations for very large datasets
    """
    print(f"Creating enhanced spatial subgraph (alternative method) for {len(center_node_ids)} center nodes")

    # Step 1: Initialize with center nodes
    subgraph_nodes = set(center_node_ids)
    current_nodes = set(center_node_ids)

    # Step 2: Expand iteratively using pandas operations
    for hop in range(max_hops):
        # Find all edges connected to current nodes (much faster with pandas)
        mask = edges_gdf['id1'].isin(current_nodes) | edges_gdf['id2'].isin(current_nodes)
        connected_edges = edges_gdf[mask]

        if connected_edges.empty:
            print(f"No more connections found at hop {hop + 1}")
            break

        # Get all connected nodes
        connected_id1 = set(connected_edges['id1'].unique())
        connected_id2 = set(connected_edges['id2'].unique())
        all_connected = connected_id1.union(connected_id2)

        # Find new nodes (not already in subgraph)
        new_nodes = all_connected - subgraph_nodes

        if not new_nodes:
            print(f"No new nodes found at hop {hop + 1}")
            break

        # Add new nodes to subgraph
        subgraph_nodes.update(new_nodes)
        current_nodes = new_nodes

        print(f"Hop {hop + 1}: Added {len(new_nodes)} new nodes (total: {len(subgraph_nodes)})")

    # Step 3: Create final subgraph
    # Filter nodes and edges
    subgraph_nodes_df = nodes_gdf[nodes_gdf['id'].isin(subgraph_nodes)].copy()
    subgraph_edges_df = edges_gdf[
        edges_gdf['id1'].isin(subgraph_nodes) &
        edges_gdf['id2'].isin(subgraph_nodes)
        ].copy()

    print(f"Filtered to {len(subgraph_nodes_df)} nodes and {len(subgraph_edges_df)} edges")

    # Step 4: Build NetworkX graph
    G = nx.Graph()

    # Add nodes
    for _, node_row in subgraph_nodes_df.iterrows():
        node_id = node_row['id']
        node_attrs = {col: node_row[col] for col in node_row.index
                      if col != 'geometry' and pd.notna(node_row[col])}
        G.add_node(node_id, **node_attrs)

    # Add edges
    for _, edge_row in subgraph_edges_df.iterrows():
        edge_attrs = {col: edge_row[col] for col in edge_row.index
                      if col not in ['id1', 'id2', 'geometry'] and pd.notna(edge_row[col])}
        G.add_edge(edge_row['id1'], edge_row['id2'], **edge_attrs)

    print(f"Created alternative subgraph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

    # Generate descriptions
    descriptions = generate_subgraph_descriptions(G, nodes_gdf, edges_gdf, center_node_ids)

    result = {
        'subgraph': G,
        'center_nodes': center_node_ids,
        'subgraph_nodes': list(subgraph_nodes),
        'descriptions': descriptions,
        'num_nodes': len(subgraph_nodes),
        'num_edges': G.number_of_edges(),
        'optimization_info': {
            'method': 'pandas_optimized_subgraph',
            'nodes_processed': len(subgraph_nodes),
            'edges_processed': len(subgraph_edges_df),
            'efficiency_gain': f"Processed {len(subgraph_nodes)}/{len(nodes_gdf)} nodes ({len(subgraph_nodes) / len(nodes_gdf) * 100:.1f}%)"
        }
    }

    return result


def create_spatial_subgraph_enhanced(nodes_gdf, edges_gdf, center_node_ids, max_hops=2):
    """
    Enhanced subgraph creation with detailed descriptions - optimized to build subgraph directly from center nodes
    """
    print(f"Creating enhanced spatial subgraph for {len(center_node_ids)} center nodes with {max_hops} hops")

    # Step 1: Start with center nodes and expand iteratively
    subgraph_nodes = set(center_node_ids)

    # Create node lookup for efficient access (use cached version)
    nodes_dict = get_nodes_dict(nodes_gdf)

    print(f"Starting with {len(center_node_ids)} center nodes")

    # Step 2: Iteratively expand by hops
    current_frontier = set(center_node_ids)

    for hop in range(max_hops):
        next_frontier = set()

        # Find all edges connected to current frontier nodes
        connected_edges = edges_gdf[
            (edges_gdf['id1'].isin(current_frontier)) |
            (edges_gdf['id2'].isin(current_frontier))
            ]

        # Add all connected nodes to next frontier
        for _, edge_row in connected_edges.iterrows():
            id1, id2 = edge_row['id1'], edge_row['id2']

            # Add the other end of each edge
            if id1 in current_frontier and id2 not in subgraph_nodes:
                next_frontier.add(int(id2))
            if id2 in current_frontier and id1 not in subgraph_nodes:
                next_frontier.add(int(id1))

        # Add new nodes to subgraph
        subgraph_nodes.update(next_frontier)

        print(f"Hop {hop + 1}: Added {len(next_frontier)} new nodes (total: {len(subgraph_nodes)})")

        # Prepare for next iteration
        current_frontier = next_frontier

        # If no new nodes found, stop early
        if not next_frontier:
            print(f"No more connected nodes found at hop {hop + 1}, stopping early")
            break

    print(f"Final subgraph contains {len(subgraph_nodes)} nodes")

    # Step 3: Filter edges that are within the subgraph
    subgraph_edges = edges_gdf[
        (edges_gdf['id1'].isin(subgraph_nodes)) &
        (edges_gdf['id2'].isin(subgraph_nodes))
        ]
    if len(subgraph_edges) > 20000:
        return None
    print(f"Found {len(subgraph_edges)} edges within subgraph")
    
    # Step 4: Create NetworkX graph with only the subgraph nodes and edges
    G = nx.Graph()

    # Add only the nodes in our subgraph
    for node_id in subgraph_nodes:
        if node_id in nodes_dict:
            node_row = nodes_dict[node_id]
            node_attrs = {}

            # Extract non-null attributes
            for col in node_row.index:
                if col != 'geometry' and pd.notna(node_row[col]):
                    node_attrs[col] = node_row[col]

            G.add_node(node_id, **node_attrs)
        # else:
        #     # Add node with minimal info if not found in nodes_gdf
        #     G.add_node(node_id, id=node_id, name=f"Node_{node_id}")
        #     print(f"Warning: Node {node_id} not found in nodes_gdf, added with minimal attributes")

    # Add only the edges within our subgraph
    for _, edge_row in subgraph_edges.iterrows():
        edge_attrs = {}
        for col in edge_row.index:
            if col not in ['id1', 'id2', 'geometry'] and pd.notna(edge_row[col]):
                edge_attrs[col] = edge_row[col]
        G.add_edge(edge_row['id1'], edge_row['id2'], **edge_attrs)

    print(f"Created optimized subgraph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

    # Step 5: Generate descriptions
    descriptions = generate_subgraph_descriptions(G, nodes_gdf, edges_gdf, center_node_ids)

    # Step 6: Create result dictionary
    result = {
        'subgraph': G,
        'center_nodes': center_node_ids,
        'subgraph_nodes': list(subgraph_nodes),
        'descriptions': descriptions,
        'num_nodes': len(subgraph_nodes),
        'num_edges': G.number_of_edges(),
        'optimization_info': {
            'method': 'direct_subgraph_creation',
            'hops_processed': min(hop + 1, max_hops),
            'early_termination': hop + 1 < max_hops and not next_frontier,
            'efficiency_gain': f"Processed {len(subgraph_nodes)} nodes instead of {len(nodes_gdf)}"
        }
    }

    return result


def create_spatial_subgraph_enhanced_with_paths(nodes_gdf, edges_gdf, center_node_ids, max_hops=2, max_image_nodes=4):
    """
    Enhanced subgraph creation with path analysis and optimal node selection
    """
    print(f"Creating enhanced spatial subgraph with path analysis for {len(center_node_ids)} center nodes")

    # Create basic subgraph
    subgraph_result = create_spatial_subgraph_enhanced(nodes_gdf, edges_gdf, center_node_ids, max_hops)

    # # Find paths through image nodes with optimal selection
    # print(f"\n🛣️ Finding optimal paths through image nodes...")
    # paths_data = None  # Initialize paths_data to avoid UnboundLocalError
    #
    # if len(center_node_ids) > 2:
    #     try:
    #         paths_data = find_paths_through_image_nodes(
    #             subgraph_result['subgraph'],
    #             center_node_ids,
    #             max_paths_per_pair=3,
    #             cutoff=max_hops + 2,
    #             max_image_nodes=max_image_nodes  # add Add the parameter
    #         )
    #         # Update the result with selected nodes information
    #         subgraph_result['paths_data'] = paths_data
    #         subgraph_result['path_statistics'] = paths_data['path_statistics']
    #         subgraph_result['selected_image_nodes'] = paths_data['selected_image_nodes']
    #         subgraph_result['original_center_nodes'] = center_node_ids
    #         subgraph_result['node_selection_info'] = paths_data['selection_info']
    #     except Exception as e:
    #         print(f"Error in find_paths_through_image_nodes: {e}")
    #         subgraph_result['paths_data'] = None
    #         subgraph_result['path_statistics'] = None
    #         subgraph_result['selected_image_nodes'] = None
    #         subgraph_result['original_center_nodes'] = center_node_ids
    #
    #     print(
    #         f"✅ Enhanced subgraph created with {subgraph_result['num_nodes']} nodes, {subgraph_result['num_edges']} edges")
    #
    #     # Safe access to paths_data with proper error handling
    #     if paths_data and paths_data.get('path_statistics'):
    #         print(f"📊 Path analysis: {paths_data['path_statistics'].get('total_paths', 0)} paths found")
    #     else:
    #         print("📊 Path analysis: No paths found or path analysis failed")
    #
    #     if paths_data and paths_data.get('traversal_path'):
    #         print(
    #             f"🎯 Traversal path: {paths_data['traversal_path']['total_length']} hops through {len(paths_data['selected_image_nodes'])} nodes")

    return subgraph_result


def extract_image_node_ids(s2cell_data):
    """Extract image node IDs from s2cell data"""
    if 'valid_images' in s2cell_data:
        valid_images = s2cell_data['valid_images']
        return [info['node_id'] for info in valid_images.values()]
    else:
        image_node_ids = []
        for rel_type, rel_list in s2cell_data['relationships'].items():
            for rel in rel_list:
                if 'nodes' in rel:
                    for node in rel['nodes']:
                        if 'node_id' in node:
                            image_node_ids.append(node['node_id'])
        return list(set(image_node_ids))


def get_node_name(node_id, nodes_gdf):
    """Get the name of a node, with fallbacks if name is not available."""
    node_row = nodes_gdf[nodes_gdf['id'] == node_id]
    if not node_row.empty:
        node_data = node_row.iloc[0]

        # Try different name fields in order of preference
        for name_field in ['name', 'address', 'street', 'type', 'category', 'building use', 'historic district',
                           'architect',
                           'city', 'postcode', 'building', 'historic', 'turn_count', 'length_meters', ]:
            if name_field in node_data and pd.notna(node_data[name_field]):
                return str(node_data[name_field])

    # Fallback to node ID
    return f"Node_{node_id}"


def create_simple_path_description(source, target, path_nodes, nodes_gdf):
    """Create a simple path description"""
    source_name = get_node_name(source, nodes_gdf)
    target_name = get_node_name(target, nodes_gdf)
    return f"Path from {source_name} to {target_name} ({len(path_nodes)} nodes)"


def determine_max_hops(num_nodes):
    """Determine max hops based on number of nodes"""
    if num_nodes <= 3:
        return 2
    elif num_nodes <= 10:
        return 2
    elif num_nodes == 1:
        return 2
    else:
        return 3


def collect_image_information(s2cell_data):
    """Collect image information from s2cell data"""
    images = {}
    for k, v in s2cell_data['valid_images'].items():

        if 'node_id' in v and 'mapillary_id' in v:
            node_id = v['node_id']
            images[node_id] = {
                "mapillary_id": v['mapillary_id'],
                "coordinates": v.get('coordinates', []),
                "image_path": os.path.join(DATA_ROOT, 'geo', 'SR', 'osm_data', 'singapore', 'images', f"{v['mapillary_id']}.jpg")
            }
    return images


def process_image_prompt_data_enhanced_with_paths(jsonl_file, nodes_gdf, edges_gdf, output_dir=None,
                                                  batch_size=1, subgraph_output_dir='', save_subgraphs=True,
                                                  use_gpu=True):
    """
    Enhanced processing with simplified path analysis
    """
    # Read the JSONL file
    data_items = []
    with open(jsonl_file, 'r') as f:
        for line in f:
            if line.strip():
                data_items.append(json.loads(line))

    print(f"Loaded {len(data_items)} items from {jsonl_file}")

    # Create output directories
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if subgraph_output_dir and not os.path.exists(subgraph_output_dir):
        os.makedirs(subgraph_output_dir)

    results = []

    # Process in batches
    for batch_start in tqdm(range(0, len(data_items), batch_size), desc="Processing batches with path analysis"):
        batch_end = min(batch_start + batch_size, len(data_items))
        batch_items = data_items[batch_start:batch_end]

        batch_results = {}

        for s2cell_data in batch_items:
            print('s2cell_data')
            print(s2cell_data)
            s2cell_id = s2cell_data['metadata']['s2cell_id']

            print(f"\n🔄 Processing S2 cell with path analysis: {s2cell_id}")

            # Extract image node IDs
            image_node_ids = extract_image_node_ids(s2cell_data)

            if not image_node_ids:
                print("No image nodes found, skipping...")
                continue

            print(f"Found {len(image_node_ids)} image nodes: {image_node_ids}")
            image_node_ids = [int(i) for i in image_node_ids]
            # Determine subgraph parameters
            max_hops = determine_max_hops(len(image_node_ids))

            # Create enhanced subgraph WITH path analysis
            try:
                subgraph_data = create_spatial_subgraph_enhanced_with_paths(
                    nodes_gdf=nodes_gdf,
                    edges_gdf=edges_gdf,
                    center_node_ids=image_node_ids,
                    max_hops=max_hops
                )

                print(
                    f"Created subgraph with {subgraph_data['num_nodes']} nodes and {subgraph_data['num_edges']} edges")
                # print(f"Path analysis: {subgraph_data.get('path_statistics', {}).get('total_paths', 0)} paths found")

            except Exception as e:
                print(f"Error creating subgraph with paths: {e}")
                continue

            # Process paths with simplified parsing
            parsed_paths_data = None
            # if subgraph_data.get('paths_data') and subgraph_data.get('paths_data', {}).get('total_paths', 0) > 0:
            #     try:
            #         print("🔄 Processing paths with enhanced coordinate parsing...")
            #         parsed_paths_data = process_paths_with_enhanced_parsing(
            #             subgraph_data['paths_data'],
            #             nodes_gdf,
            #             edges_gdf,
            #             subgraph_data['subgraph'],
            #             max_paths_to_parse=5,  # Reduced for simplicity
            #             use_gpu=use_gpu
            #         )
            #         print(f"✅ Successfully parsed {parsed_paths_data['successful_parses']} paths")
            #
            #     except Exception as e:
            #         print(f"⚠️ Error in path parsing: {e}")
            #         parsed_paths_data = None
            # else:
            #     print("ℹ️ No paths found for parsing")

            # Create enhanced prompt with path information
            try:
                prompt, downsized_subgraph = create_enhanced_caption_prompt_with_paths(
                    s2cell_data,
                    subgraph_data,
                    nodes_gdf,
                    parsed_paths_data
                )
                print("Generated enhanced prompt with path analysis")

            except Exception as e:
                print(f"Error creating enhanced prompt: {e}")
                continue

            # Save subgraph with path data
            subgraph_file = None
            if save_subgraphs and subgraph_output_dir:
                run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{s2cell_id}"
                subgraph_file = os.path.join(subgraph_output_dir, f"{run_id}_with_paths.pkl")

                # Include parsed paths in saved data
                print('downsized_subgraph', downsized_subgraph)
                save_data = {
                    'subgraph_data': downsized_subgraph,
                    'parsed_paths_data': parsed_paths_data,
                    'processing_timestamp': datetime.now().isoformat(),
                    's2cell_id': s2cell_id
                }

                try:
                    with open(subgraph_file, 'wb') as f:
                        pickle.dump(save_data, f)
                    print(f"Saved enhanced subgraph with paths to {subgraph_file}")
                except Exception as e:
                    print(f"Error saving subgraph: {e}")
                    subgraph_file = None

            # Collect image information
            images = collect_image_information(s2cell_data)

            # Store enhanced results
            batch_results[s2cell_id] = {
                "prompt": prompt,
                "images": images,
                "subgraph_file": subgraph_file,
                "subgraph_info": {
                    "num_nodes": downsized_subgraph['num_nodes'],
                    "num_edges": downsized_subgraph['num_edges'],
                    "center_nodes": downsized_subgraph['center_nodes'],
                    # "paths_found": subgraph_data.get('path_statistics', {}).get('total_paths', 0),
                    "paths_parsed": parsed_paths_data['successful_parses'] if parsed_paths_data else 0
                },
                "path_analysis_included": parsed_paths_data is not None,
                "formatted_paths_count": len(parsed_paths_data['formatted_paths']) if parsed_paths_data else 0
            }

            print(f"✅ Successfully processed S2 cell {s2cell_id} with path analysis")
            print('================================')
            print(prompt)
            print('================================')
            break

        # Save batch results
        if output_dir and batch_results:
            batch_file = os.path.join(output_dir, "enhanced_prompts_with_paths.jsonl")
            with open(batch_file, 'a') as f:
                json.dump(batch_results, f)
                f.write('\n')

        results.append(batch_results)

    print(f"✅ Processed {len(results)} batches successfully with path analysis")
    return results


def process_paths_with_enhanced_parsing(paths_data, nodes_gdf, edges_gdf, subgraph,
                                        max_paths_to_parse=10, use_gpu=True):
    """
    Simplified path processing - just get the formatted enhanced paths
    """
    print(f"🔄 Processing paths with enhanced coordinate parsing")

    # Initialize GraphProcessor with the subgraph
    graph_processor = GraphProcessor(use_gpu=use_gpu)
    graph_processor.nx_graph = subgraph

    # Convert to cuGraph if GPU enabled
    if graph_processor.use_gpu:
        graph_processor._convert_to_cugraph()

    print(f"✅ Initialized GraphProcessor with {subgraph.number_of_nodes()} nodes, {subgraph.number_of_edges()} edges")

    # Try to import the enhanced parsing function
    try:
        from generate_routes import (
            parse_single_path_directions_enhanced_with_coordinates,
        )
        enhanced_parsing_available = True
        print("✅ Enhanced parsing functions available")
    except ImportError:
        print("⚠️ Enhanced parsing functions not available, using simple parsing")
        enhanced_parsing_available = False

    # Simple result structure - just store the formatted paths
    parsed_paths = {
        'formatted_paths': [],
        'total_paths': len(paths_data['all_paths']),
        'successful_parses': 0,
        'failed_parses': 0
    }

    # Sort paths by length and limit for performance
    sorted_paths = sorted(paths_data['all_paths'], key=lambda x: x['length'])
    paths_to_parse = sorted_paths[:max_paths_to_parse]

    print(f"📝 Parsing {len(paths_to_parse)} paths")

    for i, path_info in enumerate(paths_to_parse):
        path_nodes = path_info['path']
        source = path_info['source']
        target = path_info['target']

        try:
            if enhanced_parsing_available:
                # Use enhanced parsing
                parsed_result = parse_single_path_directions_enhanced_with_coordinates(
                    path_nodes, nodes_gdf, edges_gdf, graph_processor
                )

                if parsed_result.get('success', False):
                    formatted_path = parsed_result.get('formatted_path_enhanced', '')
                    if formatted_path:
                        parsed_paths['formatted_paths'].append(formatted_path)
                        parsed_paths['successful_parses'] += 1
                        print(f"    ✅ Path {i + 1}: Enhanced parsing successful")
                    else:
                        # Fallback to simple
                        simple_path = create_simple_path_description(source, target, path_nodes, nodes_gdf)
                        parsed_paths['formatted_paths'].append(simple_path)
                        parsed_paths['successful_parses'] += 1
                        print(f"    ⚠️ Path {i + 1}: Used simple fallback")
                else:
                    # Fallback to simple
                    simple_path = create_simple_path_description(source, target, path_nodes, nodes_gdf)
                    parsed_paths['formatted_paths'].append(simple_path)
                    parsed_paths['successful_parses'] += 1
                    print(f"    ⚠️ Path {i + 1}: Enhanced failed, used simple")
            else:
                # Use simple parsing
                simple_path = create_simple_path_description(source, target, path_nodes, nodes_gdf)
                parsed_paths['formatted_paths'].append(simple_path)
                parsed_paths['successful_parses'] += 1
                print(f"    ✅ Path {i + 1}: Simple parsing")

        except Exception as e:
            parsed_paths['failed_parses'] += 1
            print(f"    ❌ Path {i + 1}: Exception - {e}")

    print(
        f"📊 Parsing complete: {parsed_paths['successful_parses']} successful, {parsed_paths['failed_parses']} failed")
    return parsed_paths


def process_image_prompt_data_enhanced_with_paths(jsonl_file, nodes_gdf, edges_gdf, output_dir=None,
                                                  batch_size=1, subgraph_output_dir='', save_subgraphs=True,
                                                  use_gpu=True):
    """
    Enhanced processing with simplified path analysis
    """
    # Read the JSONL file
    data_items = []
    with open(jsonl_file, 'r') as f:
        for line in f:
            if line.strip():
                data_items.append(json.loads(line))

    print(f"Loaded {len(data_items)} items from {jsonl_file}")

    # Create output directories
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if subgraph_output_dir and not os.path.exists(subgraph_output_dir):
        os.makedirs(subgraph_output_dir)

    results = []

    # Process in batches
    for batch_start in tqdm(range(0, len(data_items), batch_size), desc="Processing batches with path analysis"):
        batch_end = min(batch_start + batch_size, len(data_items))
        batch_items = data_items[batch_start:batch_end]

        batch_results = {}

        for s2cell_data in batch_items:
            s2cell_id = s2cell_data['s2cell_id']

            print(f"\n🔄 Processing S2 cell with path analysis: {s2cell_id}")

            # Extract image node IDs
            image_node_ids = extract_image_node_ids(s2cell_data)

            if not image_node_ids:
                print("No image nodes found, skipping...")
                continue

            print(f"Found {len(image_node_ids)} image nodes: {image_node_ids}")

            # Determine subgraph parameters
            max_hops = determine_max_hops(len(image_node_ids))

            # Create enhanced subgraph WITH path analysis
            try:
                subgraph_data = create_spatial_subgraph_enhanced_with_paths(
                    nodes_gdf=nodes_gdf,
                    edges_gdf=edges_gdf,
                    center_node_ids=image_node_ids,
                    max_hops=max_hops
                )

                print(
                    f"Created subgraph with {subgraph_data['num_nodes']} nodes and {subgraph_data['num_edges']} edges")
                # print(f"Path analysis: {subgraph_data.get('path_statistics', {}).get('total_paths', 0)} paths found")

            except Exception as e:
                print(f"Error creating subgraph with paths: {e}")
                continue

            # # Process paths with simplified parsing
            parsed_paths_data = None
            # if subgraph_data.get('paths_data') and subgraph_data.get('paths_data', {}).get('total_paths', 0) > 0:
            #     try:
            #         print("🔄 Processing paths with enhanced coordinate parsing...")
            #         parsed_paths_data = process_paths_with_enhanced_parsing(
            #             subgraph_data['paths_data'],
            #             nodes_gdf,
            #             edges_gdf,
            #             subgraph_data['subgraph'],
            #             max_paths_to_parse=5,  # Reduced for simplicity
            #             use_gpu=use_gpu
            #         )
            #         print(f"✅ Successfully parsed {parsed_paths_data['successful_parses']} paths")
            #
            #     except Exception as e:
            #         print(f"⚠️ Error in path parsing: {e}")
            #         parsed_paths_data = None
            # else:
            #     print("ℹ️ No paths found for parsing")

            # Create enhanced prompt with path information
            try:
                prompt = create_enhanced_caption_prompt_with_paths(
                    s2cell_data,
                    subgraph_data,
                    nodes_gdf,
                    parsed_paths_data
                )
                print("Generated enhanced prompt with path analysis")

            except Exception as e:
                print(f"Error creating enhanced prompt: {e}")
                continue

            # Save subgraph with path data
            subgraph_file = None
            if save_subgraphs and subgraph_output_dir:
                run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{s2cell_id}"
                subgraph_file = os.path.join(subgraph_output_dir, f"{run_id}_with_paths.pkl")

                # Include parsed paths in saved data
                save_data = {
                    'subgraph_data': subgraph_data,
                    # 'parsed_paths_data': parsed_paths_data,
                    'processing_timestamp': datetime.now().isoformat(),
                    's2cell_id': s2cell_id
                }

                try:
                    with open(subgraph_file, 'wb') as f:
                        pickle.dump(save_data, f)
                    print(f"Saved enhanced subgraph with paths to {subgraph_file}")
                except Exception as e:
                    print(f"Error saving subgraph: {e}")
                    subgraph_file = None

            # Collect image information
            images = collect_image_information(s2cell_data)

            # Store enhanced results
            batch_results[s2cell_id] = {
                "prompt": prompt,
                "images": images,
                "subgraph_file": subgraph_file,
                "subgraph_info": {
                    "num_nodes": subgraph_data['num_nodes'],
                    "num_edges": subgraph_data['num_edges'],
                    "center_nodes": subgraph_data['center_nodes'],
                    "paths_found": subgraph_data.get('path_statistics', {}).get('total_paths', 0),
                    "paths_parsed": parsed_paths_data['successful_parses'] if parsed_paths_data else 0
                },
                "path_analysis_included": parsed_paths_data is not None,
                "formatted_paths_count": len(parsed_paths_data['formatted_paths']) if parsed_paths_data else 0
            }

            print(f"✅ Successfully processed S2 cell {s2cell_id} with path analysis")
            print('================================')
            print(prompt)
            print('================================')
            break

        # Save batch results
        if output_dir and batch_results:
            batch_file = os.path.join(output_dir, "enhanced_prompts_with_paths.jsonl")
            with open(batch_file, 'a') as f:
                json.dump(batch_results, f)
                f.write('\n')

        results.append(batch_results)

    print(f"✅ Processed {len(results)} batches successfully with path analysis")
    return results


# Updated main function with distance filtering parameter
def main_enhanced_with_paths_and_captioning(jsonl_file, output_dir, place='newyork', api_key=None, api_provider='qwen', use_gpu=True,
                                            generate_captions=True,
                                            max_image_distance_km=1.0, resume_from_checkpoint=True, clear_checkpoint=False):
    """
    Enhanced main function for caption generation with detailed spatial context, path analysis, and captioning.

    Args:
        jsonl_file: Path to the JSONL file containing S2 cell data
        output_dir: Directory to save output files
        place: Place name for data folder structure (default: 'newyork')
        api_key: API key for caption generation
        api_provider: API provider for caption generation ('qwen' or 'yinli')
        use_gpu: Whether to use GPU acceleration
        generate_captions: Whether to generate captions
        max_image_distance_km: Maximum allowed distance between image nodes in km (default: 1.0)
        resume_from_checkpoint: Whether to resume from checkpoint if available (default: True)
        clear_checkpoint: Whether to clear existing checkpoint and start fresh (default: False)
    """
    # Load nodes and edges data using the same structure as s2cell_image_matching.py
    data_folder_path = os.path.join(DATA_ROOT, 'geo', 'SR', 'osm_data', place)

    print(f"Loading data from: {data_folder_path}")

    # # Try to load from optimized format first
    # edges_pkl_path = f"{data_folder}/edges.pkl"
    # nodes_pkl_path = f"{data_folder}/nodes.pkl"

    # if os.path.exists(edges_pkl_path) and os.path.exists(nodes_pkl_path):
    #     print("Loading from optimized pickle format...")
    #     with open(edges_pkl_path, 'rb') as f:
    #         edges = pickle.load(f)
    #     with open(nodes_pkl_path, 'rb') as f:
    #         nodes = pickle.load(f)
    # else:
    #     print("Loading from GeoJSON format...")
    print("Loading nodes and edges from GeoJSON files...")
    nodes = gpd.read_file(f"{data_folder_path}/nodes_with_districts.geojson")
    nodes['id'] = nodes['id'].astype(int)
    edges = gpd.read_file( f"{data_folder_path}/edges.geojson")
    mapillary_edges = gpd.read_file( f"{data_folder_path}/edges_mapillary.geojson")
    print(f"Mapillary edges types: {mapillary_edges['type'].unique()}")
    print(f"Original edges types: {edges['type'].unique()}")

    # Concatenate edges to include mapillary relationships
    edges = pd.concat([edges, mapillary_edges], ignore_index=True)
    edges['id1'] = edges['id1'].astype(int)
    edges['id2'] = edges['id2'].astype(int)

    # # Save optimized format for future use
    # print("Saving optimized format...")
    # with open(edges_pkl_path, 'wb') as f:
    #     pickle.dump(edges, f)
    # with open(nodes_pkl_path, 'wb') as f:
    #     pickle.dump(nodes, f)

    # Clean up node names - handle both Complex_Crossing and regular Crossing
    # First handle Complex_Crossing
    nodes.loc[nodes['name'].str.contains('Complex_Crossing', na=False), 'name'] = \
        nodes.loc[nodes['name'].str.contains('Complex_Crossing', na=False), 'name'].str.replace(
            'Complex_Crossing_', 'Complex Crossing of ', regex=False
        ).str.replace('_', ' and ', n=1).str.replace('_', ' ', regex=False)
    
    # Then handle regular Crossing_ prefix
    nodes.loc[nodes['name'].str.contains('^Crossing_', na=False, regex=True), 'name'] = \
        nodes.loc[nodes['name'].str.contains('^Crossing_', na=False, regex=True), 'name'].str.replace(
            'Crossing_', 'Intersection of ', regex=False
        ).str.replace('_', ' and ', regex=False)

    # Replace crossing node names with their address field (which should be cleaner)
    nodes['name'] = np.where(
        nodes['type'] == 'crossing',
        nodes['address'],
        nodes['name']
    )
    nodes=nodes[nodes.name.isnull()==False]

    # Get line nodes for type mapping (same as s2cell_image_matching.py)
    lines = nodes[nodes.geometry.geom_type.isin(['LineString', 'MultiLineString'])]

    # Create a mapping of name to the most frequent type
    type_counts = lines.groupby(['name', 'type']).size().reset_index(name='count')
    most_frequent_type = type_counts.loc[type_counts.groupby('name')['count'].idxmax()]
    name_to_type = dict(zip(most_frequent_type['name'], most_frequent_type['type']))

    # Fill in null values in the existing type column with matched types from lines
    nodes['type'] = nodes['type'].fillna(nodes['name'].map(name_to_type))

    # print('nodes columns', nodes.columns)
    print('edges types', edges['type'].unique())

    print(f"Loaded {len(nodes)} nodes and {len(edges)} edges")

    # Initialize nodes dictionary cache for performance optimization
    print("🔧 Initializing nodes dictionary cache...")
    get_nodes_dict(nodes)
    print("✅ Nodes dictionary cache initialized")

    # Set up output directories
    subgraph_output_dir = os.path.join(output_dir, "subgraphs_with_paths")

    # Process data with enhanced functionality including path analysis and individual captioning
    if generate_captions:
        print(f"🤖 Individual caption generation enabled using {api_provider} API (max distance filter: {max_image_distance_km}km)")
        prompt_data, filtering_stats = process_image_prompt_data_enhanced_with_paths_and_captioning(
            jsonl_file=jsonl_file,
            nodes_gdf=nodes,
            edges_gdf=edges,
            output_dir=output_dir,
            batch_size=1,
            subgraph_output_dir=subgraph_output_dir,
            save_subgraphs=True,
            use_gpu=use_gpu,
            generate_captions=True,
            api_key=api_key,
            api_provider=api_provider,
            max_image_distance_km=max_image_distance_km,
            resume_from_checkpoint=resume_from_checkpoint,
            clear_checkpoint=clear_checkpoint
        )
        
    else:
        print(
            f"📝 Caption generation disabled - only generating prompts (max distance filter: {max_image_distance_km}km)")
        prompt_data, filtering_stats = process_image_prompt_data_enhanced_with_paths(
            jsonl_file=jsonl_file,
            nodes_gdf=nodes,
            edges_gdf=edges,
            output_dir=output_dir,
            batch_size=1,
            subgraph_output_dir=subgraph_output_dir,
            save_subgraphs=True,
            use_gpu=use_gpu,
            max_image_distance_km=max_image_distance_km
        )

    print("Enhanced processing with path analysis and individual captioning complete!")
    return prompt_data, filtering_stats


def extract_mapillary_nodes_from_reasoning_paths(reasoning_path_file):
    """
    Extract unique Mapillary nodes and S2 cell IDs from the reasoning path data.
    
    Args:
        reasoning_path_file: Path to the reasoning path JSONL file
        
    Returns:
        dict: Dictionary with S2 cell IDs as keys and Mapillary node info as values
    """
    mapillary_data = {}
    
    if not os.path.exists(reasoning_path_file):
        print(f"❌ Reasoning path file not found: {reasoning_path_file}")
        return mapillary_data
    
    try:
        with open(reasoning_path_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    data = json.loads(line)
                    
                    # Extract key information
                    s2cell_id = data.get('s2cell_id')
                    mapillary_node = data.get('mapillary_node')
                    image_path = data.get('image_path')
                    image_coordinates = data.get('image_coordinates')
                    origin_name = data.get('origin_name')
                    destination_name = data.get('destination_name')
                    
                    if s2cell_id and mapillary_node:
                        if s2cell_id not in mapillary_data:
                            mapillary_data[s2cell_id] = []
                        
                        mapillary_data[s2cell_id].append({
                            'mapillary_node': mapillary_node,
                            'image_path': image_path,
                            'image_coordinates': image_coordinates,
                            'origin_name': origin_name,
                            'destination_name': destination_name,
                            'line_number': line_num
                        })
                        
                except json.JSONDecodeError as e:
                    print(f"⚠️ JSON decode error at line {line_num}: {e}")
                    continue
                except Exception as e:
                    print(f"⚠️ Error processing line {line_num}: {e}")
                    continue
        
        print(f"✅ Successfully extracted data from {len(mapillary_data)} S2 cells")
        total_mapillary_nodes = sum(len(nodes) for nodes in mapillary_data.values())
        print(f"   Total Mapillary nodes found: {total_mapillary_nodes}")
        
        # Print summary for first few S2 cells
        for i, (s2cell_id, nodes) in enumerate(list(mapillary_data.items())[:5]):
            print(f"   S2 Cell {s2cell_id}: {len(nodes)} Mapillary nodes")
            for node in nodes[:3]:  # Show first 3 nodes
                print(f"     - {node['mapillary_node']}: {node['image_path']}")
            if len(nodes) > 3:
                print(f"     ... and {len(nodes) - 3} more")
        
        if len(mapillary_data) > 5:
            print(f"   ... and {len(mapillary_data) - 5} more S2 cells")
            
    except Exception as e:
        print(f"❌ Error reading reasoning path file: {e}")
        traceback.print_exc()
    
    return mapillary_data


def create_filtered_s2cell_data_for_mapillary_nodes(mapillary_data, nodes_gdf, edges_gdf):
    """
    Create filtered S2 cell data structure for processing only the Mapillary nodes.
    
    Args:
        mapillary_data: Dictionary from extract_mapillary_nodes_from_reasoning_paths
        nodes_gdf: GeoDataFrame containing node information
        edges_gdf: GeoDataFrame containing edge information
        
    Returns:
        list: List of S2 cell data structures for processing
    """
    filtered_s2cell_data = []
    
    for s2cell_id, mapillary_nodes in mapillary_data.items():
        print(f"\n🔄 Creating filtered data for S2 cell: {s2cell_id}")
        
        # Create a simplified S2 cell data structure
        s2cell_item = {
            's2cell_id': s2cell_id,
            'valid_images': {},
            'mapillary_nodes': mapillary_nodes
        }
        
        # Process each Mapillary node
        for i, node_info in enumerate(mapillary_nodes):
            mapillary_node_id = node_info['mapillary_node']
            image_path = node_info['image_path']
            
            # Create image entry
            image_key = f"image_{i}"
            s2cell_item['valid_images'][image_key] = {
                'image_path': image_path,
                'mapillary_node_id': mapillary_node_id,
                'coordinates': node_info['image_coordinates'],
                'origin_name': node_info['origin_name'],
                'destination_name': node_info['destination_name']
            }
            
            print(f"   Added image {i+1}: {mapillary_node_id} -> {os.path.basename(image_path)}")
        
        filtered_s2cell_data.append(s2cell_item)
        print(f"   Created data structure with {len(mapillary_nodes)} images")
    
    print(f"\n✅ Created filtered data for {len(filtered_s2cell_data)} S2 cells")
    return filtered_s2cell_data


def process_mapillary_nodes_from_reasoning_paths(reasoning_path_file, nodes_gdf, edges_gdf, output_dir=None,
                                               batch_size=1, subgraph_output_dir='', save_subgraphs=True,
                                               use_gpu=True, generate_captions=True, api_key=None, api_provider='qwen',
                                               resume_from_checkpoint=True, clear_checkpoint=False):
    """
    Main function to process Mapillary nodes from reasoning path data.
    
    Args:
        reasoning_path_file: Path to the reasoning path JSONL file
        nodes_gdf: GeoDataFrame containing node information
        edges_gdf: GeoDataFrame containing edge information
        output_dir: Output directory for results
        batch_size: Number of S2 cells to process per batch
        subgraph_output_dir: Directory to save subgraphs
        save_subgraphs: Whether to save subgraphs
        use_gpu: Whether to use GPU acceleration
        generate_captions: Whether to generate image captions
        api_key: API key for caption generation
        api_provider: API provider for caption generation ('qwen' or 'yinli')
        resume_from_checkpoint: Whether to resume from checkpoint if available (default: True)
        clear_checkpoint: Whether to clear existing checkpoint and start fresh (default: False)
        
    Returns:
        tuple: (results, processing_stats)
    """
    print("🚀 Starting Mapillary node processing from reasoning paths...")
    
    # Step 1: Extract Mapillary nodes from reasoning path data
    print("\n📊 Step 1: Extracting Mapillary nodes from reasoning paths...")
    mapillary_data = extract_mapillary_nodes_from_reasoning_paths(reasoning_path_file)
    
    if not mapillary_data:
        print("❌ No Mapillary data found. Exiting.")
        return [], {}
    
    # Step 2: Create filtered S2 cell data structures
    print("\n📊 Step 2: Creating filtered S2 cell data structures...")
    filtered_s2cell_data = create_filtered_s2cell_data_for_mapillary_nodes(
        mapillary_data, nodes_gdf, edges_gdf
    )
    
    # Step 3: Process the filtered data using existing pipeline
    print("\n📊 Step 3: Processing filtered S2 cells with image generation...")
    results, processing_stats = process_image_prompt_data_enhanced_with_paths_and_captioning(
        jsonl_file=None,  # We're not using a JSONL file, but passing data directly
        nodes_gdf=nodes_gdf,
        edges_gdf=edges_gdf,
        output_dir=output_dir,
        batch_size=batch_size,
        subgraph_output_dir=subgraph_output_dir,
        save_subgraphs=save_subgraphs,
        use_gpu=use_gpu,
        generate_captions=generate_captions,
        api_key=api_key,
        api_provider=api_provider,
        max_image_distance_km=5.0,  # Allow larger distances for Mapillary nodes
        resume_from_checkpoint=resume_from_checkpoint,
        clear_checkpoint=clear_checkpoint,
        custom_s2cell_data=filtered_s2cell_data  # Pass our filtered data
    )
    
    print(f"\n🎉 Mapillary node processing complete!")
    print(f"   Processed {len(filtered_s2cell_data)} S2 cells")
    print(f"   Total Mapillary nodes: {sum(len(nodes) for nodes in mapillary_data.values())}")
    
    return results, processing_stats


def get_existing_captioned_images(existing_captions_file):
    """
    Extract all image paths that already have captions.
    
    Args:
        existing_captions_file: Path to the existing captions JSONL file
        
    Returns:
        set: Set of image paths that already have captions
    """
    captioned_images = set()
    
    if not os.path.exists(existing_captions_file):
        print(f"📄 No existing captions file found at: {existing_captions_file}")
        return captioned_images
    
    try:
        with open(existing_captions_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    data = json.loads(line)
                    
                    # Check for image_paths key (single image)
                    if 'image_paths' in data and data['image_paths']:
                        captioned_images.add(data['image_paths'])
                    
                    # Check for images key in swift_format (list of images)
                    if 'swift_format' in data and 'images' in data['swift_format']:
                        for img_path in data['swift_format']['images']:
                            captioned_images.add(img_path)
                            
                except json.JSONDecodeError as e:
                    print(f"⚠️ JSON decode error at line {line_num}: {e}")
                    continue
                except Exception as e:
                    print(f"⚠️ Error processing line {line_num}: {e}")
                    continue
        
        print(f"✅ Found {len(captioned_images)} images with existing captions")
        
    except Exception as e:
        print(f"❌ Error reading existing captions file: {e}")
        traceback.print_exc()
    
    return captioned_images


def extract_images_from_training_data(training_data_file):
    """
    Extract all unique image paths from the training data file.
    
    Args:
        training_data_file: Path to the training data JSONL file
        
    Returns:
        list: List of unique image paths
    """
    image_paths = set()
    
    if not os.path.exists(training_data_file):
        print(f"❌ Training data file not found: {training_data_file}")
        return []
    
    try:
        with open(training_data_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    data = json.loads(line)
                    
                    # Extract image_path if it exists
                    if 'image_path' in data and data['image_path']:
                        image_paths.add(data['image_path'])
                    
                    # Also check for images in messages if exists
                    if 'images' in data:
                        if isinstance(data['images'], list):
                            for img_path in data['images']:
                                image_paths.add(img_path)
                        elif isinstance(data['images'], str):
                            image_paths.add(data['images'])
                            
                except json.JSONDecodeError as e:
                    print(f"⚠️ JSON decode error at line {line_num}: {e}")
                    continue
                except Exception as e:
                    print(f"⚠️ Error processing line {line_num}: {e}")
                    continue
        
        print(f"✅ Found {len(image_paths)} unique images in training data")
        
    except Exception as e:
        print(f"❌ Error reading training data file: {e}")
        traceback.print_exc()
        return []
    
    return list(image_paths)


def generate_captions_for_training_data(training_data_file, existing_captions_file, output_file, 
                                       api_key=None, api_provider='qwen', delay_between_calls=3):
    """
    Generate captions for images in training data that don't have captions yet.
    
    Args:
        training_data_file: Path to the training data JSONL file
        existing_captions_file: Path to the existing captions JSONL file
        output_file: Path to save new captions
        api_key: API key for caption generation
        api_provider: API provider ('qwen' or 'yinli')
        delay_between_calls: Delay between API calls in seconds
        
    Returns:
        dict: Statistics about caption generation
    """
    print("🚀 Starting caption generation for training data...")
    
    # Step 1: Get existing captioned images
    print("\n📊 Step 1: Loading existing captions...")
    captioned_images = get_existing_captioned_images(existing_captions_file)
    
    # Step 2: Extract images from training data
    print("\n📊 Step 2: Extracting images from training data...")
    all_training_images = extract_images_from_training_data(training_data_file)
    
    if not all_training_images:
        print("❌ No images found in training data. Exiting.")
        return {'total_images': 0, 'already_captioned': 0, 'new_captions': 0, 'failed': 0}
    
    # Step 3: Filter out already captioned images
    print("\n📊 Step 3: Filtering images...")
    images_to_caption = [img for img in all_training_images if img not in captioned_images]
    
    stats = {
        'total_images': len(all_training_images),
        'already_captioned': len(captioned_images),
        'to_caption': len(images_to_caption),
        'new_captions': 0,
        'failed': 0
    }
    
    print(f"\n📈 Statistics:")
    print(f"   Total images in training data: {stats['total_images']}")
    print(f"   Already have captions: {stats['already_captioned']}")
    print(f"   Need new captions: {stats['to_caption']}")
    
    if not images_to_caption:
        print("\n✅ All images already have captions!")
        return stats
    
    # Step 4: Generate captions for new images
    print(f"\n📊 Step 4: Generating captions for {len(images_to_caption)} images...")
    
    # Create simple prompt for individual images
    prompt = """You are an advanced vision model tasked with generating detailed captions for urban street images.

Please provide a detailed caption for this street image, including:
1. **Detailed image caption**: Describe the unique visual features, including:
   - Locations and landmarks
   - Architectural features
   - Businesses, signage, or notable features
   - Vegetation and natural elements
   - Street characteristics
   - Visual environment and atmosphere

2. **Summarization**: Provide a concise summary of the key features and overall character of the location.

Format your response as:
**Image:** [detailed description]

**Summarization:** [concise summary]"""
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Process each image
    for i, image_path in enumerate(images_to_caption):
        print(f"\n📸 Processing image {i+1}/{len(images_to_caption)}: {os.path.basename(image_path)}")
        
        try:
            # Generate caption
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
                
                # Create result structure
                result = {
                    "image_paths": valid_path,
                    "individual_captions": caption,
                    "image_caption": image_caption,
                    "summarization": summarization_text,
                    "swift_format": {
                        "messages": [
                            {
                                "role": "user",
                                "content": f"<image>{image_caption} <summarization>{summarization_text}"
                            },
                            {
                                "role": "assistant",
                                "content": summarization_text
                            }
                        ],
                        "images": [valid_path],
                        "label": 1.0
                    },
                    "source": "training_data_caption_generation",
                    "timestamp": datetime.now().isoformat()
                }
                
                # Append to output file
                with open(output_file, 'a', encoding='utf-8') as f:
                    json.dump(result, f)
                    f.write('\n')
                
                stats['new_captions'] += 1
                print(f"   ✅ Caption generated and saved")
                print(f"   Caption length: {len(caption)} characters")
                
            else:
                print(f"   ⚠️ Failed to generate caption")
                stats['failed'] += 1
                
        except Exception as e:
            print(f"   ❌ Error generating caption: {e}")
            traceback.print_exc()
            stats['failed'] += 1
        
        # Add delay between calls
        if i < len(images_to_caption) - 1:
            print(f"   ⏳ Waiting {delay_between_calls} seconds before next call...")
            time.sleep(delay_between_calls)
    
    # Print final statistics
    print(f"\n🎉 Caption generation complete!")
    print(f"📊 Final Statistics:")
    print(f"   Total images processed: {stats['to_caption']}")
    print(f"   Successfully generated: {stats['new_captions']}")
    print(f"   Failed: {stats['failed']}")
    print(f"   Success rate: {stats['new_captions']/stats['to_caption']*100:.1f}%")
    
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process image captioning with spatial context')
    parser.add_argument('--place', type=str, default='singapore', help='Place name for data folder structure')
    parser.add_argument('--jsonl_file', type=str, default=None,
                        help='Path to JSONL file (auto-generated if not provided)')
    parser.add_argument('--reasoning_path_file', type=str, default='lanyun-fs/UrbanKG/data/geo/SR/osm_data/singapore/reasoning_path_mapillary_swift_qa.jsonl',
                        help='Path to reasoning path JSONL file with Mapillary nodes')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (auto-generated if not provided)')
    parser.add_argument('--api_key', type=str, default=None, help='API key for caption generation')
    parser.add_argument('--api_provider', type=str, default='qwen', choices=['qwen', 'yinli'], 
                        help='API provider for caption generation: qwen (Qwen VL) or yinli (yinli.one)')
    parser.add_argument('--use_gpu', action='store_true', default=True, help='Use GPU acceleration')
    parser.add_argument('--generate_captions', action='store_true', default=True, help='Generate captions')
    parser.add_argument('--max_image_distance_km', type=float, default=1.0,
                        help='Maximum distance between image nodes in km')
    parser.add_argument('--process_mapillary', action='store_true', default=False,
                        help='Process Mapillary nodes from reasoning paths instead of standard JSONL')
    parser.add_argument('--resume', action='store_true', default=True,
                        help='Resume processing from checkpoint if available (default: True)')
    parser.add_argument('--no-resume', action='store_true', default=False,
                        help='Disable resume functionality and start fresh')
    parser.add_argument('--clear-checkpoint', action='store_true', default=True,
                        help='Clear existing checkpoint and start fresh')
    parser.add_argument('--process_training_data', action='store_true', default=False,
                        help='Process images from training data file')
    parser.add_argument('--training_data_file', type=str, 
                        default=None,
                        help='Path to training data JSONL file (default: auto-constructed from DATA_ROOT)')
    parser.add_argument('--existing_captions_file', type=str,
                        default=None,
                        help='Path to existing captions JSONL file (default: auto-constructed from OUTPUT_ROOT)')
    parser.add_argument('--training_output_file', type=str,
                        default='./mydata/training_data_captions.jsonl',
                        help='Output file for training data captions')

    args = parser.parse_args()

    # Handle resume arguments
    resume_from_checkpoint = args.resume and not args.no_resume
    clear_checkpoint = args.clear_checkpoint
    
    if args.no_resume:
        print("🚫 Resume functionality disabled - starting fresh")
    elif args.clear_checkpoint:
        print("🗑️ Clear checkpoint requested - starting fresh")
    elif resume_from_checkpoint:
        print("🔄 Resume functionality enabled - will resume from checkpoint if available")

    # Auto-generate paths if not provided
    if args.output_dir is None:
        if args.process_mapillary:
            args.output_dir = f"./mydata/mapillary_enhanced_data_with_paths_and_captions_{args.place}"
        else:
            args.output_dir = f"./mydata/enhanced_image_data_with_paths_and_captions_{args.place}"

    if args.api_key is None:
        if args.api_provider == 'qwen':
            args.api_key = os.getenv("DASHSCOPE_API_KEY")
        elif args.api_provider == 'yinli':
            args.api_key = os.getenv("NEWAPI_API_KEY")

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Check if processing training data
    if args.process_training_data:
        print(f"🚀 Processing training data images...")
        print(f"Training data file: {args.training_data_file}")
        print(f"Existing captions file: {args.existing_captions_file}")
        print(f"Output file: {args.training_output_file}")
        print(f"API provider: {args.api_provider}")
        
        # Generate captions for training data
        stats = generate_captions_for_training_data(
            training_data_file=args.training_data_file,
            existing_captions_file=args.existing_captions_file,
            output_file=args.training_output_file,
            api_key=args.api_key,
            api_provider=args.api_provider,
            delay_between_calls=3
        )
        
        print(f"\n🎉 Training data processing completed!")
        print(f"📁 Results saved to: {args.training_output_file}")
        print(f"📊 Statistics: {stats}")
        exit(0)
        
    elif args.process_mapillary:
        # Process Mapillary nodes from reasoning paths
        if not args.reasoning_path_file:
            print("❌ --reasoning_path_file is required when using --process_mapillary")
            exit(1)
            
        print(f"🚀 Processing Mapillary nodes from reasoning paths...")
        print(f"Reasoning path file: {args.reasoning_path_file}")
        print(f"Output directory: {args.output_dir}")
        
        # Load GeoDataFrames (assuming they exist in the place folder)
        data_folder = os.path.join(DATA_ROOT, 'geo', 'SR', 'osm_data', args.place)
        NODES_PATH = f'{data_folder}/nodes_new.geojson'
        EDGES_PATH = f'{data_folder}/edges_new.parquet'
        MAPILLARY_NODES_PATH = f'{data_folder}/mapillary_nodes.geojson'
        MAPILLARY_EDGES_PATH = f'{data_folder}/mapillary_edges.geojson'
        
        print(f"Loading GeoDataFrames from {data_folder}...")
        nodes_gdf = gpd.read_file(NODES_PATH)
        mapillary_nodes_gdf = gpd.read_file(MAPILLARY_NODES_PATH)
        nodes_gdf['id'] = nodes_gdf['id'].astype(int)
        mapillary_nodes_gdf['id'] = mapillary_nodes_gdf['id'].astype(int)
        nodes_gdf = pd.concat([nodes_gdf, mapillary_nodes_gdf], ignore_index=True)
        edges_gdf = pd.read_parquet(EDGES_PATH)
        mapillary_edges_gdf = pd.read_parquet(MAPILLARY_EDGES_PATH)
        edges_gdf = pd.concat([edges_gdf, mapillary_edges_gdf], ignore_index=True)
        edges_gdf['id1'] = edges_gdf['id1'].astype(int)
        edges_gdf['id2'] = edges_gdf['id2'].astype(int)
        print(f"Loaded {len(nodes_gdf)} nodes and {len(edges_gdf)} edges")
        
        # Initialize nodes dictionary cache for performance optimization
        print("🔧 Initializing nodes dictionary cache...")
        get_nodes_dict(nodes_gdf)
        print("✅ Nodes dictionary cache initialized")
        
        # Process Mapillary nodes
        results, _ = process_mapillary_nodes_from_reasoning_paths(
            reasoning_path_file=args.reasoning_path_file,
            nodes_gdf=nodes_gdf,
            edges_gdf=edges_gdf,
            output_dir=args.output_dir,
            batch_size=1,
            subgraph_output_dir=os.path.join(args.output_dir, "subgraphs"),
            save_subgraphs=True,
            use_gpu=args.use_gpu,
            generate_captions=args.generate_captions,
            api_key=args.api_key,
            api_provider=args.api_provider,
            resume_from_checkpoint=resume_from_checkpoint,
            clear_checkpoint=clear_checkpoint
        )
    else:
        # Standard processing
        if args.jsonl_file is None:
            data_folder = os.path.join(DATA_ROOT, 'geo', 'SR', 'osm_data', args.place)
            args.jsonl_file = os.path.join(data_folder, 'filtered_image_prompt_data2.jsonl')

        print(f"Processing place: {args.place}")
        print(f"Input JSONL file: {args.jsonl_file}")
        print(f"Output directory: {args.output_dir}")

        # Run with caption generation
        results, _ = main_enhanced_with_paths_and_captioning(
            jsonl_file=args.jsonl_file,
            output_dir=args.output_dir,
            place=args.place,
            api_key=args.api_key,
            api_provider=args.api_provider,
            use_gpu=args.use_gpu,
            generate_captions=args.generate_captions,
            max_image_distance_km=args.max_image_distance_km,
            resume_from_checkpoint=resume_from_checkpoint,
            clear_checkpoint=clear_checkpoint
        )

    print(f"\n🎉 Processing completed!")
    print(f"📁 Results saved to: {args.output_dir}")
    if args.generate_captions:
        print(f"📝 Individual captions saved to: {args.output_dir}/generated_captions/")
    if args.process_mapillary:
        print(f"🗺️ Mapillary node processing completed")
        print(f"📊 Processed {len(results)} S2 cells with Mapillary nodes")
    else:
        print(f"📊 Processed {len(results)} batches total")
    
    # Clear nodes dictionary cache to free memory
    clear_nodes_dict_cache()
