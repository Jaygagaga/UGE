# Preserve original complex logic functions with GPU optimization where possible
import pandas as pd
import requests
from shapely.geometry import Point, LineString, Polygon, MultiPolygon, MultiLineString
from shapely.ops import nearest_points
import geopandas as gpd
import numpy as np
import networkx as nx
import random
import json
import os
import pickle
from datetime import datetime
from geopy.distance import geodesic
from itertools import combinations
import math
import re

# GPU acceleration imports
# try:
#     import cudf
#     import cugraph
#
#     CUGRAPH_AVAILABLE = True
#     print("✅ cuGraph available for GPU acceleration")
# except ImportError:
CUGRAPH_AVAILABLE = False
#     print("❌ cuGraph not available. Install with: conda install -c rapidsai cugraph")

# For HERE polyline decoding
try:
    from flexpolyline import decode as decode_flex_polyline
except ImportError:
    import subprocess

    subprocess.call(["pip", "install", "flexpolyline"])
    from flexpolyline import decode as decode_flex_polyline


def get_node_name_with_fallbacks(node_id, node_info):
    """
    Get node name with multiple fallback strategies:
    1. Try to get name from node_info
    2. Try to get address or street from node_info
    3. Use node ID as last resort
    """
    # Strategy 1: Try to get name from node_info
    if 'name' in node_info and pd.notna(node_info['name']) and node_info['name']:
        name = node_info['name']
        if pd.notna(name) and str(name).strip():
            return str(name).strip()
    if node_info['type'] != 'mapillary':
        # Strategy 2: Try address as fallback
        if 'address' in node_info and pd.notna(node_info['address']) and node_info['address']:
            address = node_info['address']
            if pd.notna(address) and str(address).strip():
                return str(address).strip()

        # Strategy 3: Try street as fallback
        if 'street' in node_info and pd.notna(node_info['street']) and node_info['street']:
            street = node_info['street']
            if pd.notna(street) and str(street).strip():
                return str(street).strip()

        # Strategy 4: Try other fields
        other_fields = ['type', 'category']
        for field in other_fields:
            if field in node_info and pd.notna(node_info[field]) and node_info[field]:
                value = node_info[field]
                if pd.notna(value) and str(value).strip():
                    return str(value).strip()

    # Last resort - use node ID
    return str(node_id)


def get_node_display_name_original(node_id, node_info):
    """
    Enhanced node display name logic with robust fallbacks
    """
    return get_node_name_with_fallbacks(node_id, node_info)


def calculate_bearing_between_points_original(point1, point2):
    """
    Original calculate_bearing_between_points logic (preserved)
    """
    return calculate_bearing_original(point1.x, point1.y, point2.x, point2.y)


def calculate_bearing_original(x1, y1, x2, y2):
    """
    Calculate bearing using spherical trigonometry (matching spatial_encoders.py).
    
    Uses the standard spherical trigonometry formula for accurate geographic bearings:
    y = sin(Δlon) * cos(lat2)
    x = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(Δlon)
    bearing = atan2(y, x)
    
    Args:
        x1, y1: Source coordinates (x=longitude, y=latitude) in degrees
        x2, y2: Target coordinates (x=longitude, y=latitude) in degrees
    
    Returns:
        bearing: Bearing in degrees (0-360), where 0° is North, 90° is East, etc.
    """
    lon1, lat1 = x1, y1
    lon2, lat2 = x2, y2
    
    # Check if points are identical (or very close)
    if abs(lon2 - lon1) < 1e-10 and abs(lat2 - lat1) < 1e-10:
        print(f"   - Warning: Points are identical or very close")
        print(f"   - Point1: ({lon1}, {lat1})")
        print(f"   - Point2: ({lon2}, {lat2})")
        print(f"   - Returning default bearing of 0°")
        return 0.0
    
    # Convert degrees to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    
    # Calculate difference in longitude
    dlon = lon2_rad - lon1_rad
    
    # Standard bearing formula (spherical trigonometry)
    y = math.sin(dlon) * math.cos(lat2_rad)
    x = math.cos(lat1_rad) * math.sin(lat2_rad) - \
        math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon)
    
    # Calculate bearing in radians using atan2
    bearing_rad = math.atan2(y, x)
    
    # Convert to degrees
    bearing_deg = math.degrees(bearing_rad)
    
    # Normalize to 0-360 degrees (atan2 returns -180 to +180)
    if bearing_deg < 0:
        bearing_deg += 360
    
    return bearing_deg


def convert_bearing_to_direction_original(bearing):
    """
    Original convert_bearing_to_direction logic (preserved)
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


def convert_bearing_to_cardinal_direction_original(bearing):
    """
    Original convert_bearing_to_cardinal_direction logic (preserved)
    """
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

    return best_direction


def determine_middle_component_original(source_info, target_info, edge_info, nodes_gdf):
    """
    Enhanced determine_middle_component logic with AOI-Line relationship handling
    """
    edge_type = edge_info.get('type', 'near')

    # Debug: Print edge type and info
    # print(f"🔍 determine_middle_component_original: edge_type='{edge_type}'")
    # print(f"   source_info: {source_info.get('id', 'unknown')}")
    # print(f"   target_info: {target_info.get('id', 'unknown')}")
    # print(f"   edge_info keys: {list(edge_info.keys())}")

    # Edge type--On the same street, crossings
    if edge_type == 'on_same_street' and 'common_street_id' in edge_info and pd.notna(edge_info['common_street_id']):
        street_name = nodes_gdf[nodes_gdf['id'] == edge_info['common_street_id']].name.iloc[0]
        return "on_same_street", street_name, edge_info['common_street_id']

    # Edge type--crossing
    elif edge_type == 'crossing' and pd.notna(edge_info.get('crossing_id')):
        crossing_name = nodes_gdf[nodes_gdf['id'] == edge_info['crossing_id']].name.iloc[0]
        return "crossing", crossing_name, edge_info['crossing_id']

    # Edge type--nearest with enhanced relationship determination
    elif edge_type == 'nearest':
        relationship = determine_nearest_relationship_enhanced(source_info, target_info, edge_info)
        # print(f"   🔍 Relationship: {relationship}")
        return relationship, relationship, None

    # Edge type--boundary with enhanced AOI-Line handling
    elif 'boundary' in edge_type:
        # Analyze the relationship to determine proper middle component
        aoi_info, street_info, relationship_type = identify_aoi_and_street_nodes_enhanced(source_info, target_info)

        if relationship_type == 'aoi_to_line':
            # Source is AOI, Target is Line: use "bounds"
            # Results in: "(South St, bounds, Battery Park)[0.0m, SE]"
            return "bounds", "bounds", None
        elif relationship_type == 'line_to_aoi':
            # Source is Line, Target is AOI: use "near"
            # Results in: "(Battery Park, near, South St)[0.0m, NW]"
            return "near", "near", None
        else:
            # Fallback to original boundary logic
            return "bounds", "bounds", None

    # # Edge type--mapillary (special case for Mapillary nodes)
    # elif edge_type == 'mapillary':
    #     # For mapillary edges, use the target node ID as the middle component
    #     # This ensures we get the actual Mapillary ID instead of the string "mapillary"
    #     target_id =  target_info.get('id')
    #     if target_id:
    #         print(f"   🖼️ Mapillary edge detected, using target_id: {target_id}")
    #         return "near", "near", target_id
    #     else:
    #         print(f"   ⚠️ Mapillary edge but no target_id found")
    #         return "near", "near", None

    # Edge type--near (special case for near edges, often to Mapillary nodes)
    elif edge_type == 'near':
        # For near edges, use "near" as the middle component
        # This is typically used for edges to POIs, Mapillary nodes, etc.
        # print(f"   📍 Near edge detected, using 'near' as middle component")
        return "near", "near", None

    # Special case: detect Mapillary nodes by their ID pattern (long numeric strings)
    elif (isinstance(target_info.get('id'), str) and
          target_info.get('id', '').isdigit() and
          len(str(target_info.get('id', ''))) > 10):
        # This looks like a Mapillary node ID (very long numeric string)
        # print(f"   🖼️ Mapillary node detected by ID pattern: {target_info.get('id')}")
        return "near", "near", None

    else:
        # print(f"   ⚠️ Unhandled edge type '{edge_type}', returning as-is")
        return edge_type, edge_type, None


# Additional helper function to get AOI centroid coordinates for the previous endpoint logic
def get_aoi_centroid_coordinates(aoi_info):
    """
    Get AOI centroid coordinates for boundary/intersect edge endpoints
    """
    try:
        aoi_geometry = aoi_info.get('geometry')
        if aoi_geometry:
            if hasattr(aoi_geometry, 'centroid'):
                centroid = aoi_geometry.centroid
                return (centroid.x, centroid.y)
            elif hasattr(aoi_geometry, 'x') and hasattr(aoi_geometry, 'y'):
                # If it's already a point
                return (aoi_geometry.x, aoi_geometry.y)

        # Fallback: try direct coordinate fields
        aoi_x = aoi_info.get('x')
        aoi_y = aoi_info.get('y')
        if aoi_x is not None and aoi_y is not None:
            return (aoi_x, aoi_y)

    except Exception as e:
        print(f"Error getting AOI centroid coordinates: {e}")

    return None


def get_direction_and_length_enhanced_optimized(source_info, target_info, edge_info, nodes_gdf, position, path,
                                                edges_gdf=None):
    """
    Enhanced get_direction_and_length with logic for handling previous edge types
    """
    edge_type = edge_info.get('type', '')
    source_type = source_info.get('type', '')
    target_type = target_info.get('type', '')
    # print('------------edge_type------------', edge_type)
    direction = None
    length = 0

    # Check if previous edge was nearest, boundary, or intersect
    previous_endpoint_coords = None
    if position > 0:  # Not the first edge in path
        prev_source_id = path[position - 1]
        prev_target_id = path[position]

        # Get previous edge info
        prev_edge_info = get_previous_edge_info(prev_source_id, prev_target_id, edges_gdf)
        if prev_edge_info:
            prev_edge_type = prev_edge_info.get('type', '')
            # print(f'Previous edge type: {prev_edge_type}')

            # If previous edge was nearest, boundary, or intersect, get its endpoint coordinates
            if (prev_edge_type == 'nearest' or
                    'boundary' in prev_edge_type or
                    'intersect' in prev_edge_type):
                previous_endpoint_coords = get_previous_endpoint_coordinates(
                    prev_source_id, prev_target_id, prev_edge_info, nodes_gdf, prev_edge_type
                )
                # print(f'Previous endpoint coordinates: {previous_endpoint_coords}')

    # Main direction calculation logic
    if edge_type == 'on_same_street':
        # Check if we need to use previous endpoint coordinates
        if previous_endpoint_coords:
            direction, length = calculate_direction_from_previous_endpoint(
                previous_endpoint_coords, target_info, edge_info
            )
        else:
            direction = calculate_same_street_direction_optimized(source_info, target_info, edge_info, nodes_gdf)
            length = edge_info.get('crossing_distance_meters', 0)

    elif 'crossing' in edge_type and edge_type != "boundary_crossing":
        # Check if we need to use previous endpoint coordinates
        if previous_endpoint_coords:
            direction, length = calculate_crossing_direction_from_previous_endpoint(
                previous_endpoint_coords, target_info, edge_info, nodes_gdf
            )
        else:
            direction, length = calculate_crossing_direction_optimized(
                source_info, target_info, nodes_gdf, edge_info, position, path, edges_gdf=edges_gdf
            )

    elif edge_type == 'nearest' or edge_type == 'near' or source_type == "mapillary" or target_type == "mapillary":
        # print(f"🔍 Processing nearest edge with enhanced logic...")

        # Use enhanced direction calculation
        direction = calculate_nearest_direction_enhanced(source_info, target_info, edge_info)

        # Use enhanced distance calculation
        length = calculate_nearest_distance_enhanced(source_info, target_info, edge_info)

        # print(f"🔍 Enhanced nearest edge result: direction={direction}, length={length}m")

        # # Additional check: if this is an edge to a Mapillary node, ensure we have distance/direction
        # target_id = target_info.get('id')
        # if (isinstance(target_id, str) and target_id.isdigit() and len(target_id) > 10):
        #     # This is likely a Mapillary node, ensure we have valid distance/direction
        #     if not length or not direction:
        #         print(f"   🖼️ Mapillary node detected but missing distance/direction")
        #         print(f"   Attempting fallback calculation...")

        #         # Try to get from edge_info as fallback
        #         fallback_length = edge_info.get('length', edge_info.get('distance', 0))
        #         fallback_direction = edge_info.get('direction', 'Unknown')

        #         if fallback_length and fallback_direction != 'Unknown':
        #             length = fallback_length
        #             direction = fallback_direction
        #             print(f"   ✅ Used fallback: length={length}, direction={direction}")
        #         else:
        #             print(f"   ⚠️ No fallback available for Mapillary node")

    elif 'boundary' in edge_type or 'intersect' in edge_type:
        direction, length = get_direction_for_boundary_edge_original(source_info, target_info, edge_info, nodes_gdf)

    else:
        return None, None

    return direction, length


def get_previous_edge_info(prev_source_id, prev_target_id, edges_gdf):
    """
    Get edge information for the previous edge in the path
    """
    if edges_gdf is None:
        return None

    try:
        # Find edge between previous source and target
        edge_rows = edges_gdf[
            ((edges_gdf['id1'] == prev_source_id) & (edges_gdf['id2'] == prev_target_id)) |
            ((edges_gdf['id1'] == prev_target_id) & (edges_gdf['id2'] == prev_source_id))
            ]

        if len(edge_rows) > 0:
            return edge_rows.iloc[0].to_dict()

    except Exception as e:
        print(f"Error getting previous edge info: {e}")

    return None


def get_previous_endpoint_coordinates(prev_source_id, prev_target_id, prev_edge_info, nodes_gdf, prev_edge_type):
    """
    Get the endpoint coordinates from the previous edge based on its type
    """
    try:
        if prev_edge_type == 'nearest':
            # For nearest edges, get nearest_x and nearest_y from edge_info
            nearest_x = prev_edge_info.get('nearest_x')
            nearest_y = prev_edge_info.get('nearest_y')

            if nearest_x is not None and nearest_y is not None:
                # print(f"Using nearest coordinates from edge_info: ({nearest_x}, {nearest_y})")
                return (nearest_x, nearest_y)
            else:
                # print("nearest_x/nearest_y not found in edge_info, trying alternative fields...")
                # Fallback: try other coordinate fields that might contain nearest point info
                for x_field, y_field in [('x', 'y'), ('lon', 'lat'), ('longitude', 'latitude')]:
                    x_val = prev_edge_info.get(x_field)
                    y_val = prev_edge_info.get(y_field)
                    if x_val is not None and y_val is not None:
                        print(f"Using fallback coordinates: {x_field}={x_val}, {y_field}={y_val}")
                        return (x_val, y_val)

        elif 'boundary' in prev_edge_type or 'intersect' in prev_edge_type:
            # For boundary/intersect edges, get AOI's centroid coordinates
            source_info = nodes_gdf[nodes_gdf['id'] == prev_source_id].iloc[0].to_dict()
            target_info = nodes_gdf[nodes_gdf['id'] == prev_target_id].iloc[0].to_dict()

            # Identify which node is the AOI
            aoi_info, street_info = identify_aoi_and_street_nodes_original(source_info, target_info)

            if aoi_info:
                aoi_geometry = aoi_info.get('geometry')
                if aoi_geometry:
                    try:
                        # Get AOI's centroid coordinates
                        if hasattr(aoi_geometry, 'centroid'):
                            centroid = aoi_geometry.centroid
                            aoi_x, aoi_y = centroid.x, centroid.y
                            # print(f"Using AOI centroid coordinates: ({aoi_x}, {aoi_y})")
                            return (aoi_x, aoi_y)
                        elif hasattr(aoi_geometry, 'x') and hasattr(aoi_geometry, 'y'):
                            # If it's already a point
                            aoi_x, aoi_y = aoi_geometry.x, aoi_geometry.y
                            # print(f"Using AOI point coordinates: ({aoi_x}, {aoi_y})")
                            return (aoi_x, aoi_y)
                    except Exception as geom_error:
                        print(f"Error extracting AOI geometry coordinates: {geom_error}")

                # Fallback: try direct coordinate fields from AOI info
                aoi_x = aoi_info.get('x')
                aoi_y = aoi_info.get('y')
                if aoi_x is not None and aoi_y is not None:
                    # print(f"Using AOI direct coordinates: ({aoi_x}, {aoi_y})")
                    return (aoi_x, aoi_y)

            # print("Could not identify AOI or extract its coordinates")

    except Exception as e:
        print(f"Error getting previous endpoint coordinates: {e}")

    return None


def calculate_direction_from_previous_endpoint(previous_coords, target_info, edge_info):
    """
    Calculate direction from previous endpoint coordinates to current target
    """
    try:
        prev_x, prev_y = previous_coords

        # Get current target coordinates
        target_x = get_coordinate_from_node_original(target_info, 'x')
        target_y = get_coordinate_from_node_original(target_info, 'y')

        if all(coord is not None for coord in [prev_x, prev_y, target_x, target_y]):
            bearing = calculate_bearing_original(prev_x, prev_y, target_x, target_y)
            direction = convert_bearing_to_direction_original(bearing)

            # Calculate distance
            distance = geodesic((prev_y, prev_x), (target_y, target_x)).meters

            # print(f"📐 Direction from previous endpoint:")
            # print(f"   From: ({prev_x}, {prev_y}) [Previous endpoint]")
            # print(f"   To:   ({target_x}, {target_y}) [Current target]")
            # print(f"   Bearing: {bearing:.1f}°")
            # print(f"   Direction: {direction}")
            # print(f"   Distance: {distance:.1f}m")

            return direction, distance

    except Exception as e:
        print(f"Error calculating direction from previous endpoint: {e}")

    return None, 0


def calculate_crossing_direction_from_previous_endpoint(previous_coords, target_info, edge_info, nodes_gdf):
    """
    Calculate crossing direction from previous endpoint coordinates to crossing point
    """
    try:
        prev_x, prev_y = previous_coords

        # For crossing edge, get the crossing point coordinates
        crossing_id = edge_info.get('crossing_id')
        if crossing_id:
            crossing_coords = get_crossing_coordinates_original(None, edge_info, nodes_gdf)
            if crossing_coords:
                crossing_x, crossing_y = crossing_coords

                bearing = calculate_bearing_original(prev_x, prev_y, crossing_x, crossing_y)
                direction = convert_bearing_to_direction_original(bearing)
                distance = geodesic((prev_y, prev_x), (crossing_y, crossing_x)).meters

                # print(f"📐 Crossing direction from previous endpoint:")
                # print(f"   From: ({prev_x}, {prev_y}) [Previous endpoint]")
                # print(f"   To:   ({crossing_x}, {crossing_y}) [Crossing point]")
                # print(f"   Bearing: {bearing:.1f}°")
                # print(f"   Direction: {direction}")
                # print(f"   Distance: {distance:.1f}m")

                return direction, distance

        # Fallback to target coordinates if crossing coordinates not available
        return calculate_direction_from_previous_endpoint(previous_coords, target_info, edge_info)

    except Exception as e:
        print(f"Error calculating crossing direction from previous endpoint: {e}")

    return None, 0


def calculate_same_street_direction_optimized(source_info, target_info, edge_info, nodes_gdf):
    """
    Original calculate_same_street_direction logic (preserved)
    """
    source_id = source_info.get('id')
    target_id = target_info.get('id')
    # common_street_id = edge_info.get('common_street_id')
    #
    # print(f"🛣️  Same street direction analysis: {source_id} -> {target_id}")
    # print(f"Common street ID: {common_street_id}")

    # Method 1: Direct coordinate calculation (most reliable)
    coordinate_direction = calculate_direction_from_coordinates_original(source_info, target_info, edge_info)
    if coordinate_direction:
        print(f"✅ Coordinate-based direction: {coordinate_direction}")
    return f"{coordinate_direction}"


def identify_aoi_and_street_nodes_original(source_info, target_info):
    """
    Original identify_aoi_and_street_nodes logic (preserved)
    """
    # Check node types and geometries to identify AOI vs street
    for node_info in [source_info, target_info]:
        geometry = node_info.get('geometry')

        # Geometry-based detection
        if isinstance(geometry, (Polygon, MultiPolygon)):
            aoi_info = node_info
            street_info = target_info if node_info == source_info else source_info
            return aoi_info, street_info

    # If can't identify clearly, use source as AOI, target as street
    return source_info, target_info


def calculate_direction_from_coordinates_original(source_info, target_info, edge_info):
    """
    Original calculate_direction_from_coordinates logic (preserved exactly)
    """
    # print(f"🧭 Calculating direction from coordinates...")

    # Priority order for coordinates
    coordinate_sources = [
        # From edge coordinates
        ('crossing1_x', 'crossing1_y', 'crossing2_x', 'crossing2_y', edge_info),
        # From node coordinates
        ('x', 'y', 'x', 'y', None)  # Will use source and target info
    ]

    for x1_field, y1_field, x2_field, y2_field, data_source in coordinate_sources:
        if data_source is not None:
            # Use edge coordinates
            x1 = data_source.get(x1_field)
            y1 = data_source.get(y1_field)
            x2 = data_source.get(x2_field)
            y2 = data_source.get(y2_field)

            # print(f"📍 Trying edge coordinates: {x1_field}, {y1_field} -> {x2_field}, {y2_field}")
        else:
            # Use node coordinates
            x1 = get_coordinate_from_node_original(source_info, 'x')
            y1 = get_coordinate_from_node_original(source_info, 'y')
            x2 = get_coordinate_from_node_original(target_info, 'x')
            y2 = get_coordinate_from_node_original(target_info, 'y')

            # print(f"📍 Trying node coordinates: source({x1}, {y1}) -> target({x2}, {y2})")

        if all(coord is not None and pd.notna(coord) for coord in [x1, y1, x2, y2]):
            if edge_info['id1'] == source_info['id']:
                # print("??????????????????????????STOP??????????????????????????")
                bearing = calculate_bearing_original(x1, y1, x2, y2)
            else:
                bearing = calculate_bearing_original(x2, y2, x1, y1, )
            direction = convert_bearing_to_direction_original(bearing)
            # print(f"📐 Calculated from coordinates: {direction}")
            return direction
        # else:
        #     print(f"❌ Missing coordinates: x1={x1}, y1={y1}, x2={x2}, y2={y2}")

    # print(f"❌ No valid coordinates found")
    return None


# Helper functions for original logic processing
def is_same_street_original(street1, street2, node1_id, node2_id, edge_info):
    """Original is_same_street logic (preserved)"""
    # print('!!street1', street1)
    # print('!!street2', street2)

    if not street1 or not street2:
        return False

    street1 = str(street1).strip().lower()
    street2 = str(street2).strip().lower()

    if street1 == street2:
        return True
    if edge_info['type'] == 'on_same_street':
        return True

    if street1 in street2 or street2 in street1:
        return True

    def clean_street_name(name):
        clean_terms = ['crossing', 'street', 'st', 'avenue', 'ave', 'road', 'rd', 'boulevard', 'blvd']
        words = name.split()
        return ' '.join([word for word in words if word not in clean_terms])

    clean1 = clean_street_name(street1)
    clean2 = clean_street_name(street2)

    return clean1 == clean2 and clean1 != ''


def crossing_streets_original(node_id, target_id, edges_gdf, nodes_gdf):
    """Original crossing_streets logic (preserved)"""
    # print(f"🔍 Checking for crossing between {node_id} and {target_id}")

    try:
        edge_rows = edges_gdf[
            ((edges_gdf['id1'] == node_id) & (edges_gdf['id2'] == target_id)) |
            ((edges_gdf['id1'] == target_id) & (edges_gdf['id2'] == node_id))
            ]

        if len(edge_rows) > 0:
            edge_info = edge_rows.iloc[0].to_dict()
            if 'crossing_id' in edge_info and pd.notna(edge_info['crossing_id']):
                crossing_x, crossing_y = get_crossing_coordinates_original(target_id, edge_info, nodes_gdf)
                return crossing_x, crossing_y
        else:
            # print(f"❌ No crossing edge found between {node_id} and {target_id}")
            return None, None

    except Exception as e:
        # print(f"❌ Error checking for crossing: {e}")
        return None, None


def enhanced_boundary_relationship_analysis_original(source_info, target_info, edge_info, nodes_gdf):
    """Original enhanced_boundary_relationship_analysis logic (preserved)"""
    boundary_result = {
        'description': 'boundary',
        'direction': None,
        'reverse_direction': None,
        'reference_id': None,
        'spatial_relationship': None,
        'detailed_analysis': None
    }

    # Identify which node is the AOI/polygon and which is the street/line
    aoi_info, street_info = identify_aoi_and_street_nodes_original(source_info, target_info)
    # print('???????????????aoi_info????????????????', aoi_info)
    if not aoi_info or not street_info:
        return boundary_result

    # Get geometries
    aoi_geometry = aoi_info.get('geometry')
    street_geometry = street_info.get('geometry')

    if not aoi_geometry or not street_geometry:
        return boundary_result

    # Calculate comprehensive directional relationship
    try:
        if isinstance(aoi_geometry, (Polygon, MultiPolygon)):
            directional_analysis = get_directional_relationship_original(aoi_geometry, street_geometry, street_info)

            if not directional_analysis.get('error'):
                directions = directional_analysis['directions']

                boundary_result.update({
                    'description': f"boundary_{directions['polygon_to_street']}" if directions[
                        'polygon_to_street'] else 'boundary',
                    'direction': directions['polygon_to_street'],
                    'reverse_direction': directions['street_to_polygon'],
                    'reference_id': aoi_info.get('id'),
                    'spatial_relationship': 'polygon_boundary',
                    'detailed_analysis': directional_analysis
                })
        else:
            # Fallback to coordinate-based calculation
            bearing, direction = calculate_direction_from_coordinates_aoi_street_original(aoi_info, street_info)
            if direction:
                reverse_direction = convert_bearing_to_cardinal_direction_original(
                    calculate_reverse_bearing_original(
                        convert_cardinal_direction_to_bearing_original(direction)
                    )
                )

                boundary_result.update({
                    'description': f"boundary_{direction}" if direction else 'boundary',
                    'direction': direction,
                    'reverse_direction': reverse_direction,
                    'reference_id': aoi_info.get('id'),
                    'spatial_relationship': 'coordinate_based'
                })

    except Exception as e:
        print(f"Warning: Could not analyze boundary relationship: {e}")

    return boundary_result


def boundary_street_original(node_id, target_id, edge_info, nodes_gdf):
    """Original boundary_street logic (preserved)"""
    if 'boundary' in edge_info['type'] or 'intersect' in edge_info['type']:
        source_info = nodes_gdf[nodes_gdf['id'] == target_id].iloc[0].to_dict()
        target_info = nodes_gdf[nodes_gdf['id'] == node_id].iloc[0].to_dict()

        boundary_info = enhanced_boundary_relationship_analysis_original(source_info, target_info, edge_info, nodes_gdf)
        return boundary_info.get('direction'), boundary_info.get('reverse_direction')
    else:
        return None, None


def process_crossing_case_original(node_id, target_id, edges_gdf, nodes_gdf, crossing_x, crossing_y):
    """Process crossing case with original logic"""
    next_x, next_y = crossing_streets_original(node_id, target_id, edges_gdf, nodes_gdf)

    if next_x is not None and next_y is not None:
        bearing = calculate_bearing_original(crossing_x, crossing_y, next_x, next_y)
        direction = convert_bearing_to_direction_original(bearing)
        distance = geodesic((crossing_x, crossing_y), (next_x, next_y)).kilometers * 1000
        #
        # print(f"📐 Direction calculation:")
        # print(f"   From: ({crossing_x}, {crossing_y}) [Current crossing]")
        # print(f"   To:   ({next_x}, {next_y}) [Next crossing point]")
        # print(f"   Bearing: {bearing:.1f}°")
        # print(f"   Direction: {direction}")
        # print(f"   Distance: {distance}m")

        return direction, distance

    return None, None


def process_boundary_case_original(node_id, target_id, edge_info, nodes_gdf, crossing_x, crossing_y):
    """Process boundary case with original logic"""
    direction, reversed_direction = boundary_street_original(node_id, target_id, edge_info, nodes_gdf)
    distance = 0

    # print(f"📐 Direction calculation:")
    # print(f"   From: ({crossing_x}, {crossing_y}) [Current crossing]")

    if nodes_gdf[(nodes_gdf['id'] == node_id)].geometry.geom_type.iloc[0] in ['LineString', 'MultiLineString']:
        street = nodes_gdf[(nodes_gdf['id'] == node_id)].name.iloc[0]
        aoi = nodes_gdf[(nodes_gdf['id'] == target_id)].name.iloc[0]
        # print(f"   To:   boundary road {street} by {aoi}")
        # print(f"   polygon_to_street Direction: {direction}")
        # print(f"   Distance: {distance}m")
        return direction, distance
    else:
        street = nodes_gdf[(nodes_gdf['id'] == target_id)].name.iloc[0]
        aoi = nodes_gdf[(nodes_gdf['id'] == node_id)].name.iloc[0]
        # print(f"   To:   boundary road {street} by {aoi}")
        # print(f"   street_to_polygon reversed_direction: {reversed_direction}")
        # print(f"   Distance: {distance}m")
        return reversed_direction, distance


def process_nearest_case_original(node_id, target_id, edge_info, nodes_gdf, crossing_x, crossing_y):
    """Process nearest case with original logic"""
    # print(f"📐 Direction calculation:")
    # print(f"   From: ({crossing_x}, {crossing_y}) [Current crossing]")

    if nodes_gdf[(nodes_gdf['id'] == node_id)].geometry.geom_type.iloc[0] in ['LineString', 'MultiLineString']:
        bearing = edge_info['bearing']
        distance = edge_info['distance']
        direction = convert_bearing_to_direction_original(bearing)
        # print(
        #     f"   To:   nearest street {nodes_gdf[(nodes_gdf['id'] == node_id)].name.iloc[0]} from {nodes_gdf[(nodes_gdf['id'] == target_id)].name.iloc[0]}")
        # print(f"   polygon_to_street Direction: {direction}")
        # print(f"   Distance: {distance}m")
        return direction, distance
    else:
        bearing = calculate_reverse_bearing_original(edge_info['bearing'])
        direction = convert_bearing_to_direction_original(bearing)
        distance = edge_info['distance']
        # print(
        #     f"   To:   nearest street {nodes_gdf[(nodes_gdf['id'] == target_id)].name.iloc[0]} from {nodes_gdf[(nodes_gdf['id'] == node_id)].name.iloc[0]}")
        # print(f"   polygon_to_street Direction: {direction}")
        # print(f"   Distance: {distance}m")
        return direction, distance


def process_same_street_case_original(node_data, crossing_x, crossing_y, target_street_name):
    """Process same street case with original logic"""
    node_geom = node_data.get('geometry')
    if node_geom:
        try:
            next_x, next_y = node_geom.x, node_geom.y
        except:
            next_x, next_y = node_geom.centroid.x, node_geom.centroid.y

        if next_x is not None and next_y is not None:
            bearing = calculate_bearing_original(crossing_x, crossing_y, next_x, next_y)
            direction = convert_bearing_to_direction_original(bearing)
            distance = geodesic((crossing_x, crossing_y), (next_x, next_y)).kilometers * 1000

            # print(f"📐 Direction calculation:")
            # print(f"   From: ({crossing_x}, {crossing_y}) [Current crossing]")
            # print(f"   To:   ({next_x}, {next_y}) [Next point on {target_street_name}]")
            # print(f"   Bearing: {bearing:.1f}°")
            # print(f"   Direction: {direction}")
            # print(f"   Distance: {distance}m")

            return direction, distance

    return None, None


def find_direction_from_path_continuation_original(target_id, target_street_name, current_path, current_position,
                                                   nodes_gdf, edge_info, edges_gdf=None):
    """
    Original find_direction_from_path_continuation logic (preserved exactly)
    """
    # print(f"🔍 Looking for next point on '{target_street_name}' from path continuation")
    print(f"Current position in path: {current_position}, Target ID: {target_id}")
    print(f"Complete path: {current_path}")

    # Get crossing coordinates (the reference point)
    if "Intersection" in nodes_gdf[nodes_gdf['id'] == target_id].name.iloc[0]:
        crossing_x, crossing_y = (nodes_gdf[nodes_gdf['id'] == target_id].iloc[0].geometry.x,
                                  nodes_gdf[nodes_gdf['id'] == target_id].iloc[0].geometry.y)
        # print(f"📍 Current Intersection coordinates: ({crossing_x}, {crossing_y})")
    elif nodes_gdf[nodes_gdf['id'] == target_id].geometry.geom_type.iloc[0] in ['Point']:
        crossing_x, crossing_y = (nodes_gdf[nodes_gdf['id'] == target_id].iloc[0].geometry.x,
                                  nodes_gdf[nodes_gdf['id'] == target_id].iloc[0].geometry.y)
        # print(f"📍 Current Point coordinates: ({crossing_x}, {crossing_y})")
    elif nodes_gdf[nodes_gdf['id'] == target_id].geometry.geom_type.iloc[0] in ['MultiLineString', 'LineString']:
        crossing_x, crossing_y = get_crossing_coordinates_original(target_id, edge_info, nodes_gdf)
        # print(f"📍 Current Crossing coordinates: ({crossing_x}, {crossing_y})")
    else:
        # print(f"📍 Cannot find current coordinates")
        return None, None

    # Look at remaining nodes in the path
    remaining_path = current_path[current_position + 2:]
    if len(remaining_path) != 0:
        # print(f"🔍 Remaining path to search: {remaining_path}")

        # Search for the next point on the same street
        for i, node_id in enumerate(remaining_path):
            node_info = nodes_gdf[nodes_gdf['id'] == node_id]
            if len(node_info) == 0:
                continue

            node_data = node_info.iloc[0]
            node_street_name = node_data.get('street') or node_data.get('name', '')

            # print(f"🔍 Checking node {node_id}: street='{node_street_name}'")
            found_edge = edges_gdf[
                ((edges_gdf['id1'] == target_id) & (edges_gdf['id2'] == node_id)) |
                ((edges_gdf['id1'] == node_id) & (edges_gdf['id2'] == target_id))
                ]

            if len(found_edge) > 0:
                found_edge_info = found_edge.iloc[0].to_dict()

                # Process different edge types (original logic preserved)
                if is_same_street_original(target_street_name, node_street_name, target_id, node_id, found_edge_info):
                    # Handle same street case
                    return process_same_street_case_original(node_data, crossing_x, crossing_y, target_street_name)

                elif 'crossing' in found_edge_info['type'] and found_edge_info['type'] != "boundary_crossing":
                    # Handle crossing case
                    return process_crossing_case_original(node_id, target_id, edges_gdf, nodes_gdf, crossing_x,
                                                          crossing_y)

                elif 'boundary' in found_edge_info['type'] or 'intersect' in found_edge_info['type']:
                    # Handle boundary case
                    return process_boundary_case_original(node_id, target_id, found_edge_info, nodes_gdf, crossing_x,
                                                          crossing_y)

                elif found_edge_info['type'] == 'nearest':
                    # Handle nearest case
                    return process_nearest_case_original(node_id, target_id, found_edge_info, nodes_gdf, crossing_x,
                                                         crossing_y)

    # print("❌ No next point found on same street in remaining path")
    return None, None


def find_target_street_actual_direction_original(target_id, nodes_gdf, edge_info=None,
                                                 current_position=None, current_path=None, edges_gdf=None):
    """
    Original find_target_street_actual_direction logic (preserved)
    """
    try:
        # Get target node info
        target_node = nodes_gdf[nodes_gdf['id'] == target_id]
        if len(target_node) == 0:
            return None, None

        target_row = target_node.iloc[0]
        target_street_name = target_row.get('street') or target_row.get('name')
        if not target_street_name:
            # print(f"Warning: No street name found for target node {target_id}")
            return None, None

        # Method 1: Use the complete path to find next point on target street
        if current_path and current_position is not None:
            next_direction, distance = find_direction_from_path_continuation_original(
                target_id, target_street_name, current_path, current_position, nodes_gdf, edge_info, edges_gdf
            )
            return next_direction, distance

    except Exception as e:
        # print(f"Warning: Could not find target street direction: {e}")
        return None, None


def get_coordinate_from_node_original(node_info, coord_type):
    """
    Original get_coordinate_from_node logic (preserved)
    """
    # Direct coordinate field
    coord = node_info.get(coord_type)
    if coord is not None and pd.notna(coord):
        return coord

    # From geometry
    geom = node_info.get('geometry')
    if geom:
        if hasattr(geom, coord_type):
            return getattr(geom, coord_type)
        elif hasattr(geom, 'centroid'):
            return getattr(geom.centroid, coord_type)

    return None


def calculate_crossing_direction_optimized(source_info, target_info, nodes_gdf, edge_info, position, path=None,
                                           edges_gdf=None):
    """
    Original calculate_crossing_direction logic with optimizations
    """
    source_id = source_info.get('id')
    target_id = target_info.get('id')

    actual_target_direction, distance = find_target_street_actual_direction_original(
        target_id, nodes_gdf, edge_info, position, path, edges_gdf
    )

    return actual_target_direction, distance


def calculate_distance_from_geometries(source_geometry, target_geometry, source_is_point, target_is_point):
    """
    Calculate distance from geometries when edge_info distance is missing.
    Handles different geometry type combinations.
    """
    try:
        from geopy.distance import geodesic

        if source_is_point and target_is_point:
            # Point to Point: direct calculation
            source_point = source_geometry
            target_point = target_geometry
            distance = geodesic((source_point.y, source_point.x), (target_point.y, target_point.x)).meters
            return distance

        elif source_is_point and not target_is_point:
            # Point to LineString: from point to nearest point on line
            source_point = source_geometry
            if isinstance(target_geometry, (LineString, MultiLineString)):
                nearest_point_on_line = nearest_points(source_point, target_geometry)[1]
                distance = geodesic((source_point.y, source_point.x),
                                    (nearest_point_on_line.y, nearest_point_on_line.x)).meters
                return distance
            else:
                # Fallback to centroid
                target_centroid = target_geometry.centroid
                distance = geodesic((source_point.y, source_point.x), (target_centroid.y, target_centroid.x)).meters
                return distance

        elif not source_is_point and target_is_point:
            # LineString to Point: from nearest point on line to point
            target_point = target_geometry
            if isinstance(source_geometry, (LineString, MultiLineString)):
                nearest_point_on_line = nearest_points(target_point, source_geometry)[1]
                distance = geodesic((nearest_point_on_line.y, nearest_point_on_line.x),
                                    (target_point.y, target_point.x)).meters
                return distance
            else:
                # Fallback to centroid
                source_centroid = source_geometry.centroid
                distance = geodesic((source_centroid.y, source_centroid.x), (target_point.y, target_point.x)).meters
                return distance

        else:
            # LineString to LineString: from nearest point on source to nearest point on target
            if isinstance(source_geometry, (LineString, MultiLineString)) and isinstance(target_geometry,
                                                                                         (LineString, MultiLineString)):
                nearest_points_pair = nearest_points(source_geometry, target_geometry)
                source_point = nearest_points_pair[0]
                target_point = nearest_points_pair[1]
                distance = geodesic((source_point.y, source_point.x), (target_point.y, target_point.x)).meters
                return distance
            else:
                # Fallback to centroids
                source_centroid = source_geometry.centroid
                target_centroid = target_geometry.centroid
                distance = geodesic((source_centroid.y, source_centroid.x),
                                    (target_centroid.y, target_centroid.x)).meters
                return distance

    except Exception as e:
        # print(f"Warning: Could not calculate distance from geometries: {e}")
        return None


def calculate_bearing_from_geometries(source_geometry, target_geometry, source_is_point, target_is_point):
    """
    Calculate bearing from geometries when edge_info bearing is missing.
    Handles different geometry type combinations.
    """
    try:
        if source_is_point and target_is_point:
            # Point to Point: direct bearing calculation
            source_point = source_geometry
            target_point = target_geometry
            bearing = calculate_bearing_between_points_original(source_point, target_point)
            return bearing

        elif source_is_point and not target_is_point:
            # Point to LineString: from point to nearest point on line
            source_point = source_geometry
            if isinstance(target_geometry, (LineString, MultiLineString)):
                nearest_point_on_line = nearest_points(source_point, target_geometry)[1]
                bearing = calculate_bearing_between_points_original(source_point, nearest_point_on_line)
                return bearing
            else:
                # Fallback to centroid
                target_centroid = target_geometry.centroid
                bearing = calculate_bearing_between_points_original(source_point, target_centroid)
                return bearing

        elif not source_is_point and target_is_point:
            # LineString to Point: from nearest point on line to point
            target_point = target_geometry
            if isinstance(source_geometry, (LineString, MultiLineString)):
                nearest_point_on_line = nearest_points(target_point, source_geometry)[1]
                bearing = calculate_bearing_between_points_original(nearest_point_on_line, target_point)
                return bearing
            else:
                # Fallback to centroid
                source_centroid = source_geometry.centroid
                bearing = calculate_bearing_between_points_original(source_centroid, target_point)
                return bearing

        else:
            # LineString to LineString: from nearest point on source to nearest point on target
            if isinstance(source_geometry, (LineString, MultiLineString)) and isinstance(target_geometry,
                                                                                         (LineString, MultiLineString)):
                nearest_points_pair = nearest_points(source_geometry, target_geometry)
                source_point = nearest_points_pair[0]
                target_point = nearest_points_pair[1]
                bearing = calculate_bearing_between_points_original(source_point, target_point)
                return bearing
            else:
                # Fallback to centroids
                source_centroid = source_geometry.centroid
                target_centroid = target_geometry.centroid
                bearing = calculate_bearing_between_points_original(source_centroid, target_centroid)
                return bearing

    except Exception as e:
        # print(f"Warning: Could not calculate bearing from geometries: {e}")
        return None


def calculate_nearest_direction_original(source_info, target_info, edge_info):
    """
    Calculate nearest direction and distance between source and target.
    Returns both direction and distance.
    """
    # Get geometry types
    source_geometry = source_info.get('geometry')
    target_geometry = target_info.get('geometry')

    source_is_point = is_point_geometry_original(source_geometry)
    target_is_point = is_point_geometry_original(target_geometry)

    bearing = edge_info.get('bearing')
    distance = edge_info.get('distance')

    # Calculate bearing if missing
    if bearing is None:
        bearing = calculate_bearing_from_geometries(source_geometry, target_geometry, source_is_point, target_is_point)
        if bearing is None:
            return 'unknown', None

    # Calculate distance if missing
    if distance == 0.0:
        distance = calculate_distance_from_geometries(source_geometry, target_geometry, source_is_point, target_is_point)
        if distance is None:
            distance = 0  # Default to 0 if calculation fails

    # Determine direction based on Point type location
    if source_is_point and not target_is_point:
        direction = convert_bearing_to_direction_original(bearing)
        return direction, distance
    elif target_is_point and not source_is_point:
        reversed_bearing = (bearing + 180) % 360
        direction = convert_bearing_to_direction_original(reversed_bearing)
        return direction, distance
    elif source_is_point and target_is_point:
        direction = convert_bearing_to_direction_original(bearing)
        return direction, distance
    else:
        direction = convert_bearing_to_direction_original(bearing)
        return direction, distance


def is_point_geometry_original(geometry):
    """
    Original is_point_geometry logic (preserved)
    """
    if geometry is None:
        return False

    if isinstance(geometry, Point):
        return True

    if isinstance(geometry, str):
        return 'POINT' in geometry.upper()

    if hasattr(geometry, 'geom_type'):
        return geometry.geom_type == 'Point'

    return False


def get_direction_for_boundary_edge_original(source_info, target_info, edge_info, nodes_gdf):
    """
    Modified get_direction_for_boundary_edge logic with AOI-Line relationship handling
    """
    edge_type = edge_info.get('type', '')

    if 'boundary' in edge_type or 'intersect' in edge_type:
        # Analyze boundary relationship with improved AOI-Line logic
        boundary_analysis = analyze_boundary_relationship_enhanced(source_info, target_info, edge_info, nodes_gdf)

        direction = boundary_analysis.get('direction')
        spatial_relationship = boundary_analysis.get('spatial_relationship')
        relation_string = boundary_analysis.get('relation_string', 'boundary')

        # Create descriptive direction string
        if direction:
            if spatial_relationship == 'polygon_boundary':
                direction_str = f"{direction}"  # Just the direction without _of_AOI suffix
            else:
                direction_str = f"{direction}"
        else:
            direction_str = "boundary_unknown"

        # Get length information
        length = edge_info.get('distance', edge_info.get('crossing_distance_meters', 0))

        return direction_str, length

    return None, 0


def calculate_line_to_aoi_direction(aoi_geometry, line_geometry, line_info):
    """
    Calculate direction from nearest point of Line to AOI centroid
    For case: "(South St, bounds, Battery Park)[0.0m, SE]"
    """
    try:
        # Get AOI centroid
        if isinstance(aoi_geometry, MultiPolygon):
            aoi_centroid = aoi_geometry.centroid
        else:
            aoi_centroid = aoi_geometry.centroid

        # Get nearest point on line to AOI
        if isinstance(line_geometry, (LineString, MultiLineString)):
            # Find nearest point on line to AOI centroid
            nearest_point_on_line = nearest_points(aoi_geometry, line_geometry)[1]
        else:
            # Fallback to line centroid
            nearest_point_on_line = line_geometry.centroid

        # Calculate bearing from nearest point on line TO AOI centroid
        bearing = calculate_bearing_between_points_original(nearest_point_on_line, aoi_centroid)
        direction = convert_bearing_to_cardinal_direction_original(bearing)

        # print(f"Line to AOI direction calculation:")
        # print(f"   Nearest point on line: ({nearest_point_on_line.x:.6f}, {nearest_point_on_line.y:.6f})")
        # print(f"   AOI centroid: ({aoi_centroid.x:.6f}, {aoi_centroid.y:.6f})")
        # print(f"   Bearing (line->AOI): {bearing:.1f}°")
        # print(f"   Direction: {direction}")

        return bearing, direction

    except Exception as e:
        print(f"Warning: Could not calculate line to AOI direction: {e}")
        return None, None


def calculate_aoi_to_line_direction(aoi_geometry, line_geometry, aoi_info):
    """
    Calculate direction from AOI centroid to nearest point on Line
    For case: "(Battery Park, near, South St)[0.0m, NW]"
    """
    try:
        # Get AOI centroid
        if isinstance(aoi_geometry, MultiPolygon):
            aoi_centroid = aoi_geometry.centroid
        else:
            aoi_centroid = aoi_geometry.centroid

        # Get nearest point on line to AOI
        if isinstance(line_geometry, (LineString, MultiLineString)):
            # Find nearest point on line to AOI centroid
            nearest_point_on_line = nearest_points(aoi_geometry, line_geometry)[1]
        else:
            # Fallback to line centroid
            nearest_point_on_line = line_geometry.centroid

        # Calculate bearing from AOI centroid TO nearest point on line
        bearing = calculate_bearing_between_points_original(aoi_centroid, nearest_point_on_line)
        direction = convert_bearing_to_cardinal_direction_original(bearing)

        # print(f"AOI to line direction calculation:")
        # print(f"   AOI centroid: ({aoi_centroid.x:.6f}, {aoi_centroid.y:.6f})")
        # print(f"   Nearest point on line: ({nearest_point_on_line.x:.6f}, {nearest_point_on_line.y:.6f})")
        # print(f"   Bearing (AOI->line): {bearing:.1f}°")
        # print(f"   Direction: {direction}")

        return bearing, direction

    except Exception as e:
        print(f"Warning: Could not calculate AOI to line direction: {e}")
        return None, None


def analyze_boundary_relationship_enhanced(source_info, target_info, edge_info, nodes_gdf):
    """
    Enhanced analyze_boundary_relationship with proper AOI-Line handling
    """
    boundary_result = {
        'description': 'boundary',
        'direction': None,
        'reference_id': None,
        'spatial_relationship': None,
        'bearing': None,
        'relation_string': 'boundary',
        'relationship_type': None
    }

    # Identify which node is the AOI/polygon and which is the street/line
    aoi_info, street_info, relationship_type = identify_aoi_and_street_nodes_enhanced(source_info, target_info)

    if not aoi_info or not street_info:
        return boundary_result

    # Get geometries
    aoi_geometry = aoi_info.get('geometry')
    street_geometry = street_info.get('geometry')

    if not aoi_geometry or not street_geometry:
        return boundary_result

    # Calculate spatial relationship based on AOI-Line direction
    try:
        if relationship_type == 'aoi_to_line':
            # Source is AOI, Target is Line: direction from nearest point of Line to AOI
            bearing, direction = calculate_line_to_aoi_direction(aoi_geometry, street_geometry, street_info)
            relation_string = 'bounds'

        elif relationship_type == 'line_to_aoi':
            # Source is Line, Target is AOI: direction from AOI to nearest point on Line
            bearing, direction = calculate_aoi_to_line_direction(aoi_geometry, street_geometry, aoi_info)
            relation_string = 'near'

        else:
            # Fallback to original polygon logic
            bearing, direction = calculate_street_direction_from_polygon_original(
                aoi_geometry, street_geometry, street_info
            )
            relation_string = 'boundary'

        boundary_result.update({
            'description': f"boundary_{direction}" if direction else 'boundary',
            'direction': direction,
            'reference_id': aoi_info.get('id'),
            'spatial_relationship': 'polygon_boundary',
            'bearing': bearing,
            'relation_string': relation_string,
            'relationship_type': relationship_type
        })

    except Exception as e:
        print(f"Warning: Could not analyze boundary relationship: {e}")

    return boundary_result


def calculate_direction_from_coordinates_aoi_street_original(aoi_info, street_info):
    """
    Original calculate_direction_from_coordinates_aoi_street logic (preserved)
    """
    # Get coordinates from node info
    aoi_x = aoi_info.get('x')
    aoi_y = aoi_info.get('y')
    street_x = street_info.get('x')
    street_y = street_info.get('y')

    # Try to extract from geometry if coordinates not directly available
    if not all([aoi_x, aoi_y, street_x, street_y]):
        aoi_geom = aoi_info.get('geometry')
        street_geom = street_info.get('geometry')

        if aoi_geom and hasattr(aoi_geom, 'centroid'):
            aoi_x, aoi_y = aoi_geom.centroid.x, aoi_geom.centroid.y
        if street_geom and hasattr(street_geom, 'centroid'):
            street_x, street_y = street_geom.centroid.x, street_geom.centroid.y

    if all([aoi_x, aoi_y, street_x, street_y]):
        bearing = calculate_bearing_original(aoi_x, aoi_y, street_x, street_y)
        return bearing, convert_bearing_to_cardinal_direction_original(bearing)

    return None, None


def calculate_street_direction_from_polygon_original(polygon_geometry, street_geometry, street_info):
    """
    Original calculate_street_direction_from_polygon logic (preserved)
    """
    try:
        # Get polygon centroid
        if isinstance(polygon_geometry, MultiPolygon):
            polygon_centroid = polygon_geometry.centroid
        else:
            polygon_centroid = polygon_geometry.centroid

        # Get street point
        if isinstance(street_geometry, Point):
            street_point = street_geometry
        elif isinstance(street_geometry, LineString):
            # Use closest point on line to polygon
            nearest_point = nearest_points(polygon_geometry, street_geometry)[1]
            street_point = nearest_point
        else:
            # Fallback to coordinates
            x = street_info.get('x') or (street_geometry.centroid.x if hasattr(street_geometry, 'centroid') else 0)
            y = street_info.get('y') or (street_geometry.centroid.y if hasattr(street_geometry, 'centroid') else 0)
            street_point = Point(x, y)

        # Calculate bearing from polygon centroid to street
        bearing_polygon_to_street = calculate_bearing_between_points_original(polygon_centroid, street_point)
        direction_polygon_to_street = convert_bearing_to_cardinal_direction_original(bearing_polygon_to_street)

        return bearing_polygon_to_street, direction_polygon_to_street

    except Exception as e:
        print(f"Warning: Could not calculate street direction from polygon: {e}")
        return None, None


def get_directional_relationship_original(polygon_geometry, street_geometry, street_info):
    """Original get_directional_relationship logic (preserved)"""
    direction_info = calculate_street_direction_from_polygon_original(
        polygon_geometry, street_geometry, street_info
    )

    if not direction_info[0]:  # bearing is first element
        return {'error': 'Could not calculate directional relationship'}

    # Calculate distance between centroids
    polygon_centroid = polygon_geometry.centroid
    if isinstance(street_geometry, Point):
        street_point = street_geometry
    else:
        street_point = nearest_points(polygon_geometry, street_geometry)[1]

    distance_meters = geodesic(
        (polygon_centroid.y, polygon_centroid.x),
        (street_point.y, street_point.x)
    ).meters

    # Create descriptive relationship
    polygon_to_street_dir = direction_info[1]  # direction is second element
    street_to_polygon_dir = convert_bearing_to_cardinal_direction_original(
        calculate_reverse_bearing_original(direction_info[0])
    )

    return {
        'directions': {
            'polygon_to_street': polygon_to_street_dir,
            'street_to_polygon': street_to_polygon_dir
        },
        'distance_meters': distance_meters,
        'relationships': {
            'street_is_X_of_polygon': f"Street is {polygon_to_street_dir} of polygon",
            'polygon_is_Y_of_street': f"Polygon is {street_to_polygon_dir} of street"
        }
    }


def calculate_reverse_bearing_original(bearing):
    """Original calculate_reverse_bearing logic (preserved)"""
    return (bearing + 180) % 360


def convert_cardinal_direction_to_bearing_original(cardinal_direction):
    """Original convert_cardinal_direction_to_bearing logic (preserved)"""
    cardinal_to_bearing = {
        'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5,
        'E': 90, 'ESE': 112.5, 'SE': 135, 'SSE': 157.5,
        'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5,
        'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5
    }

    return cardinal_to_bearing.get(cardinal_direction, 0)


def is_direction_change_original(prev_triple, curr_triple):
    """
    Original is_direction_change logic (preserved)
    """
    # If current triple and prev_triple are on the same street, no direction change
    if (curr_triple['middle_component'] == prev_triple['middle_component'] and
            curr_triple['edge_type'] == prev_triple['edge_type']):
        return False

    # If current triple is a decision point, it's likely a direction change
    if curr_triple.get('triple_type') == 'decision_point':
        return True

    # Compare directions if available
    try:
        prev_direction = re.search(r'\(([^)]+)\)', prev_triple.get('direction', '')).group(1)
        curr_direction = re.search(r'\(([^)]+)\)', curr_triple.get('direction', '')).group(1)

        if prev_direction and curr_direction and prev_direction != curr_direction:
            return True
    except:
        pass

    # Compare street names
    prev_street = prev_triple.get('middle_component', '')
    curr_street = curr_triple.get('middle_component', '')

    if prev_street and curr_street and prev_street != curr_street:
        return True

    return False


# Update the optimized functions to use original logic
def is_direction_change_simple(prev_triple, curr_triple):
    """Use original direction change logic instead of simplified"""
    return is_direction_change_original(prev_triple, curr_triple)


def format_triples_as_string_simple(triples):
    """Use original triple formatting"""
    if not triples:
        return ""
    return " -> ".join(triple['formatted'] for triple in triples)
    import pandas as pd


class GraphProcessor:
    """
    Graph processing class with GPU acceleration support - FIXED FOR NODE ID GAPS
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
        """Create mapping between NetworkX and cuGraph node IDs - FIXED FOR GAPS"""
        try:
            nodes = list(self.nx_graph.nodes())

            print(f"📊 Graph analysis:")
            print(f"   Total nodes: {len(nodes)}")
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
            print(f"   NetworkX node range: {min(nodes)} to {max(nodes)}")
            print(f"   cuGraph node range: 0 to {len(nodes) - 1}")
            print(f"   Mapping examples:")

            # Show some mapping examples
            sample_nodes = sorted(nodes)[:5]
            for nx_node in sample_nodes:
                cu_node = self.nx_to_cu_mapping[nx_node]
                print(f"      {nx_node} → {cu_node}")

        except Exception as e:
            print(f"❌ Failed to create node mapping: {e}")
            import traceback
            traceback.print_exc()
            self.needs_mapping = False

    def _convert_to_cugraph(self):
        """Convert NetworkX graph to cuGraph with empty graph handling - FIXED"""
        try:
            print("🔄 Starting cuGraph conversion...")

            # FIXED: Create node mapping first
            self._create_node_mapping()

            # Extract edges
            edges = list(self.nx_graph.edges())

            # FIXED: Handle empty graph gracefully
            if not edges:
                print("⚠️ Graph has no edges - disabling GPU acceleration")
                print(f"   Nodes: {self.nx_graph.number_of_nodes()}, Edges: {len(edges)}")
                self.use_gpu = False
                self.cu_graph = None
                return  # Exit gracefully instead of raising error

            print(f"📊 Processing {len(edges)} edges...")

            # FIXED: Always map edges if needed, with validation
            if self.needs_mapping:
                mapped_edges = []
                skipped_edges = 0

                for u, v in edges:
                    cu_u = self.nx_to_cu_mapping.get(u)
                    cu_v = self.nx_to_cu_mapping.get(v)

                    if cu_u is not None and cu_v is not None:
                        mapped_edges.append((cu_u, cu_v))
                    else:
                        skipped_edges += 1
                        if skipped_edges <= 5:  # Only show first few errors
                            print(f"⚠️ Skipping edge {u}→{v}: mapping failed (u→{cu_u}, v→{cu_v})")

                if skipped_edges > 0:
                    print(f"⚠️ Total skipped edges: {skipped_edges}/{len(edges)}")

                edges = mapped_edges
                print(f"✅ Successfully mapped {len(edges)} edges")
            else:
                print("✅ No edge mapping needed")

            # FIXED: Check again after mapping
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

            # Add weights if available
            if nx.is_weighted(self.nx_graph):
                print("🔄 Processing edge weights...")
                if self.needs_mapping:
                    weights = []
                    for u, v in self.nx_graph.edges():
                        cu_u = self.nx_to_cu_mapping.get(u)
                        cu_v = self.nx_to_cu_mapping.get(v)
                        if cu_u is not None and cu_v is not None:
                            weights.append(self.nx_graph[u][v].get('weight', 1.0))

                    if len(weights) != len(edges):
                        print(f"⚠️ Weight count mismatch: {len(weights)} weights for {len(edges)} edges")
                        # Pad or truncate weights to match edges
                        if len(weights) < len(edges):
                            weights.extend([1.0] * (len(edges) - len(weights)))
                        else:
                            weights = weights[:len(edges)]
                else:
                    weights = [self.nx_graph[u][v].get('weight', 1.0) for u, v in edges]

                self.edge_df['weight'] = weights
                print(f"✅ Added {len(weights)} edge weights")

            # Create cuGraph
            print("🔄 Creating cuGraph object...")
            self.cu_graph = cugraph.Graph()

            if 'weight' in self.edge_df.columns:
                self.cu_graph.from_cudf_edgelist(
                    self.edge_df,
                    source='src',
                    destination='dst',
                    edge_attr='weight'
                )
            else:
                self.cu_graph.from_cudf_edgelist(
                    self.edge_df,
                    source='src',
                    destination='dst'
                )

            print("✅ Successfully converted to cuGraph with node mapping")
            print(f"   Final cuGraph: {len(self.cu_to_nx_mapping)} nodes, {len(edges)} edges")

        except Exception as e:
            print(f"❌ Failed to convert to cuGraph: {e}")
            import traceback
            traceback.print_exc()
            self.use_gpu = False
            self.cu_graph = None
            # Clear mappings on failure
            self.nx_to_cu_mapping.clear()
            self.cu_to_nx_mapping.clear()
            self.needs_mapping = False

    def _map_nx_to_cu_node(self, nx_node):
        """Convert NetworkX node ID to cuGraph node ID with validation"""
        if not self.needs_mapping:
            return nx_node

        cu_node = self.nx_to_cu_mapping.get(nx_node)
        if cu_node is None:
            print(f"⚠️ Warning: Node {nx_node} not found in mapping")
        return cu_node

    def _map_cu_to_nx_node(self, cu_node):
        """Convert cuGraph node ID to NetworkX node ID with validation"""
        if not self.needs_mapping:
            return cu_node

        nx_node = self.cu_to_nx_mapping.get(cu_node)
        if nx_node is None:
            print(f"⚠️ Warning: cuGraph node {cu_node} not found in reverse mapping")
        return nx_node

    def _validate_mapping(self):
        """Validate that the node mapping is complete and correct"""
        if not self.needs_mapping:
            return True

        nx_nodes = set(self.nx_graph.nodes())
        mapped_nx_nodes = set(self.nx_to_cu_mapping.keys())

        missing_nodes = nx_nodes - mapped_nx_nodes
        if missing_nodes:
            print(f"❌ Mapping validation failed: {len(missing_nodes)} nodes not mapped")
            print(f"   Examples of missing nodes: {list(missing_nodes)[:5]}")
            return False

        # Check reverse mapping
        expected_cu_nodes = set(range(len(nx_nodes)))
        mapped_cu_nodes = set(self.cu_to_nx_mapping.keys())

        if expected_cu_nodes != mapped_cu_nodes:
            print(f"❌ Reverse mapping validation failed")
            return False

        print("✅ Node mapping validation passed")
        return True

    # Include all the other methods from the previous implementation
    # (keeping the same structure but with better error handling)

    def load_graph(self, graph_path):
        """Load NetworkX graph and optionally convert to cuGraph"""
        # Load NetworkX graph
        if graph_path.endswith('.pkl'):
            with open(graph_path, 'rb') as f:
                self.nx_graph = pickle.load(f)
        else:
            raise ValueError(f"Unsupported file format: {graph_path}")

        print(
            f"Loaded NetworkX graph: {self.nx_graph.number_of_nodes()} nodes, {self.nx_graph.number_of_edges()} edges")

        # Convert to cuGraph if GPU enabled
        if self.use_gpu:
            self._convert_to_cugraph()

        return self.nx_graph

    def shortest_path_length(self, source, target):
        """GPU-accelerated shortest path length with robust node mapping"""
        if self.use_gpu and self.cu_graph:
            try:
                # Map NetworkX node IDs to cuGraph node IDs
                cu_source = self._map_nx_to_cu_node(source)
                cu_target = self._map_nx_to_cu_node(target)

                if cu_source is None or cu_target is None:
                    raise ValueError(f"Node mapping failed: {source}→{cu_source}, {target}→{cu_target}")

                # Use cuGraph SSSP
                distances = cugraph.sssp(self.cu_graph, cu_source)
                target_row = distances[distances['vertex'] == cu_target]
                if len(target_row) > 0:
                    return int(target_row['distance'].iloc[0])
            except Exception as e:
                print(f"⚠️ cuGraph SSSP failed: {e}, falling back to NetworkX")

        # Fallback to NetworkX
        try:
            return nx.shortest_path_length(self.nx_graph, source, target)
        except nx.NetworkXNoPath:
            return float('inf')

    def all_pairs_shortest_path_length(self, max_nodes=100):
        """Main APSP method with corrected cuGraph integration"""
        return self.all_pairs_shortest_path_length_adaptive(max_nodes)

    def all_pairs_shortest_path_length_adaptive(self, max_nodes=100):
        """Adaptive implementation with node mapping support"""

        if not (self.use_gpu and self.cu_graph and self.nx_graph.number_of_nodes() <= max_nodes):
            return dict(nx.all_pairs_shortest_path_length(self.nx_graph))

        try:
            num_nodes = self.nx_graph.number_of_nodes()
            print(f"🚀 Adaptive cuGraph APSP for {num_nodes} nodes")

            # Strategy 1: Try Floyd-Warshall (best for small dense graphs)
            if hasattr(cugraph, 'floyd_warshall') and num_nodes <= 50:
                try:
                    print("🔍 Trying Floyd-Warshall...")
                    distances_df = cugraph.floyd_warshall(self.cu_graph)

                    result_dict = {}
                    for _, row in distances_df.to_pandas().iterrows():
                        cu_source = row['source'] if 'source' in row else row[0]
                        cu_dest = row['destination'] if 'destination' in row else row[1]
                        distance = row['distance'] if 'distance' in row else row[2]

                        # Map back to NetworkX node IDs
                        nx_source = self._map_cu_to_nx_node(cu_source)
                        nx_dest = self._map_cu_to_nx_node(cu_dest)

                        if nx_source is not None and nx_dest is not None:
                            if nx_source not in result_dict:
                                result_dict[nx_source] = {}
                            result_dict[nx_source][nx_dest] = distance

                    print("✅ Floyd-Warshall successful")
                    return result_dict

                except Exception as e:
                    print(f"Floyd-Warshall failed: {e}")

            # Strategy 2: Multiple SSSP calls (works for most cuGraph versions)
            if hasattr(cugraph, 'sssp'):
                print("🔍 Using multiple SSSP approach...")
                return self._apsp_using_multiple_sssp()

            else:
                print("❌ No suitable cuGraph methods available")

        except Exception as e:
            print(f"❌ All cuGraph methods failed: {e}")

        # Final fallback to NetworkX
        print("🔄 Using NetworkX fallback")
        return dict(nx.all_pairs_shortest_path_length(self.nx_graph))

    # Add the other methods from the previous implementation...
    # (I'll include the key ones with the same fixes)

    def _apsp_using_multiple_sssp(self):
        """Alternative APSP implementation using multiple SSSP calls with node mapping - COMPLETE FIX"""
        try:
            nodes = list(self.nx_graph.nodes())
            result_dict = {}

            print(f"🔄 Running SSSP for {len(nodes)} source nodes...")
            print(f"Node mapping active: {self.needs_mapping}")

            for i, source_node in enumerate(nodes):
                try:
                    # Map NetworkX node to cuGraph node
                    cu_source = self._map_nx_to_cu_node(source_node)
                    if cu_source is None:
                        raise ValueError(f"Cannot map node {source_node}")

                    # Use cuGraph SSSP
                    distances_df = cugraph.sssp(self.cu_graph, cu_source)

                    # FIXED: Convert to pandas before iterating
                    distances_pandas = distances_df.to_pandas()

                    # Convert to dictionary format with NetworkX node IDs
                    source_distances = {}
                    for _, row in distances_pandas.iterrows():
                        cu_dest = row['vertex']
                        dist = row['distance']

                        # Map back to NetworkX node ID
                        nx_dest = self._map_cu_to_nx_node(cu_dest)
                        if nx_dest is not None:
                            source_distances[nx_dest] = dist

                    result_dict[source_node] = source_distances

                    # Progress reporting
                    if (i + 1) % 10 == 0:
                        print(f"   Completed {i + 1}/{len(nodes)} sources...")

                except Exception as sssp_error:
                    print(f"   ❌ SSSP failed for source {source_node}: {sssp_error}")
                    # Fallback to NetworkX for this source
                    try:
                        distances = nx.single_source_shortest_path_length(self.nx_graph, source_node)
                        result_dict[source_node] = distances
                    except:
                        continue

            print(f"✅ Multiple SSSP completed: {len(result_dict)} sources processed")
            return result_dict

        except Exception as e:
            print(f"❌ Multiple SSSP approach failed: {e}")
            raise

    def _gpu_expand_subgraph_nodes(self, center_node_ids, hop_distance):
        """GPU-accelerated node expansion using cuGraph SSSP with node mapping - FIXED"""

        subgraph_nodes = set(center_node_ids)

        for center_node in center_node_ids:
            try:
                # Map NetworkX node to cuGraph node
                cu_center = self._map_nx_to_cu_node(center_node)
                if cu_center is None:
                    print(f"⚠️ Cannot map center node {center_node}, using NetworkX fallback")
                    fallback_nodes = self._cpu_expand_single_node(center_node, hop_distance)
                    subgraph_nodes.update(fallback_nodes)
                    continue

                # Use cuGraph SSSP to find all nodes within hop_distance
                distances = cugraph.sssp(self.cu_graph, cu_center)

                # Filter nodes within hop distance and map back to NetworkX IDs
                # FIXED: Convert to pandas before processing
                distances_pandas = distances.to_pandas()
                nearby_cu_nodes = distances_pandas[distances_pandas['distance'] <= hop_distance]['vertex']

                for cu_node in nearby_cu_nodes:
                    nx_node = self._map_cu_to_nx_node(cu_node)
                    if nx_node is not None:
                        subgraph_nodes.add(nx_node)

                print(f"✅ GPU SSSP for node {center_node} found {len(nearby_cu_nodes)} nearby nodes")

            except Exception as e:
                print(f"❌ GPU SSSP failed for node {center_node}: {e}")
                # Fallback to NetworkX for this node
                fallback_nodes = self._cpu_expand_single_node(center_node, hop_distance)
                subgraph_nodes.update(fallback_nodes)

        return subgraph_nodes

    # Add other methods as needed...
    def _cpu_expand_single_node(self, center_node, hop_distance):
        """Expand single node using NetworkX shortest path"""
        try:
            distances = nx.single_source_shortest_path_length(
                self.nx_graph, center_node, cutoff=hop_distance
            )
            return set(distances.keys())
        except nx.NetworkXError:
            print(f"Warning: Node {center_node} not found in graph")
            return {center_node}

    def _cpu_expand_subgraph_nodes(self, center_node_ids, hop_distance):
        """CPU-based node expansion using NetworkX"""
        subgraph_nodes = set(center_node_ids)
        for center_node in center_node_ids:
            node_expansion = self._cpu_expand_single_node(center_node, hop_distance)
            subgraph_nodes.update(node_expansion)
        return subgraph_nodes

    def _apsp_using_bfs(self):
        """APSP using BFS (for unweighted graphs) - COMPLETE FIX"""
        try:
            nodes = list(self.nx_graph.nodes())
            result_dict = {}

            print(f"🔄 Using BFS approach for {len(nodes)} nodes...")

            for source_node in nodes:
                try:
                    # Map NetworkX node to cuGraph node
                    cu_source = self._map_nx_to_cu_node(source_node)
                    if cu_source is None:
                        print(f"⚠️ Cannot map node {source_node} for BFS")
                        continue

                    # Use BFS from each source
                    bfs_result = cugraph.bfs(self.cu_graph, cu_source)

                    # FIXED: Convert to pandas before iterating
                    bfs_pandas = bfs_result.to_pandas()

                    # Convert BFS distances to dictionary
                    source_distances = {}
                    for _, row in bfs_pandas.iterrows():
                        cu_dest = row['vertex']
                        dist = row['distance']

                        # Map back to NetworkX node ID
                        nx_dest = self._map_cu_to_nx_node(cu_dest)
                        if nx_dest is not None:
                            source_distances[nx_dest] = dist

                    result_dict[source_node] = source_distances

                except Exception as bfs_error:
                    print(f"BFS failed for source {source_node}: {bfs_error}")
                    # Fallback to NetworkX
                    try:
                        distances = nx.single_source_shortest_path_length(self.nx_graph, source_node)
                        result_dict[source_node] = distances
                    except:
                        continue

            print(f"✅ BFS approach completed: {len(result_dict)} sources")
            return result_dict

        except Exception as e:
            print(f"BFS approach failed: {e}")
            raise

    def create_subgraph_from_centers_optimized(self, center_node_ids, hop_distance=2, buffer_distance=None,
                                               nodes_gdf=None):
        """GPU-optimized subgraph creation with empty graph handling - FIXED"""

        print(f"🔍 Creating subgraph from {len(center_node_ids)} center nodes")
        print(f"Hop distance: {hop_distance}")
        print(f"GPU acceleration: {self.use_gpu}")

        # Ensure we have a graph to work with
        if self.nx_graph is None:
            raise ValueError("No graph loaded. Call load_graph() first or set nx_graph directly.")

        subgraph_nodes = set(center_node_ids)

        # Method 1: GPU-accelerated hop-based expansion
        if self.use_gpu and self.cu_graph:
            try:
                subgraph_nodes = self._gpu_expand_subgraph_nodes(center_node_ids, hop_distance)
                print(f"✅ GPU expansion found {len(subgraph_nodes)} nodes")
            except Exception as e:
                print(f"❌ GPU expansion failed: {e}, falling back to NetworkX")
                subgraph_nodes = self._cpu_expand_subgraph_nodes(center_node_ids, hop_distance)
        else:
            subgraph_nodes = self._cpu_expand_subgraph_nodes(center_node_ids, hop_distance)

        # Create NetworkX subgraph
        subgraph_nx = self.nx_graph.subgraph(list(subgraph_nodes)).copy()

        print(f"📊 Final subgraph: {subgraph_nx.number_of_nodes()} nodes, {subgraph_nx.number_of_edges()} edges")

        # FIXED: Handle empty subgraph case
        if subgraph_nx.number_of_edges() == 0:
            print("⚠️ WARNING: Subgraph has no edges!")
            print("   This may happen when:")
            print("   - Center nodes are isolated")
            print("   - Hop distance is too small")
            print("   - Graph connectivity is poor")
            print("💡 Suggestion: Try increasing hop_distance or check graph connectivity")

        return {
            'subgraph': subgraph_nx,
            'all_nodes': list(subgraph_nodes),
            'center_nodes': center_node_ids,
            'hop_distance': hop_distance,
            'buffer_distance': buffer_distance,
            'gpu_acceleration_used': self.use_gpu,
            'node_mapping_used': getattr(self, 'needs_mapping', False),
            'total_nodes': len(subgraph_nodes),
            'total_edges': subgraph_nx.number_of_edges(),
            'has_edges': subgraph_nx.number_of_edges() > 0  # Flag for downstream processing
        }

    def find_simple_paths_diverse(self, source, target, max_paths=5, diversity_method="similarity"):
        """
        Find diverse paths using generator approach

        Args:
            diversity_method: "similarity" or "length"
        """
        try:
            if diversity_method == "similarity":
                paths = find_diverse_paths_generator(
                    self.nx_graph, source, target, max_paths=max_paths
                )
            else:  # length diversity
                paths = find_paths_by_length_diversity(
                    self.nx_graph, source, target, max_paths=max_paths
                )

            # Analyze diversity of results
            if paths:
                diversity_analysis = analyze_path_diversity(paths)
                return paths, diversity_analysis
            else:
                return [], {"diversity_score": 0.0, "analysis": "No paths found"}

        except Exception as e:
            print(f"❌ Error in diverse path finding: {e}")
            return [], {"error": str(e)}

    def find_simple_paths(self, source, target, cutoff=None, max_paths=5):
        """Find simple paths (NetworkX only for now)"""
        try:
            paths = list(nx.all_simple_paths(self.nx_graph, source, target, cutoff=cutoff))
            return paths[:max_paths] if max_paths else paths
        except nx.NetworkXNoPath:
            return []

    def create_subgraph_from_centers_optimized(self, center_node_ids, hop_distance=2, buffer_distance=None,
                                               nodes_gdf=None):
        """
        GPU-optimized subgraph creation from center nodes.

        Args:
            center_node_ids: List of center node IDs
            hop_distance: Maximum hop distance from center nodes
            buffer_distance: Buffer distance in meters (optional)
            nodes_gdf: GeoDataFrame for spatial buffer (optional)

        Returns:
            Dictionary with subgraph information compatible with original format
        """

        print(f"🔍 Creating subgraph from {len(center_node_ids)} center nodes")
        print(f"Hop distance: {hop_distance}")
        print(f"GPU acceleration: {self.use_gpu}")

        subgraph_nodes = set(center_node_ids)

        # Method 1: GPU-accelerated hop-based expansion
        if self.use_gpu and self.cu_graph:
            try:
                subgraph_nodes = self._gpu_expand_subgraph_nodes(center_node_ids, hop_distance)
                print(f"✅ GPU expansion found {len(subgraph_nodes)} nodes")
            except Exception as e:
                print(f"❌ GPU expansion failed: {e}, falling back to NetworkX")
                subgraph_nodes = self._cpu_expand_subgraph_nodes(center_node_ids, hop_distance)
        else:
            subgraph_nodes = self._cpu_expand_subgraph_nodes(center_node_ids, hop_distance)

        # Method 2: Add spatial buffer if nodes_gdf provided
        # if nodes_gdf is not None and buffer_distance > 0:
        #     spatial_nodes = self._add_spatial_buffer_nodes(center_node_ids, buffer_distance, nodes_gdf)
        #     subgraph_nodes.update(spatial_nodes)
        #     print(f"📍 Added {len(spatial_nodes)} nodes from spatial buffer")

        # Create NetworkX subgraph
        subgraph_nx = self.nx_graph.subgraph(list(subgraph_nodes)).copy()

        print(f"📊 Final subgraph: {subgraph_nx.number_of_nodes()} nodes, {subgraph_nx.number_of_edges()} edges")

        # Return in original format
        return {
            'subgraph': subgraph_nx,
            'all_nodes': list(subgraph_nodes),
            'center_nodes': center_node_ids,
            'hop_distance': hop_distance,
            # 'buffer_distance': buffer_distance,
            'gpu_acceleration_used': self.use_gpu,
            'total_nodes': len(subgraph_nodes),
            'total_edges': subgraph_nx.number_of_edges()
        }


def get_node_info_from_gdf(node_id, nodes_gdf):
    """Get node information from GeoDataFrame"""
    try:
        node_row = nodes_gdf[nodes_gdf.id == node_id]
        if len(node_row) > 0:
            row = node_row.iloc[0]
            return {
                'name': str(row.get('name', 'Unnamed')),
                'category': row.get('category', row.get('type', None)),
                'geometry_type': row.geometry.geom_type if hasattr(row, 'geometry') else 'Unknown'
            }
    except Exception as e:
        print(f"Error getting node info for {node_id}: {e}")
    return None


def find_all_paths_between_nodes(graph_processor, source_id, target_id, nodes_gdf,
                                 max_path_length=None, max_paths=5, category=None):
    """
    Find all possible paths between two nodes with optional category filtering.
    Optimized with cuGraph where possible.
    """
    print(f"Finding paths from {source_id} to {target_id}")
    if category:
        print(f"Filtering for paths that pass through category: {category}")

    try:
        # Set reasonable cutoff if not specified
        if max_path_length is None:
            shortest_path_length = graph_processor.shortest_path_length(source_id, target_id)
            if shortest_path_length == float('inf'):
                return {'total_paths_found': 0, 'paths': []}
            max_path_length = min(shortest_path_length + 10, 20)

        # Get all simple paths (NetworkX only for now)
        all_paths = graph_processor.find_simple_paths(source_id, target_id, max_path_length, max_paths * 2)

        if category:
            # Filter paths that pass through nodes with the specified category
            filtered_paths = []
            for path in all_paths:
                path_has_category = False
                for node_id in path:
                    node_info = get_node_info_from_gdf(node_id, nodes_gdf)
                    if node_info and node_info.get('category') == category:
                        path_has_category = True
                        break

                if path_has_category:
                    filtered_paths.append(path)

            all_paths = filtered_paths
            print(f"Found {len(all_paths)} paths passing through '{category}' category")

        # Sort by path length and limit
        all_paths.sort(key=len)
        if max_paths:
            all_paths = all_paths[:max_paths]

        paths_info = {
            'total_paths_found': len(all_paths),
            'paths': []
        }

        for i, path in enumerate(all_paths):
            path_info = {
                'path_id': i,
                'nodes': path,
                'length': len(path) - 1,
                'has_required_category': category is not None
            }

            if category:
                category_nodes = []
                for node_id in path:
                    node_info = get_node_info_from_gdf(node_id, nodes_gdf)
                    if node_info and node_info.get('category') == category:
                        category_nodes.append(node_id)
                path_info['category_nodes'] = category_nodes

            paths_info['paths'].append(path_info)

        return paths_info

    except Exception as e:
        print(f"Error finding paths: {e}")
        return {'total_paths_found': 0, 'paths': [], 'error': str(e)}


def find_all_paths_between_nodes_diverse(graph_processor, source_id, target_id, nodes_gdf,
                                         max_paths=5, category=None, diversity_method="similarity"):
    """
    Find diverse paths between nodes with immediate stopping at max_paths
    """
    print(f"Finding {max_paths} diverse paths from {source_id} to {target_id}")
    if category:
        print(f"Filtering for paths that pass through category: {category}")

    try:
        # Use diverse path finding
        all_paths, diversity_analysis = graph_processor.find_simple_paths_diverse(
            source_id, target_id, max_paths=max_paths, diversity_method=diversity_method
        )

        if category:
            # Filter paths that pass through nodes with the specified category
            filtered_paths = []
            for path in all_paths:
                path_has_category = False
                for node_id in path:
                    node_info = get_node_info_from_gdf(node_id, nodes_gdf)
                    if node_info and node_info.get('category') == category:
                        path_has_category = True
                        break

                if path_has_category:
                    filtered_paths.append(path)

            all_paths = filtered_paths
            print(f"Found {len(all_paths)} diverse paths passing through '{category}' category")

        # Sort by path length for consistency
        all_paths.sort(key=len)

        paths_info = {
            'total_paths_found': len(all_paths),
            'diversity_analysis': diversity_analysis,
            'paths': []
        }

        for i, path in enumerate(all_paths):
            path_info = {
                'path_id': i,
                'nodes': path,
                'length': len(path) - 1,
                'has_required_category': category is not None
            }

            if category:
                category_nodes = []
                for node_id in path:
                    node_info = get_node_info_from_gdf(node_id, nodes_gdf)
                    if node_info and node_info.get('category') == category:
                        category_nodes.append(node_id)
                path_info['category_nodes'] = category_nodes

            paths_info['paths'].append(path_info)

        return paths_info

    except Exception as e:
        print(f"Error finding diverse paths: {e}")
        return {'total_paths_found': 0, 'paths': [], 'error': str(e)}


import networkx as nx
from itertools import islice


def find_simple_paths_generator(graph, source, target, cutoff=None, max_paths=5):
    """
    Generator-based path finding that stops immediately when reaching max_paths
    """
    print(f"🔍 Finding up to {max_paths} paths from {source} to {target}")

    try:
        # Use NetworkX generator - this is lazy and doesn't compute all paths upfront
        path_generator = nx.all_simple_paths(graph, source, target, cutoff=cutoff)

        # Use islice to take only the first max_paths paths
        # This stops the generator immediately when we reach the limit
        limited_paths = list(islice(path_generator, max_paths))

        print(f"✅ Found {len(limited_paths)} paths (requested {max_paths})")
        return limited_paths

    except nx.NetworkXNoPath:
        print("❌ No path exists between source and target")
        return []
    except Exception as e:
        print(f"❌ Error during path finding: {e}")
        return []


def calculate_path_similarity(path1, path2):
    """
    Calculate similarity between two paths based on shared nodes
    Returns similarity ratio (0.0 = completely different, 1.0 = identical)
    """
    set1 = set(path1)
    set2 = set(path2)

    # Jaccard similarity: intersection / union
    intersection = len(set1 & set2)
    union = len(set1 | set2)

    return intersection / union if union > 0 else 0.0


def is_path_diverse(new_path, existing_paths, min_diversity_threshold=0.3):
    """
    Check if a new path is sufficiently diverse from existing paths

    Args:
        new_path: The path to check
        existing_paths: List of already selected paths
        min_diversity_threshold: Minimum required diversity (lower = more similar allowed)

    Returns:
        True if the path is diverse enough, False otherwise
    """
    if not existing_paths:
        return True

    # Check similarity with all existing paths
    for existing_path in existing_paths:
        similarity = calculate_path_similarity(new_path, existing_path)

        # If too similar to any existing path, reject it
        if similarity > (1.0 - min_diversity_threshold):
            return False

    return True


def find_diverse_paths_generator(graph, source, target, max_paths=5, min_diversity=0.3, max_search_paths=50):
    """
    Find diverse paths using generator approach with diversity filtering

    Args:
        graph: NetworkX graph
        source: Source node
        target: Target node
        max_paths: Maximum number of paths to return
        min_diversity: Minimum diversity threshold (0.0-1.0)
        max_search_paths: Maximum paths to examine before giving up
    """
    print(f"🔍 Finding up to {max_paths} diverse paths from {source} to {target}")
    print(f"📊 Diversity threshold: {min_diversity}, max search: {max_search_paths}")

    try:
        # Get shortest path for intelligent cutoff
        shortest_path = nx.shortest_path(graph, source, target)
        shortest_length = len(shortest_path) - 1
        cutoff = min(shortest_length + 5, 12)  # Reasonable cutoff

        print(f"📏 Shortest path: {shortest_length} edges, using cutoff: {cutoff}")

        # Generate paths and filter for diversity
        path_generator = nx.all_simple_paths(graph, source, target, cutoff=cutoff)
        diverse_paths = []
        paths_examined = 0

        for path in path_generator:
            paths_examined += 1

            # Check if this path is diverse enough
            if is_path_diverse(path, diverse_paths, min_diversity):
                diverse_paths.append(path)
                print(f"✅ Path {len(diverse_paths)}: length {len(path) - 1}, diversity OK")

                # Stop when we have enough diverse paths
                if len(diverse_paths) >= max_paths:
                    print(f"🎯 Found {max_paths} diverse paths, stopping search")
                    break
            else:
                # Path was too similar, continue searching
                if paths_examined % 10 == 0:
                    print(f"🔄 Examined {paths_examined} paths, found {len(diverse_paths)} diverse ones...")

            # Safety break to avoid infinite search
            if paths_examined >= max_search_paths:
                print(f"⏹️ Reached search limit of {max_search_paths} paths")
                break

        print(f"📊 Final result: {len(diverse_paths)} diverse paths from {paths_examined} examined")
        return diverse_paths

    except nx.NetworkXNoPath:
        print("❌ No path exists between source and target")
        return []
    except Exception as e:
        print(f"❌ Error during path finding: {e}")
        return []


def find_paths_by_length_diversity(graph, source, target, max_paths=5):
    """
    Alternative approach: Find paths of different lengths for guaranteed diversity
    """
    print(f"🔍 Finding {max_paths} paths with length diversity")

    try:
        shortest_path = nx.shortest_path(graph, source, target)
        shortest_length = len(shortest_path) - 1

        diverse_paths = []
        max_search_length = min(shortest_length + 8, 15)

        # Search for paths of increasing length
        for target_length in range(shortest_length, max_search_length + 1):
            if len(diverse_paths) >= max_paths:
                break

            print(f"🔍 Searching for paths of length {target_length}...")

            # Find paths of specific length
            path_generator = nx.all_simple_paths(graph, source, target, cutoff=target_length)
            paths_at_length = []

            for path in path_generator:
                if len(path) - 1 == target_length:  # Exact length match
                    paths_at_length.append(path)
                    if len(paths_at_length) >= 3:  # Max 3 paths per length
                        break

            # Add diverse paths from this length
            for path in paths_at_length:
                if is_path_diverse(path, diverse_paths, min_diversity_threshold=0.2):
                    diverse_paths.append(path)
                    print(f"✅ Added path of length {len(path) - 1}")

                    if len(diverse_paths) >= max_paths:
                        break

        print(f"📊 Found {len(diverse_paths)} paths with length diversity")
        return diverse_paths

    except Exception as e:
        print(f"❌ Error in length-based diversity search: {e}")
        return []


def analyze_path_diversity(paths):
    """
    Analyze and report diversity metrics of the found paths
    """
    if len(paths) < 2:
        return {"diversity_score": 1.0, "analysis": "Insufficient paths for diversity analysis"}

    similarities = []
    path_lengths = [len(p) - 1 for p in paths]

    # Calculate pairwise similarities
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            similarity = calculate_path_similarity(paths[i], paths[j])
            similarities.append(similarity)

    avg_similarity = sum(similarities) / len(similarities)
    diversity_score = 1.0 - avg_similarity

    analysis = {
        "diversity_score": diversity_score,
        "average_similarity": avg_similarity,
        "path_lengths": path_lengths,
        "length_variance": max(path_lengths) - min(path_lengths),
        "total_paths": len(paths),
        "pairwise_similarities": similarities
    }

    print(f"📊 Diversity Analysis:")
    print(f"   Diversity Score: {diversity_score:.3f} (higher is more diverse)")
    print(f"   Average Similarity: {avg_similarity:.3f}")
    print(f"   Path Lengths: {path_lengths}")
    print(f"   Length Range: {min(path_lengths)}-{max(path_lengths)}")

    return analysis


# Enhanced version with more control
def find_distant_nodes_and_route_enhanced(subgraph_result, nodes_gdf, here_api_key,
                                          min_hop_distance=3, profile="pedestrian",
                                          max_candidates=50, distance_method='geodesic',
                                          auto_switch_distance_km=8.0, driving_profile="car",
                                          category=None, filter='similarity', diversity_config=None, max_paths=5,
                                          language='en'):
    """
    Enhanced version with configurable diversity settings

    Args:
        diversity_config: Dict with keys:
            - method: "similarity" or "length"
            - max_paths: int (default 5)
            - min_diversity: float (default 0.3)
            - max_search_paths: int (default 50)
    """

    # Set default diversity configuration
    if diversity_config is None:
        diversity_config = {
            'method': filter,
            'max_paths': 5,
            'min_diversity': 0.3,
            'max_search_paths': 50
        }

    diversity_method = diversity_config.get('method', 'similarity')
    max_paths = diversity_config.get('max_paths', 5)

    print(f"🎯 Using diversity configuration: {diversity_config}")

    return find_distant_nodes_and_route_optimized(
        subgraph_result, nodes_gdf, here_api_key,
        min_hop_distance=min_hop_distance, profile=profile,
        alternatives=True, max_candidates=max_candidates,
        distance_method=distance_method,
        auto_switch_distance_km=auto_switch_distance_km,
        driving_profile=driving_profile,
        category=category,
        diversity_method=diversity_method,
        max_paths=max_paths,
        language=language
    )


def find_distant_nodes_and_route_optimized(subgraph_result, nodes_gdf, here_api_key,
                                           min_hop_distance=3, profile="pedestrian",
                                           alternatives=True, max_candidates=50,
                                           distance_method='geodesic',
                                           auto_switch_distance_km=8.0,
                                           driving_profile="car",
                                           category=None,
                                           diversity_method="similarity",
                                           max_paths=5, language='en'):
    """
    Optimized version using GPU acceleration and diverse path finding
    """

    if not subgraph_result.get('has_edges', True):
        print("❌ Subgraph has no edges - cannot find paths")
        print("💡 Suggestions:")
        print("   1. Increase hop_distance when creating subgraph")
        print("   2. Choose different center nodes")
        print("   3. Check graph connectivity")
        return None

    if subgraph_result.get('total_edges', 0) == 0:
        print("❌ Subgraph has no edges - cannot find paths")
        return None

        # Create graph processor with validation
    graph_processor = GraphProcessor(use_gpu=CUGRAPH_AVAILABLE)
    graph_processor.nx_graph = subgraph_result['subgraph']

    # Only convert to cuGraph if there are edges
    if graph_processor.nx_graph.number_of_edges() > 0 and graph_processor.use_gpu:
        graph_processor._convert_to_cugraph()
    else:
        print("🔄 Using NetworkX only (no edges for cuGraph)")
        graph_processor.use_gpu = False

    all_nodes = subgraph_result['all_nodes']

    print(f"=== Finding farthest nodes in subgraph ===")
    print(f"Subgraph nodes: {len(all_nodes)}")
    print(f"Subgraph edges: {graph_processor.nx_graph.number_of_edges()}")
    print(f"Min hop distance: {min_hop_distance}")
    print(f"Diversity method: {diversity_method}")
    print(f"Max paths: {max_paths}")
    if category:
        print(f"Required category in path: {category}")

    # Helper functions (simplified)
    def get_coordinates(geom):
        if geom.geom_type == 'Point':
            return geom.x, geom.y
        else:
            centroid = geom.centroid
            return centroid.x, centroid.y

    def is_valid_pair(node1, node2):
        """Simplified validation - at least one node should be POI"""
        try:
            node1_row = nodes_gdf[nodes_gdf.id == node1]
            node2_row = nodes_gdf[nodes_gdf.id == node2]

            if len(node1_row) == 0 or len(node2_row) == 0:
                return False

            node1_geom = node1_row.iloc[0].geometry.geom_type
            node2_geom = node2_row.iloc[0].geometry.geom_type

            return (node1_geom in ['Point', 'Polygon']) or (node2_geom in ['Point', 'Polygon'])
        except:
            return False

    def has_category_in_path_fast(node1, node2, max_paths):
        """Fast check if path has required category using diverse path finding"""
        if category is None:
            return True

        try:
            # Use the new diverse path finding with early stopping
            paths, _ = graph_processor.find_simple_paths_diverse(
                node1, node2, max_paths=max_paths, diversity_method="similarity"
            )

            # Check if any path contains the required category
            for path in paths:
                for node_id in path:
                    node_info = get_node_info_from_gdf(node_id, nodes_gdf)
                    if node_info and node_info.get('category') == category:
                        return True
            return False
        except:
            return False

    # Ensure nodes_gdf is in WGS84
    if nodes_gdf.crs and nodes_gdf.crs != "EPSG:4326":
        nodes_gdf_wgs84 = nodes_gdf.to_crs("EPSG:4326")
    else:
        nodes_gdf_wgs84 = nodes_gdf.copy()

    # Pre-calculate coordinates
    print("Pre-calculating coordinates for subgraph nodes...")
    node_coordinates = {}
    valid_nodes = []

    for node_id in all_nodes:
        node_row = nodes_gdf_wgs84[nodes_gdf_wgs84.id == node_id]
        if len(node_row) > 0:
            try:
                geom = node_row.geometry.iloc[0]
                lon, lat = get_coordinates(geom)
                node_coordinates[node_id] = (lat, lon)
                valid_nodes.append(node_id)
            except Exception as e:
                print(f"Warning: Could not extract coordinates for node {node_id}: {e}")

    print(f"Valid nodes with coordinates: {len(valid_nodes)}")

    if len(valid_nodes) < 2:
        raise ValueError("Not enough valid nodes with coordinates found")

    # Find candidate pairs - OPTIMIZED with GPU
    print("Finding candidate pairs with sufficient hop distance...")
    candidate_pairs = []

    if len(valid_nodes) <= 100:
        # Use GPU-accelerated all pairs shortest path
        print("Using GPU-accelerated all pairs shortest path...")
        all_pairs_distances = graph_processor.all_pairs_shortest_path_length(max_nodes=100)

        for node1, node2 in combinations(valid_nodes, 2):
            if (node1 in all_pairs_distances and
                    node2 in all_pairs_distances[node1] and
                    all_pairs_distances[node1][node2] >= min_hop_distance):

                if is_valid_pair(node1, node2) and has_category_in_path_fast(node1, node2, max_paths):
                    candidate_pairs.append((node1, node2, all_pairs_distances[node1][node2]))

                    # Print progress for category filtering
                    if category and len(candidate_pairs) % 10 == 0:
                        print(f"Found {len(candidate_pairs)} pairs with category '{category}'...")
    else:
        # Sample random pairs for larger graphs
        print(f"Large graph detected, sampling {max_candidates} random pairs...")
        sampled_pairs = random.sample(list(combinations(valid_nodes, 2)),
                                      min(max_candidates, len(valid_nodes) * (len(valid_nodes) - 1) // 2))

        for node1, node2 in sampled_pairs:
            hop_distance = graph_processor.shortest_path_length(node1, node2)
            if hop_distance >= 3 and hop_distance != float('inf'):
                if is_valid_pair(node1, node2) and has_category_in_path_fast(node1, node2, max_paths):
                    candidate_pairs.append((node1, node2, hop_distance))

    print(f"Found {len(candidate_pairs)} candidate pairs meeting requirements")

    if not candidate_pairs:
        print("❌ No candidate pairs found meeting all criteria")
        if category:
            print(f"💡 Consider relaxing category requirement '{category}' or reducing min_hop_distance")
        return None

    # Calculate geometric distances
    print("Calculating geometric distances...")
    distance_pairs = []

    for node1, node2, hop_dist in candidate_pairs:
        coord1 = node_coordinates[node1]
        coord2 = node_coordinates[node2]

        try:
            if distance_method == 'geodesic':
                geometric_distance = geodesic(coord1, coord2).kilometers
            else:
                lat1, lon1 = coord1
                lat2, lon2 = coord2
                geometric_distance = np.sqrt((lat2 - lat1) ** 2 + (lon2 - lon1) ** 2) * 111

            score = geometric_distance * (1 + hop_dist * 0.1)

            distance_pairs.append({
                'node1': node1,
                'node2': node2,
                'hop_distance': hop_dist,
                'geometric_distance_km': geometric_distance,
                'score': score
            })
        except Exception as e:
            print(f"Error calculating distance between {node1} and {node2}: {e}")
            continue

    if not distance_pairs:
        raise ValueError("Could not calculate distances for any candidate pairs")

    # Select the farthest pair
    distance_pairs.sort(key=lambda x: x['score'], reverse=True)
    selected_pair = distance_pairs[0]
    source_id = selected_pair['node1']
    target_id = selected_pair['node2']

    print(f"\nSelected farthest pair: {source_id} -> {target_id}")
    print(f"Geometric distance: {selected_pair['geometric_distance_km']:.2f} km")
    print(f"Hop distance: {selected_pair['hop_distance']}")

    # Get diverse paths using the new function
    print(f"\nFinding {max_paths} diverse paths...")
    all_paths_info = find_all_paths_between_nodes_diverse(
        graph_processor, source_id, target_id, nodes_gdf,
        max_paths=max_paths, category=category, diversity_method=diversity_method
    )

    # Print diversity analysis if available
    if 'diversity_analysis' in all_paths_info:
        diversity_analysis = all_paths_info['diversity_analysis']
        if 'diversity_score' in diversity_analysis:
            print(f"📊 Path diversity score: {diversity_analysis['diversity_score']:.3f}")
        if 'path_lengths' in diversity_analysis:
            print(f"📏 Path lengths: {diversity_analysis['path_lengths']}")

    # Get shortest path
    try:
        shortest_path = nx.shortest_path(graph_processor.nx_graph, source=source_id, target=target_id)
    except nx.NetworkXNoPath:
        print("❌ No path found between selected nodes")
        return None
    # Check if shortest path is already in the diverse paths and add if not
    if 'paths' in all_paths_info:
        existing_paths = [path_info['nodes'] for path_info in all_paths_info['paths']]
        shortest_path_exists = False

        # Check if shortest path already exists in the diverse paths
        for existing_path in existing_paths:
            if existing_path == shortest_path:
                shortest_path_exists = True
                print("✅ Shortest path already included in diverse paths")
                break

        # Add shortest path if it's not already included
        if not shortest_path_exists:
            print("➕ Adding shortest path to diverse paths collection")

            # Create path info structure consistent with diverse path format
            shortest_path_info = {
                'path_id': len(all_paths_info['paths']),  # Assign next available ID
                'nodes': shortest_path,
                'length': len(shortest_path) - 1,
                'has_required_category': category is not None,
                'is_shortest_path': True  # Mark this as the shortest path
            }

            # Add category nodes if category filtering was applied
            if category:
                category_nodes = []
                for node_id in shortest_path:
                    node_info = get_node_info_from_gdf(node_id, nodes_gdf)
                    if node_info and node_info.get('category') == category:
                        category_nodes.append(node_id)
                shortest_path_info['category_nodes'] = category_nodes

            # Add to paths list
            all_paths_info['paths'].append(shortest_path_info)

            # Update total count
            all_paths_info['total_paths_found'] = len(all_paths_info['paths'])

            # Recalculate diversity analysis if it exists
            if 'diversity_analysis' in all_paths_info:
                try:
                    updated_paths = [path_info['nodes'] for path_info in all_paths_info['paths']]
                    updated_diversity = analyze_path_diversity(updated_paths)
                    all_paths_info['diversity_analysis'] = updated_diversity

                    print(f"📊 Updated diversity score: {updated_diversity.get('diversity_score', 'N/A'):.3f}")
                    print(f"📏 Updated path lengths: {updated_diversity.get('path_lengths', [])}")

                except Exception as div_error:
                    print(f"⚠️ Could not recalculate diversity analysis: {div_error}")

            print(f"✅ Total paths now: {all_paths_info['total_paths_found']}")

    else:
        print("⚠️ No paths structure found in all_paths_info")

    # Get coordinates for HERE API
    source_coord = node_coordinates[source_id]
    target_coord = node_coordinates[target_id]
    source_lat, source_lon = source_coord
    target_lat, target_lon = target_coord

    # Determine routing profile
    selected_distance_km = selected_pair['geometric_distance_km']
    profile_switched = False
    routing_profile = profile

    if selected_distance_km > auto_switch_distance_km and profile.lower() == "pedestrian":
        routing_profile = driving_profile
        profile_switched = True
        print(f"Switching from '{profile}' to '{driving_profile}' for long distance")

    # Call HERE API (simplified - remove the complex fallback logic)
    print(f"Calling HERE API for route...")
    here_response = get_here_route(
        source_lat, source_lon, target_lat, target_lon, here_api_key, routing_profile, language=language
    )

    # Enhanced return with diversity information
    result = {
        'origin_node': source_id,
        'destination_node': target_id,
        'hop_distance': selected_pair['hop_distance'],
        'geometric_distance_km': selected_pair['geometric_distance_km'],
        'routing_profile_used': routing_profile,
        'profile_switched': profile_switched,
        'routes': here_response,
        'shortest_path': shortest_path,
        'all_paths': all_paths_info,
        'gpu_acceleration_used': graph_processor.use_gpu,
        'diversity_method_used': diversity_method,
        'max_paths_requested': max_paths,
        'paths_found': all_paths_info.get('total_paths_found', 0)
    }

    # Add category filtering statistics if category was specified
    if category:
        result['category_filter'] = {
            'required_category': category,
            'candidate_pairs_with_category': len(candidate_pairs),
            'category_filtering_applied': True
        }

    # Add diversity metrics to result
    if 'diversity_analysis' in all_paths_info:
        result['diversity_metrics'] = all_paths_info['diversity_analysis']

    print(f"✅ Found {result['paths_found']} diverse paths successfully")

    return result


def get_here_route(source_lat, source_lon, target_lat, target_lon,
                   api_key, profile="pedestrian", alternatives=True, language="en"):
    """
    Get a route between two points using HERE Routing API v8

    Args:
        source_lat: Source latitude
        source_lon: Source longitude
        target_lat: Target latitude
        target_lon: Target longitude
        api_key: HERE API key
        profile: Routing profile ('pedestrian', 'car', 'bicycle', etc.)
        alternatives: Whether to request alternative routes
        language: Language for route instructions ('en', 'zh', 'ja', 'ko', etc.)

    Returns:
        dict: The HERE API response
    """
    # Base URL for HERE Routing API v8
    base_url = "https://router.hereapi.com/v8/routes"

    # Set parameters based on the profile and alternatives
    params = {
        'apiKey': api_key,
        'origin': f"{source_lat},{source_lon}",
        'destination': f"{target_lat},{target_lon}",
        'transportMode': profile,
        'return': 'polyline,actions,instructions,summary',
        'lang': language,  # Use the specified language
        'units': 'metric',
        'spans': 'names',  # Include street names
        'alternatives': 3 if alternatives else 0  # Request up to 3 alternatives if enabled
    }

    # Make request
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()  # Raise exception for 4XX/5XX responses

        print(f"HERE API response status: {response.status_code}")
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error making HERE API request: {e}")
        # Return error information in a structured way
        return {
            'error': True,
            'error_message': str(e),
            'status_code': getattr(response, 'status_code', None),
            'url': base_url
        }


def extract_content_from_brackets(text):
    """Extract content within square brackets"""
    import re
    pattern = r'\[([^\]]*)\]'
    matches = re.findall(pattern, text)
    return matches


def format_triples_as_path_string_enhanced(triples):
    """
    Format triples as a path string with arrows and direction/distance tuples

    Input: List of triple dictionaries from create_path_triple_optimized
    Output: "(Battery Pl, Intersection...) -> (101.0m, 190°(S)) -> (Hugh L Carey Tunl, Intersection...) -> (19.9m, 312°(NW))"
    """

    if not triples:
        return ""

    path_parts = []

    for i, triple in enumerate(triples):
        # Add the triple (location part only)
        source_name = triple.get('source_name', 'Unknown')
        middle_name = triple.get('middle_component', 'Unknown')
        target_name = triple.get('target_name', 'Unknown')

        triple_part = f"({source_name}, {middle_name}, {target_name})"
        path_parts.append(triple_part)

        # Add direction/distance tuple if not the last triple
        if i < len(triples) - 1:
            length = triple.get('length')
            direction = triple.get('direction')
            direction_tuple = format_direction_distance_tuple(length, direction)
            path_parts.append(direction_tuple)

    # Join with arrows
    return " -> ".join(path_parts)


# Keep the original format function for backward compatibility
def format_location_string_flexible_original(source_name, middle_name, target_name, length=None, direction=None):
    """Original string formatter for backward compatibility"""

    # Base location part
    location_part = f"({source_name}, {middle_name}, {target_name})"

    # Build bracket content
    bracket_parts = []

    if length is not None:
        bracket_parts.append(f"{length:.1f}m")

    if direction and str(direction).strip():
        bracket_parts.append(str(direction))

    # Return with or without brackets
    if bracket_parts:
        return f"{location_part}[{', '.join(bracket_parts)}]"
    else:
        return f"{location_part}[]"


# Function to be used in the higher-level path processing
def format_complete_path_with_arrows(triples):
    """
    This function should replace format_triples_as_string_simple in your path processing
    """
    return format_triples_as_path_string_enhanced(triples)


def format_location_string_flexible(source_name, middle_name, target_name, length=None, direction=None):
    """Modified string formatter - returns only the location part for new path format"""

    # Return only the location part without brackets
    return f"({source_name}, {middle_name}, {target_name})"


def format_direction_distance_tuple(length=None, direction=None):
    """Format the direction and distance as a separate tuple"""

    tuple_parts = []

    if length is not None:
        tuple_parts.append(f"{length:.1f}m")

    if direction and str(direction).strip():
        tuple_parts.append(str(direction))

    if tuple_parts:
        return f"({', '.join(tuple_parts)})"
    else:
        return "()"


# Remove redundant functions (these were duplicated or unused):
# - Multiple direction calculation functions (consolidated into one)
# - Redundant coordinate extraction functions
# - Unused geometry processing functions
# - Complex fallback logic in HERE API calls

def parse_single_path_directions_enhanced(path_nodes, nodes_gdf, edges_gdf, graph_processor,
                                          node_id_col='id', edge_source_col='id1', edge_target_col='id2'):
    """
    Enhanced path parsing function optimized for cuGraph integration.

    Args:
        path_nodes: List of node IDs representing the path
        nodes_gdf: GeoDataFrame with node information
        edges_gdf: GeoDataFrame with edge information
        graph_processor: GraphProcessor instance (with GPU support)
        node_id_col: Column name for node IDs in nodes_gdf
        edge_source_col: Column name for source node IDs in edges_gdf
        edge_target_col: Column name for target node IDs in edges_gdf

    Returns:
        Dictionary with comprehensive path analysis including both formatting styles
    """

    if not path_nodes or len(path_nodes) < 2:
        return {
            'path': path_nodes,
            'path_triples': [],
            'formatted_path': '',
            'formatted_path_enhanced': '',
            'total_length_meters': 0,
            'total_direction_changes': 0,
            'path_analysis': {'error': 'Path too short'},
            'success': False,
            'gpu_acceleration_used': False
        }

    print(f"🔍 Parsing enhanced path with {len(path_nodes)} nodes: {path_nodes}")

    try:
        # Use the optimized triple parsing with GPU support
        triple_result = parse_single_path_to_triples_optimized(
            path_nodes, nodes_gdf, edges_gdf, graph_processor,
            node_id_col, edge_source_col, edge_target_col
        )
        print("=================triple====================")
        print(triple_result['formatted_path_enhanced'])
        print("=================triple====================")

        # Enhanced path analysis
        path_analysis = analyze_path_characteristics_optimized(
            path_nodes, triple_result, nodes_gdf, edges_gdf, graph_processor
        )

        # Compile results with both formatting styles
        enhanced_result = {
            'path': path_nodes,
            'path_triples': triple_result.get('path_triples', []),
            'formatted_path': triple_result.get('formatted_path', ''),  # Original bracket format
            'formatted_path_enhanced': triple_result.get('formatted_path_enhanced', ''),  # New arrow format
            'total_length_meters': triple_result.get('total_length_meters', 0),
            'total_direction_changes': triple_result.get('total_direction_changes', 0),
            'path_analysis': path_analysis,
            'triple_details': {
                'total_triples': len(triple_result.get('path_triples', [])),
                'triples_with_directions': sum(1 for t in triple_result.get('path_triples', [])
                                               if t.get('direction') is not None),
                'triples_with_lengths': sum(1 for t in triple_result.get('path_triples', [])
                                            if t.get('length') is not None and t.get('length') > 0)
            },
            'formatting_details': {
                'original_format_length': len(triple_result.get('formatted_path', '')),
                'enhanced_format_length': len(triple_result.get('formatted_path_enhanced', '')),
                'formatting_style_used': 'dual_format'
            },
            'success': True,
            'error': None,
            'gpu_acceleration_used': graph_processor.use_gpu if graph_processor else False
        }

        return enhanced_result

    except Exception as e:
        print(f"❌ Error in parse_single_path_directions_enhanced: {e}")
        return {
            'path': path_nodes,
            'path_triples': [],
            'formatted_path': '',
            'formatted_path_enhanced': '',
            'total_length_meters': 0,
            'total_direction_changes': 0,
            'path_analysis': {'error': str(e)},
            'success': False,
            'error': str(e),
            'gpu_acceleration_used': graph_processor.use_gpu if graph_processor else False
        }


def parse_single_path_to_triples_optimized(path_nodes, nodes_gdf, edges_gdf, graph_processor,
                                           node_id_col='id', edge_source_col='id1', edge_target_col='id2'):
    """
    Optimized version of parse_single_path_to_triples with GPU support and enhanced formatting.
    """

    if len(path_nodes) < 2:
        return {
            'path': path_nodes,
            'path_triples': [],
            'total_length_meters': 0,
            'total_direction_changes': 0,
            'formatted_path': '',
            'formatted_path_enhanced': '',
            'error': 'Path too short'
        }

    # print(f"🔍 Parsing path triples with {len(path_nodes)} nodes")

    # Pre-load all node information
    path_nodes_info = {}
    for node_id in path_nodes:
        node_row = nodes_gdf[nodes_gdf[node_id_col] == node_id]
        if len(node_row) > 0:
            path_nodes_info[node_id] = node_row.iloc[0].to_dict()
        else:
            print(f"❌ Node {node_id} not found in nodes_gdf {len(nodes_gdf)}")
            print(f"type of node id: {type(node_id)}")
            import sys
            sys.exit(1)
        # else:
        #     path_nodes_info[node_id] = {'id': node_id, 'name': f'{node_id}'}

    # Pre-load all edge information using optimized lookup
    path_edges_info = {}
    for i in range(len(path_nodes) - 1):
        source_node = path_nodes[i]
        target_node = path_nodes[i + 1]
        edge_key = (source_node, target_node)

        edge_info = get_edge_info_optimized(source_node, target_node, edges_gdf, graph_processor,
                                            edge_source_col, edge_target_col)
        path_edges_info[edge_key] = edge_info

    # Process triples
    triples = []
    total_length = 0
    direction_changes = 0

    for i in range(len(path_nodes) - 1):
        source_node = path_nodes[i]
        target_node = path_nodes[i + 1]
        edge_key = (source_node, target_node)

        # Get pre-loaded information
        edge_info = path_edges_info[edge_key]
        source_info = path_nodes_info[source_node]
        target_info = path_nodes_info[target_node]

        # Create triple with enhanced processing
        triple = create_path_triple_optimized(
            source_node, target_node, source_info, target_info,
            edge_info, i, nodes_gdf, path_nodes, edges_gdf
        )

        triples.append(triple)
        total_length += triple.get('length', 0) or 0

        # Check for direction change (simplified)
        if i > 0 and is_direction_change_simple(triples[i - 1], triple):
            direction_changes += 1

    # Generate both old and new formatted paths
    formatted_path_old = format_triples_as_string_simple(triples)
    formatted_path_enhanced = format_complete_path_with_arrows(triples)

    return {
        'path': path_nodes,
        'path_triples': triples,
        'total_length_meters': total_length,
        'total_direction_changes': direction_changes,
        'formatted_path': formatted_path_old,  # Keep original format for compatibility
        'formatted_path_enhanced': formatted_path_enhanced,  # New arrow-separated format
        'error': None
    }


# Helper function to maintain compatibility with existing code
def format_triples_as_string_simple(triples):
    """
    Maintains the original simple formatting for backward compatibility
    """
    if not triples:
        return ""

    # Use the original bracket format for each triple
    formatted_triples = []
    for triple in triples:
        source_name = triple.get('source_name', 'Unknown')
        middle_name = triple.get('middle_component', 'Unknown')
        target_name = triple.get('target_name', 'Unknown')
        length = triple.get('length')
        direction = triple.get('direction')

        # Use the original formatting function
        formatted_triple = format_location_string_flexible_original(source_name, middle_name, target_name, length,
                                                                    direction)
        formatted_triples.append(formatted_triple)

    return " -> ".join(formatted_triples)


def get_edge_info_optimized(source_node, target_node, edges_gdf, graph_processor,
                            edge_source_col, edge_target_col):
    """
    Optimized edge information retrieval using both NetworkX and cuGraph.
    Enhanced to handle data type mismatches and missing edges.
    """

    edge_info = {
        'id1': source_node,
        'id2': target_node,
        'type': 'unknown',
        'length': 0,
        'distance': 0,
        'direction': None,
        'crossing_id': None
    }

    # print(f"🔍 DEBUG: Getting edge info for {source_node} -> {target_node}")
    # print(f"🔍 DEBUG: edges_gdf columns: {list(edges_gdf.columns)}")
    # print(f"🔍 DEBUG: edges_gdf shape: {edges_gdf.shape}")
    # print(f"🔍 DEBUG: edge_source_col: {edge_source_col}, edge_target_col: {edge_target_col}")

    # Try NetworkX graph first (faster for small lookups)
    if graph_processor.nx_graph and graph_processor.nx_graph.has_edge(source_node, target_node):
        nx_edge_data = graph_processor.nx_graph[source_node][target_node]
        # print(f"🔍 DEBUG: Found edge in NetworkX graph: {nx_edge_data}")
        edge_info.update(nx_edge_data)

        # Ensure we have length
        if not edge_info.get('length'):
            edge_info['length'] = nx_edge_data.get('weight', 1.0)
    # else:
    #     print(f"🔍 DEBUG: Edge not found in NetworkX graph")

    # Get additional info from edges_gdf with robust data type handling
    if len(str(source_node)) > 10:
        source_node = str(source_node)
    if len(str(target_node)) > 10:
        target_node = str(target_node)
    edge_rows = edges_gdf[
        ((edges_gdf[edge_source_col] == source_node) & (edges_gdf[edge_target_col] == target_node)) |
        ((edges_gdf[edge_source_col] == target_node) & (edges_gdf[edge_target_col] == source_node))
        ]

    # If no exact match found, try with data type conversion
    if len(edge_rows) == 0:
        # Try converting to int if source_node/target_node are strings
        try:
            source_int = int(source_node) if isinstance(source_node, str) else source_node
            target_int = int(target_node) if isinstance(target_node, str) else target_node

            edge_rows = edges_gdf[
                ((edges_gdf[edge_source_col] == source_int) & (edges_gdf[edge_target_col] == target_int)) |
                ((edges_gdf[edge_source_col] == target_int) & (edges_gdf[edge_target_col] == source_int))
                ]
        except (ValueError, TypeError):
            pass

        # Try converting to string if source_node/target_node are ints
        if len(edge_rows) == 0:
            try:
                source_str = str(source_node) if isinstance(source_node, int) else source_node
                target_str = str(target_node) if isinstance(target_node, int) else target_node

                edge_rows = edges_gdf[
                    ((edges_gdf[edge_source_col] == source_str) & (edges_gdf[edge_target_col] == target_str)) |
                    ((edges_gdf[edge_source_col] == target_str) & (edges_gdf[edge_target_col] == source_str))
                    ]
            except (ValueError, TypeError):
                pass

    # print(f"🔍 DEBUG: Found {len(edge_rows)} edge rows in edges_gdf")

    if len(edge_rows) > 0:
        edge_row = edge_rows.iloc[0]
        # print(f"🔍 DEBUG: Edge row from edges_gdf: {edge_row.to_dict()}")
        for col in edge_row.index:
            if col in edge_info and pd.notna(edge_row[col]):
                edge_info[col] = edge_row[col]
                # print(f"🔍 DEBUG: Updated edge_info[{col}] = {edge_row[col]}")
    else:
        # print(f"🔍 DEBUG: No edge found in edges_gdf")
        # If no edge found in edges_gdf but exists in NetworkX graph,
        # this might be a synthetic/pseudo-edge (like intersection connectors)
        if graph_processor.nx_graph and graph_processor.nx_graph.has_edge(source_node, target_node):
            # Use NetworkX data as fallback
            nx_edge_data = graph_processor.nx_graph[source_node][target_node]
            if 'type' in nx_edge_data and nx_edge_data['type'] != 'unknown':
                edge_info['type'] = nx_edge_data['type']
            if 'length' in nx_edge_data and nx_edge_data['length'] > 0:
                edge_info['length'] = nx_edge_data['length']
            # Mark as synthetic edge
            edge_info['synthetic'] = True

    # print(f"🔍 DEBUG: Final edge_info: {edge_info}")

    # Only print 'stop' for debugging if this is a real issue (not synthetic edges)
    if edge_info['type'] == 'unknown' and not edge_info.get('synthetic', False):
        print(f'🔍 DEBUG: Unknown edge type for {source_node} -> {target_node}')
        # Uncomment for detailed debugging:
        # print(f'🔍 DEBUG: Final edge_info: {edge_info}')
    return edge_info


def create_path_triple_optimized(source_node, target_node, source_info, target_info,
                                 edge_info, position, nodes_gdf, path_nodes, edges_gdf):
    """
    Optimized triple creation preserving original complex logic.
    """

    # Extract names and types (preserve original logic)
    source_name = get_node_display_name_original(source_node, source_info)
    target_name = get_node_display_name_original(target_node, target_info)

    # Determine the "street/POI" component using original logic
    edge_type, middle_name, middle_id = determine_middle_component_original(source_info, target_info, edge_info,
                                                                            nodes_gdf)

    # Get direction information using original enhanced logic
    direction, length = get_direction_and_length_enhanced_optimized(
        source_info, target_info, edge_info, nodes_gdf, position, path_nodes, edges_gdf
    )
    print(f"🔍 Direction: {direction}, Length: {length}")

    # Determine triple type (preserve original logic)
    triple_type = determine_triple_type_original(source_info, target_info, edge_info)
    if position == 0:
        triple_type = 'start'
    if position == len(path_nodes) - 1:
        triple_type = 'end'
    # Format using original flexible formatter
    formatted = format_location_string_flexible(source_name, middle_name, target_name, length, direction)

    return {
        'source_node': source_node,
        'target_node': target_node,
        'source_name': source_name,
        'target_name': target_name,
        'middle_component': middle_name,
        'direction': direction,
        'length': length,
        'triple_type': triple_type,
        'edge_type': edge_info.get('type'),
        'edge_info': edge_info,
        'target_info':target_info,
        'source_info':source_info,
        'position': position,
        'formatted': formatted
    }


def analyze_path_characteristics_optimized(path_nodes, triple_result, nodes_gdf, edges_gdf, graph_processor):
    """
    Optimized path analysis using GPU where possible.
    """

    analysis = {
        'path_length_nodes': len(path_nodes),
        'path_length_edges': len(path_nodes) - 1,
        'gpu_acceleration_used': graph_processor.use_gpu,
        'complexity_metrics': {},
        'spatial_metrics': {}
    }

    try:
        # Basic complexity metrics
        triples = triple_result.get('path_triples', [])
        total_length = triple_result.get('total_length_meters', 0)

        analysis['complexity_metrics'] = {
            'direction_changes': triple_result.get('total_direction_changes', 0),
            'direction_change_rate': triple_result.get('total_direction_changes', 0) / max(len(path_nodes) - 1, 1),
            'path_complexity_score': calculate_complexity_score_simple(triples)
        }

        analysis['spatial_metrics'] = {
            'total_length_meters': total_length,
            'total_length_km': total_length / 1000,
            'average_segment_length': total_length / max(len(path_nodes) - 1, 1)
        }

        # Use GPU for path distance calculations if available
        if graph_processor.use_gpu and len(path_nodes) > 2:
            analysis['gpu_metrics'] = calculate_gpu_path_metrics(path_nodes, graph_processor)

    except Exception as e:
        analysis['error'] = str(e)

    return analysis


# Simplified helper functions
def get_node_display_name_simple(node_id, node_info):
    """Simplified node name extraction"""
    return str(node_info.get('name', f'{node_id}'))


def determine_middle_component_simple(edge_type, edge_info, nodes_gdf):
    """Simplified middle component determination"""
    if edge_type == 'on_same_street':
        return 'same_street'
    elif 'crossing' in edge_type:
        return 'crossing'
    elif edge_type == 'nearest':
        return 'nearest'
    elif 'boundary' in edge_type:
        return 'boundary'
    else:
        return edge_type


def get_direction_simple(edge_info, source_info, target_info):
    """Simplified direction calculation"""
    direction = edge_info.get('direction') or edge_info.get('bearing')
    if direction:
        return f"{direction}°"
    return None


def is_direction_change_simple(prev_triple, curr_triple):
    """Simplified direction change detection"""
    prev_type = prev_triple.get('edge_type', '')
    curr_type = curr_triple.get('edge_type', '')
    return prev_type != curr_type or curr_type == 'crossing'


def format_triples_as_string_simple(triples):
    """Simplified triple formatting"""
    if not triples:
        return ""
    return " -> ".join(triple['formatted'] for triple in triples)


def calculate_complexity_score_simple(triples):
    """Simple complexity score calculation"""
    if not triples:
        return 0

    score = 0
    score += len(triples)  # Base complexity
    score += sum(1 for t in triples if t.get('edge_type') == 'crossing') * 2  # Crossings add complexity
    score += sum(1 for t in triples if t.get('direction')) * 0.5  # Directions add minor complexity

    return score


def calculate_gpu_path_metrics(path_nodes, graph_processor):
    """Calculate path metrics using GPU where possible"""
    metrics = {}

    try:
        if graph_processor.use_gpu and len(path_nodes) > 2:
            # Calculate total hop distance using GPU
            total_hops = 0
            for i in range(len(path_nodes) - 1):
                hop_dist = graph_processor.shortest_path_length(path_nodes[i], path_nodes[i + 1])
                if hop_dist != float('inf'):
                    total_hops += hop_dist

            metrics['total_graph_hops'] = total_hops
            metrics['average_hops_per_segment'] = total_hops / max(len(path_nodes) - 1, 1)
            metrics['gpu_calculation'] = True

    except Exception as e:
        metrics['gpu_error'] = str(e)
        metrics['gpu_calculation'] = False

    return metrics


def clean_route_result_for_json(route_result):
    """
    Specifically clean route_result object which contains subgraph info.
    """
    if not isinstance(route_result, dict):
        return make_json_serializable(route_result)

    cleaned = {}
    for key, value in route_result.items():
        if key == 'subgraph_info' and isinstance(value, dict):
            # Handle subgraph_info specially - this likely contains the NetworkX graph
            cleaned_subgraph = {}
            for sub_key, sub_value in value.items():
                if sub_key == 'subgraph' and hasattr(sub_value, 'nodes'):
                    # This is likely the NetworkX graph causing the issue
                    cleaned_subgraph[sub_key] = {
                        'type': 'networkx_graph_summary',
                        'num_nodes': len(sub_value.nodes()) if hasattr(sub_value, 'nodes') else 'unknown',
                        'num_edges': len(sub_value.edges()) if hasattr(sub_value, 'edges') else 'unknown',
                        # 'nodes_sample': list(sub_value.nodes())[:10] if hasattr(sub_value, 'nodes') else [],
                        'graph_type': type(sub_value).__name__,
                        'note': 'Original graph object converted to summary'
                    }
                else:
                    cleaned_subgraph[sub_key] = make_json_serializable(sub_value)
            cleaned[key] = cleaned_subgraph
        else:
            cleaned[key] = make_json_serializable(value)

    return cleaned


# Example usage integration
def process_paths_with_gpu_optimization(route_result, s2cell_id, graph_file, nodes, edges, graph_processor,
                                        filename='reasoning_path_QA',mapillary_nodes_gdf=None ):
    """
    Process paths using GPU-optimized parsing.
    This replaces your original processing loop.
    """

    if route_result:
        paths = route_result['all_paths']['paths']
        path_meta = {}
        results = []
        print("-----------------------------------------------------------")
        print(f"🚀 Processing {len(paths)} paths with GPU optimization...")
        paths_final = []
        print('path1', paths[0])

        for path_idx, path in enumerate(paths):
            print("-----------------------------------------------------------")
            print(f"Processing path {path_idx + 1}/{len(paths)}")
            print(path['nodes'])
            print("-----------------------------------------------------------")

            # Use GPU-optimized parsing
            result = parse_single_path_directions_enhanced_with_coordinates(
                path['nodes'], nodes, edges, graph_processor, mapillary_nodes_gdf=mapillary_nodes_gdf
            )

            if len(result['path_triples']) > 0:
                results.append(result)
                paths_final.append(path['nodes'])

        # Compile metadata
        if len(results) > 0:
            path_meta['paths'] = paths_final
            path_meta['s2cell'] = s2cell_id
            path_meta['subgraph'] = graph_file
            path_meta['paths_parsed'] = results
            path_meta['gpu_acceleration_used'] = graph_processor.use_gpu
            path_meta['total_processing_time'] = sum(1 for r in results if r.get('success', False))
            path_meta['route_results'] = clean_route_result_for_json(route_result)

            # Add image-related fields
            if 'image_nodes' in route_result:
                path_meta['image_nodes'] = route_result['image_nodes']
            if 'intermediate_images' in route_result:
                path_meta['intermediate_images'] = route_result['intermediate_images']
            if 'origin_image' in route_result:
                path_meta['origin_image'] = route_result['origin_image']
            if 'destination_image' in route_result:
                path_meta['destination_image'] = route_result['destination_image']

            # Make JSON serializable
            path_meta = make_json_serializable(path_meta)

            # Save results
            with open(f'{filename}.jsonl', 'a') as f:
                f.write(json.dumps(path_meta) + '\n')

            print(f"✅ Saved path analysis for {len(paths)} paths (GPU: {graph_processor.use_gpu})")

    return path_meta if route_result else None


def make_json_serializable(data):
    """Convert geometries and other non-serializable objects to serializable format"""
    if isinstance(data, dict):
        return {k: make_json_serializable(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [make_json_serializable(item) for item in data]
    elif hasattr(data, 'geom_type'):  # Shapely geometry
        return data.wkt
    elif hasattr(data, '__geo_interface__'):  # Other geometry objects
        return data.__geo_interface__
    elif isinstance(data, (np.integer, np.floating)):
        return data.item()
    elif isinstance(data, np.ndarray):
        return data.tolist()
    # Handle NetworkX Graph objects
    elif hasattr(data, 'nodes') and hasattr(data, 'edges'):
        # Check if it's a NetworkX graph-like object
        try:
            import networkx as nx
            if isinstance(data, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
                return {
                    'type': 'networkx_graph_summary',
                    'num_nodes': len(data.nodes()) if hasattr(data, 'nodes') else 'unknown',
                    'num_edges': len(data.edges()) if hasattr(data, 'edges') else 'unknown',
                    'graph_type': type(data).__name__,
                    'note': 'Original graph object converted to summary for JSON serialization'
                }
        except (ImportError, AttributeError):
            pass
    # Handle other graph-like objects
    elif hasattr(data, '__class__') and 'Graph' in type(data).__name__:
        return {
            'type': 'graph_object_summary',
            'graph_type': type(data).__name__,
            'note': 'Graph object removed for JSON serialization'
        }
    else:
        return data


"""New"""


def get_direction_and_length_enhanced_with_coordinates(source_info, target_info, edge_info, nodes_gdf, position, path,
                                                       edges_gdf=None, recorded_coordinates=None):
    """
    Enhanced get_direction_and_length with coordinate recording for better transition calculation

    Args:
        recorded_coordinates: Dictionary to store coordinates for each position in path
                             Format: {position: (x, y, coordinate_type)}
    """
    edge_type = edge_info.get('type', '')
    print(f'------------edge_type at position {position}------------', edge_type)
    direction = None
    length = 0
    current_coordinates = None

    # Check if previous coordinates are available for transition calculation
    previous_coordinates = None
    if recorded_coordinates and position > 0 and (position - 1) in recorded_coordinates:
        prev_x, prev_y, prev_type = recorded_coordinates[position - 1]
        previous_coordinates = (prev_x, prev_y)
        print(f'Previous coordinates from position {position - 1}: ({prev_x}, {prev_y}) [{prev_type}]')

    # Main direction calculation logic with coordinate recording
    if edge_type == 'on_same_street':
        # Check if we need to use previous coordinates for transition
        if previous_coordinates:
            direction, length = calculate_direction_from_previous_coordinates(
                previous_coordinates, target_info, edge_info
            )
        else:
            direction = calculate_same_street_direction_optimized(source_info, target_info, edge_info, nodes_gdf)
            length = edge_info.get('crossing_distance_meters', 0)

        # Record current coordinates (street intersection or point)
        current_coordinates = get_and_record_street_coordinates(target_info, edge_info, nodes_gdf)

    elif 'crossing' in edge_type and edge_type != "boundary_crossing":
        # Check if we need to use previous coordinates for transition
        if previous_coordinates:
            direction, length = calculate_crossing_direction_from_previous_coordinates(
                previous_coordinates, target_info, edge_info, nodes_gdf
            )
        else:
            direction, length = calculate_crossing_direction_optimized(
                source_info, target_info, nodes_gdf, edge_info, position, path, edges_gdf=edges_gdf
            )

        # Record crossing coordinates
        current_coordinates = get_and_record_crossing_coordinates(target_info, edge_info, nodes_gdf)

    elif edge_type == 'nearest' or edge_type == 'near':

        print(f"🔍 Processing {edge_type} edge with enhanced coordinates...")

        # Use enhanced direction calculation
        print("source_info", source_info)
        print("target_info", target_info)
        print("edge_info", edge_info)

        # Check if geometry is preserved
        source_geom = source_info.get('geometry')
        target_geom = target_info.get('geometry')
        print(f"🔍 Source geometry type: {type(source_geom).__name__ if source_geom else 'None'}")
        print(f"🔍 Target geometry type: {type(target_geom).__name__ if target_geom else 'None'}")

        # If geometry is missing, try to reconstruct from coordinates
        if not source_geom and 'x' in source_info and 'y' in source_info:
            from shapely.geometry import Point
            source_geom = Point(source_info['x'], source_info['y'])
            source_info['geometry'] = source_geom
            print(f"🔍 Reconstructed source geometry from coordinates")

        if not target_geom and 'x' in target_info and 'y' in target_info:
            from shapely.geometry import Point
            target_geom = Point(target_info['x'], target_info['y'])
            target_info['geometry'] = target_geom
            print(f"🔍 Reconstructed target geometry from coordinates")

        direction = calculate_nearest_direction_enhanced(source_info, target_info, edge_info)

        # Use enhanced distance calculation
        # length = edge_info['calculated_distance']
        length = calculate_nearest_distance_enhanced(source_info, target_info, edge_info)
        print("edge_info of nearest", edge_info)
        # Use enhanced coordinate recording
        current_coordinates = get_and_record_nearest_coordinates_enhanced(source_info, target_info, edge_info)

        print(f"🔍 Enhanced {edge_type} edge result: direction={direction}, length={length}m")

    elif 'boundary' in edge_type or 'intersect' in edge_type:
        direction, length = get_direction_for_boundary_edge_original(source_info, target_info, edge_info, nodes_gdf)

        # Record boundary coordinates (polygon centroid or intersection point)
        current_coordinates = get_and_record_boundary_coordinates(source_info, target_info, edge_info, nodes_gdf)

    else:
        # Fallback: record basic node coordinates
        current_coordinates = get_basic_node_coordinates(target_info)

    # Store coordinates for this position
    if recorded_coordinates is not None and current_coordinates is not None:
        x, y, coord_type = current_coordinates
        recorded_coordinates[position] = (x, y, coord_type)
        print(f'Recorded coordinates for position {position}: ({x}, {y}) [{coord_type}]')

    return direction, length, current_coordinates


def get_and_record_street_coordinates(target_info, edge_info, nodes_gdf):
    """Get coordinates for street/intersection nodes"""
    try:
        # First try to get intersection coordinates if it's a crossing
        if 'crossing_id' in edge_info and pd.notna(edge_info['crossing_id']):
            crossing_coords = get_crossing_coordinates_original(None, edge_info, nodes_gdf)
            if crossing_coords:
                return crossing_coords[0], crossing_coords[1], 'intersection'

        # Fallback to target node coordinates
        target_x = get_coordinate_from_node_original(target_info, 'x')
        target_y = get_coordinate_from_node_original(target_info, 'y')

        if target_x is not None and target_y is not None:
            return target_x, target_y, 'street_node'

        # Final fallback to geometry
        geom = target_info.get('geometry')
        if geom:
            if hasattr(geom, 'x') and hasattr(geom, 'y'):
                return geom.x, geom.y, 'point_geometry'
            elif hasattr(geom, 'centroid'):
                return geom.centroid.x, geom.centroid.y, 'centroid'

    except Exception as e:
        print(f"Error getting street coordinates: {e}")

    return None


def identify_aoi_and_street_nodes_enhanced(source_info, target_info):
    """
    Enhanced identify_aoi_and_street_nodes with relationship type detection
    """
    source_geometry = source_info.get('geometry')
    target_geometry = target_info.get('geometry')

    # Check if source is AOI (Polygon) and target is Line
    if isinstance(source_geometry, (Polygon, MultiPolygon)) and isinstance(target_geometry,
                                                                           (LineString, MultiLineString)):
        return source_info, target_info, 'aoi_to_line'

    # Check if source is Line and target is AOI (Polygon)
    elif isinstance(source_geometry, (LineString, MultiLineString)) and isinstance(target_geometry,
                                                                                   (Polygon, MultiPolygon)):
        return target_info, source_info, 'line_to_aoi'

    # Check by geometry type strings if shapely objects not available
    elif hasattr(source_geometry, 'geom_type') and hasattr(target_geometry, 'geom_type'):
        if source_geometry.geom_type in ['Polygon', 'MultiPolygon'] and target_geometry.geom_type in ['LineString',
                                                                                                      'MultiLineString']:
            return source_info, target_info, 'aoi_to_line'
        elif source_geometry.geom_type in ['LineString', 'MultiLineString'] and target_geometry.geom_type in ['Polygon',
                                                                                                              'MultiPolygon']:
            return target_info, source_info, 'line_to_aoi'

    # Fallback: check for polygon geometry in either node
    for node_info in [source_info, target_info]:
        geometry = node_info.get('geometry')
        if isinstance(geometry, (Polygon, MultiPolygon)):
            aoi_info = node_info
            street_info = target_info if node_info == source_info else source_info
            return aoi_info, street_info, 'fallback'

    # If can't identify clearly, use source as AOI, target as street
    return source_info, target_info, 'fallback'


def get_and_record_crossing_coordinates(target_info, edge_info, nodes_gdf):
    """Get coordinates for crossing points"""
    try:
        # Try to get crossing coordinates from edge info
        crossing_coords = get_crossing_coordinates_original(None, edge_info, nodes_gdf)
        if crossing_coords:
            return crossing_coords[0], crossing_coords[1], 'crossing_point'

        # Fallback to target coordinates
        return get_and_record_street_coordinates(target_info, edge_info, nodes_gdf)

    except Exception as e:
        print(f"Error getting crossing coordinates: {e}")
        return None


def get_and_record_nearest_coordinates(source_info, target_info, edge_info):
    """Get coordinates for nearest edge cases - record the target node coordinates, not the nearest point"""
    try:
        # For nearest edges, we want to record the actual target node coordinates
        # (the Point or POI that we're navigating to), not the nearest point on the street

        # # First try target node coordinates
        # target_x = get_coordinate_from_node_original(target_info, 'x')
        # target_y = get_coordinate_from_node_original(target_info, 'y')
        #
        # if target_x is not None and target_y is not None:
        #     return target_x, target_y, 'nearest_target_node'

        # Try target geometry
        # Only fallback to nearest point coordinates if target coordinates not available
        nearest_x = edge_info.get('nearest_x')
        nearest_y = edge_info.get('nearest_y')

        if nearest_x is not None and nearest_y is not None:
            print('@@Found coordinates via nearest_x in edge')
            return nearest_x, nearest_y, 'nearest_point_fallback'

        target_geom = target_info.get('geometry')

        if target_geom:
            if hasattr(target_geom, 'x') and hasattr(target_geom, 'y'):
                print('@@Found coordinates via node coordinates')
                return target_geom.x, target_geom.y, 'nearest_target_point'
            elif hasattr(target_geom, 'centroid'):
                return target_geom.centroid.x, target_geom.centroid.y, 'nearest_target_centroid'

        # Try other coordinate fields
        for x_field, y_field in [('x', 'y')]:
            x_val = edge_info.get(x_field)
            y_val = edge_info.get(y_field)
            if x_val is not None and y_val is not None:
                print('@@Found coordinates via edge')
                return x_val, y_val, f'nearest_{x_field}_fallback'

    except Exception as e:
        print(f"Error getting nearest coordinates: {e}")

    return None


def get_and_record_boundary_coordinates(source_info, target_info, edge_info, nodes_gdf):
    """Get coordinates for boundary/intersect edge cases"""
    try:
        # Identify AOI and street nodes
        aoi_info, street_info, relationship_type = identify_aoi_and_street_nodes_enhanced(source_info, target_info)

        if aoi_info:
            # For boundary cases, record the AOI centroid
            aoi_coords = get_aoi_centroid_coordinates(aoi_info)
            if aoi_coords:
                return aoi_coords[0], aoi_coords[1], 'aoi_centroid'

        # Fallback to target coordinates
        target_x = get_coordinate_from_node_original(target_info, 'x')
        target_y = get_coordinate_from_node_original(target_info, 'y')

        if target_x is not None and target_y is not None:
            return target_x, target_y, 'boundary_node'

    except Exception as e:
        print(f"Error getting boundary coordinates: {e}")

    return None


def get_basic_node_coordinates(node_info):
    """Get basic coordinates for any node"""
    try:
        # Try direct coordinates
        x = get_coordinate_from_node_original(node_info, 'x')
        y = get_coordinate_from_node_original(node_info, 'y')

        if x is not None and y is not None:
            return x, y, 'basic_node'

        # Try geometry
        geom = node_info.get('geometry')
        if geom:
            if hasattr(geom, 'x') and hasattr(geom, 'y'):
                return geom.x, geom.y, 'basic_point'
            elif hasattr(geom, 'centroid'):
                return geom.centroid.x, geom.centroid.y, 'basic_centroid'

    except Exception as e:
        print(f"Error getting basic coordinates: {e}")

    return None


def calculate_direction_from_previous_coordinates(previous_coords, target_info, edge_info):
    """Calculate direction from previous recorded coordinates to current target"""
    try:
        prev_x, prev_y = previous_coords

        # Get current target coordinates
        target_x = get_coordinate_from_node_original(target_info, 'x')
        target_y = get_coordinate_from_node_original(target_info, 'y')

        if all(coord is not None for coord in [prev_x, prev_y, target_x, target_y]):
            bearing = calculate_bearing_original(prev_x, prev_y, target_x, target_y)
            direction = convert_bearing_to_direction_original(bearing)

            # Calculate distance
            from geopy.distance import geodesic
            distance = geodesic((prev_y, prev_x), (target_y, target_x)).meters

            print(f"📐 Direction from previous coordinates:")
            print(f"   From: ({prev_x}, {prev_y}) [Previous recorded]")
            print(f"   To:   ({target_x}, {target_y}) [Current target]")
            print(f"   Bearing: {bearing:.1f}°")
            print(f"   Direction: {direction}")
            print(f"   Distance: {distance:.1f}m")

            return direction, distance

    except Exception as e:
        print(f"Error calculating direction from previous coordinates: {e}")

    return None, 0


def get_crossing_coordinates_original(target_id, edge_info, nodes_gdf):
    """Original get_crossing_coordinates logic (preserved)"""
    crossing_id = edge_info['crossing_id']
    crossing_node = nodes_gdf[nodes_gdf['id'] == crossing_id]
    print('next crossing', crossing_node.name.iloc[0])

    if len(crossing_node) > 0:
        crossing_geom = crossing_node.iloc[0].get('geometry')
        if crossing_geom:
            try:
                return crossing_geom.x, crossing_geom.y
            except:
                try:
                    return crossing_geom.centroid.x, crossing_geom.centroid.y
                except:
                    pass

        crossing_x = crossing_node.iloc[0].get('x')
        crossing_y = crossing_node.iloc[0].get('y')
        if crossing_x is not None and crossing_y is not None:
            return crossing_x, crossing_y

    return None, None


def calculate_crossing_direction_from_previous_coordinates(previous_coords, target_info, edge_info, nodes_gdf):
    """Calculate crossing direction from previous coordinates to crossing point"""
    try:
        prev_x, prev_y = previous_coords

        # For crossing edge, get the crossing point coordinates
        crossing_coords = get_crossing_coordinates_original(None, edge_info, nodes_gdf)
        if crossing_coords:
            crossing_x, crossing_y = crossing_coords

            bearing = calculate_bearing_original(prev_x, prev_y, crossing_x, crossing_y)
            direction = convert_bearing_to_direction_original(bearing)

            from geopy.distance import geodesic
            distance = geodesic((prev_y, prev_x), (crossing_y, crossing_x)).meters

            print(f"📐 Crossing direction from previous coordinates:")
            print(f"   From: ({prev_x}, {prev_y}) [Previous recorded]")
            print(f"   To:   ({crossing_x}, {crossing_y}) [Crossing point]")
            print(f"   Bearing: {bearing:.1f}°")
            print(f"   Direction: {direction}")
            print(f"   Distance: {distance:.1f}m")

            return direction, distance

        # Fallback to target coordinates
        return calculate_direction_from_previous_coordinates(previous_coords, target_info, edge_info)

    except Exception as e:
        print(f"Error calculating crossing direction from previous coordinates: {e}")

    return None, 0


def determine_triple_type_original(source_info, target_info, edge_info):
    """
    Original determine_triple_type logic (preserved)
    """
    edge_type = edge_info.get('type', '').lower()

    # Street segments (no direction change)
    if 'on_same_street' in edge_type:
        return 'street_segment'

    # Decision points (potential direction change)
    if 'crossing' in edge_type:
        return 'decision_point'

    # Check if both nodes are crossings (likely street segment)
    source_has_crossing = 'crossing_id' in source_info and pd.notna(source_info['crossing_id'])
    target_has_crossing = 'crossing_id' in target_info and pd.notna(target_info['crossing_id'])

    if source_has_crossing and target_has_crossing:
        return 'street_segment'

    # Default to decision point
    return None


def create_path_triple_with_coordinates(source_node, target_node, source_info, target_info,
                                        edge_info, position, nodes_gdf, path_nodes, edges_gdf,
                                        recorded_coordinates=None):
    """
    Enhanced triple creation with coordinate recording system
    """

    # Extract names and types (preserve original logic)
    source_name = get_node_display_name_original(source_node, source_info)
    target_name = get_node_display_name_original(target_node, target_info)

    # Determine the "street/POI" component using original logic
    edge_type, middle_name, middle_id = determine_middle_component_original(source_info, target_info, edge_info,
                                                                            nodes_gdf)

    # Get direction information using enhanced logic with coordinate recording
    direction, length, current_coordinates = get_direction_and_length_enhanced_with_coordinates(
        source_info, target_info, edge_info, nodes_gdf, position, path_nodes, edges_gdf, recorded_coordinates
    )

    # Determine triple type (preserve original logic)
    triple_type = determine_triple_type_original(source_info, target_info, edge_info)
    if position == 0:
        triple_type = 'start'
    if position == len(path_nodes) - 2:  # Last edge
        triple_type = 'end'

    # Format using original flexible formatter
    formatted = format_location_string_flexible(source_name, middle_name, target_name, length, direction)

    # Store coordinate information in triple for reference
    coordinate_info = {}
    if current_coordinates:
        x, y, coord_type = current_coordinates
        coordinate_info = {
            'x': x,
            'y': y,
            'coordinate_type': coord_type,
            'position': position
        }

    return {
        'source_node': source_node,
        'target_node': target_node,
        'source_name': source_name,
        'target_name': target_name,
        'middle_component': middle_name,
        'direction': direction,
        'length': length,
        'triple_type': triple_type,
        'edge_type': edge_info.get('type'),
        'edge_info': edge_info,
        'position': position,
        'formatted': formatted,
        'source_info': source_info,
        'target_info': target_info,
        'coordinates': coordinate_info
    }


def parse_single_path_to_triples_with_coordinates(path_nodes, nodes_gdf, edges_gdf, graph_processor,
                                                  node_id_col='id', edge_source_col='id1', edge_target_col='id2', mapillary_nodes_gdf=None):
    """
    Enhanced path parsing with coordinate recording system
    """
    if len(path_nodes) < 2:
        return {
            'path': path_nodes,
            'path_triples': [],
            'total_length_meters': 0,
            'total_direction_changes': 0,
            'formatted_path': '',
            'formatted_path_enhanced': '',
            'recorded_coordinates': {},
            'error': 'Path too short'
        }

    print(f"🔍 Parsing path with coordinate recording: {len(path_nodes)} nodes")

    # Initialize coordinate recording dictionary
    recorded_coordinates = {}

    # Pre-load all node information
    path_nodes_info = {}
    for node_id in path_nodes:

        if len(str(node_id)) > 10:
            node_id = str(node_id)
        else:
            node_id = int(node_id)

        node_row = nodes_gdf[nodes_gdf[node_id_col] == node_id]
        if len(node_row) > 0:
            node_dict = node_row.iloc[0].to_dict()
            # Ensure geometry is preserved
            if 'geometry' in node_row.iloc[0]:
                node_dict['geometry'] = node_row.iloc[0]['geometry']
                # print(f"🔍 Preserved geometry for node {node_id}: {type(node_dict['geometry']).__name__}")
            else:
                print(f"🔍 No geometry found for node {node_id}")
                # Try to supplement geometry from mapillary_nodes_gdf if provided
                try:
                    if mapillary_nodes_gdf is not None and hasattr(mapillary_nodes_gdf, 'columns') \
                            and 'geometry' in getattr(mapillary_nodes_gdf, 'columns', []):
                        # Compare as strings to avoid dtype mismatches
                        node_id_str = str(node_id)
                        id_series = mapillary_nodes_gdf[node_id_col].astype(str)
                        mapillary_node_row = mapillary_nodes_gdf[id_series == node_id_str]
                        if len(mapillary_node_row) > 0:
                            node_dict['geometry'] = mapillary_node_row.iloc[0]['geometry']
                            print(f"🔍 Preserved geometry for node {node_id} from mapillary_nodes_gdf")
                        else:
                            print(f"🔍 No geometry found for node {node_id} in mapillary_nodes_gdf")
                    else:
                        print("🔍 mapillary_nodes_gdf not provided or missing 'geometry' column; skipping supplement.")
                except Exception as e:
                    print(f"⚠️ Error checking mapillary_nodes_gdf for node {node_id}: {e}")

            path_nodes_info[node_id] = node_dict
        else:
            path_nodes_info[node_id] = {'id': node_id, 'name': f'{node_id}'}
            import sys
            print(f"🔍 No node data found for node {node_id}")
            sys.exit(1)
            #

    # Pre-load all edge information
    path_edges_info = {}
    for i in range(len(path_nodes) - 1):
        source_node = path_nodes[i]
        target_node = path_nodes[i + 1]
        if len(str(source_node)) > 10:
            source_node = str(source_node)
        if len(str(target_node)) > 10:
            target_node = str(target_node)
        edge_key = (source_node, target_node)

        edge_info = get_edge_info_optimized(source_node, target_node, edges_gdf, graph_processor,
                                            edge_source_col, edge_target_col)
        path_edges_info[edge_key] = edge_info

    # Process triples with coordinate recording
    triples = []
    total_length = 0
    direction_changes = 0
    for i in range(len(path_nodes) - 1):
        source_node = path_nodes[i]
        target_node = path_nodes[i + 1]
        if len(str(source_node)) > 10:
            source_node = str(source_node)
        if len(str(target_node)) > 10:
            target_node = str(target_node)
        edge_key = (source_node, target_node)

        # Get pre-loaded information
        edge_info = path_edges_info[edge_key]
        source_info = path_nodes_info[source_node]
        target_info = path_nodes_info[target_node]

        # Create triple with coordinate recording
        triple = create_path_triple_with_coordinates(
            source_node, target_node, source_info, target_info,
            edge_info, i, nodes_gdf, path_nodes, edges_gdf, recorded_coordinates
        )

        triples.append(triple)
        total_length += triple.get('length', 0) or 0

        # Check for direction change
        if i > 0 and is_direction_change_simple(triples[i - 1], triple):
            direction_changes += 1

    # Generate formatted paths
    # try:
    formatted_path_enhanced = format_enhanced_path_with_coordinates_improved(triples, recorded_coordinates)
    # except Exception as e:
    #     print(f"❌ Error in enhanced path formatting: {e}")
    #     formatted_path_enhanced = ""

    formatted_path_old = format_triples_as_string_simple(triples)

    # print('-----------------------REASONING PATH-------------------------')
    # print(formatted_path_enhanced)

    print(f"📍 Recorded coordinates for {len(recorded_coordinates)} positions")

    return {
        'path': path_nodes,
        'path_triples': triples,
        'total_length_meters': total_length,
        'total_direction_changes': direction_changes,
        'formatted_path': formatted_path_old,
        'formatted_path_enhanced': formatted_path_enhanced,
        'recorded_coordinates': recorded_coordinates,
        'error': None
    }


def calculate_transition_to_final_destination(previous_coords, target_info, edge_info):
    """
    Calculate transition from previous recorded coordinates to final destination
    Used specifically for last node nearest/boundary cases
    """
    try:
        prev_x, prev_y = previous_coords

        # For the final destination, we want the actual target coordinates
        # Get target node coordinates (the final POI/destination)
        target_x = get_coordinate_from_node_original(target_info, 'x')
        target_y = get_coordinate_from_node_original(target_info, 'y')

        # If target coordinates not available, try geometry
        if target_x is None or target_y is None:
            target_geom = target_info.get('geometry')
            if target_geom:
                if hasattr(target_geom, 'x') and hasattr(target_geom, 'y'):
                    target_x, target_y = target_geom.x, target_geom.y
                elif hasattr(target_geom, 'centroid'):
                    target_x, target_y = target_geom.centroid.x, target_geom.centroid.y

        if all(coord is not None for coord in [prev_x, prev_y, target_x, target_y]):
            bearing = calculate_bearing_original(prev_x, prev_y, target_x, target_y)
            direction = convert_bearing_to_direction_original(bearing)

            # Calculate distance
            from geopy.distance import geodesic
            distance = geodesic((prev_y, prev_x), (target_y, target_x)).meters

            print(f"📐 Transition to final destination:")
            print(f"   From: ({prev_x}, {prev_y}) [Previous recorded position]")
            print(f"   To:   ({target_x}, {target_y}) [Final destination]")
            print(f"   Bearing: {bearing:.1f}°")
            print(f"   Direction: {direction}")
            print(f"   Distance: {distance:.1f}m")

            return direction, distance

    except Exception as e:
        print(f"Error calculating transition to final destination: {e}")

    return None, 0


def parse_single_path_directions_enhanced_with_coordinates(path_nodes, nodes_gdf, edges_gdf, graph_processor,
                                                           node_id_col='id', edge_source_col='id1',
                                                           edge_target_col='id2', mapillary_nodes_gdf=None):
    """
    Enhanced path parsing with coordinate recording system and proper first/last handling
    """
    path_nodes = [int(p) for p in path_nodes]
    if not path_nodes or len(path_nodes) < 2:
        return {
            'path': path_nodes,
            'path_triples': [],
            'formatted_path': '',
            'formatted_path_enhanced': '',
            'total_length_meters': 0,
            'total_direction_changes': 0,
            'path_analysis': {'error': 'Path too short'},
            'success': False,
            'gpu_acceleration_used': False
        }

    print(f"🔍 Parsing enhanced path with coordinates: {len(path_nodes)} nodes: {path_nodes}")

    try:
        # Use the coordinate recording parsing
        triple_result = parse_single_path_to_triples_with_coordinates(
            path_nodes, nodes_gdf, edges_gdf, graph_processor,
            node_id_col, edge_source_col, edge_target_col, mapillary_nodes_gdf
        )

        print("=================enhanced coordinate triple====================")
        print(triple_result['formatted_path_enhanced'])
        print("=================enhanced coordinate triple====================")

        # Enhanced path analysis (reuse existing function)
        path_analysis = analyze_path_characteristics_optimized(
            path_nodes, triple_result, nodes_gdf, edges_gdf, graph_processor
        )

        # Compile results with both formatting styles
        enhanced_result = {
            'path': path_nodes,
            'path_triples': triple_result.get('path_triples', []),
            'formatted_path': triple_result.get('formatted_path', ''),  # Original bracket format
            'formatted_path_enhanced': triple_result.get('formatted_path_enhanced', ''),  # New enhanced format
            'total_length_meters': triple_result.get('total_length_meters', 0),
            'total_direction_changes': triple_result.get('total_direction_changes', 0),
            'recorded_coordinates': triple_result.get('recorded_coordinates', {}),
            'path_analysis': path_analysis,
            'triple_details': {
                'total_triples': len(triple_result.get('path_triples', [])),
                'triples_with_directions': sum(1 for t in triple_result.get('path_triples', [])
                                               if t.get('direction') is not None),
                'triples_with_lengths': sum(1 for t in triple_result.get('path_triples', [])
                                            if t.get('length') is not None and t.get('length') > 0),
                'coordinates_recorded': len(triple_result.get('recorded_coordinates', {}))
            },
            'formatting_details': {
                'original_format_length': len(triple_result.get('formatted_path', '')),
                'enhanced_format_length': len(triple_result.get('formatted_path_enhanced', '')),
                'formatting_style_used': 'coordinate_recording_with_first_last_handling'
            },
            'success': True,
            'error': None,
            'gpu_acceleration_used': graph_processor.use_gpu if graph_processor else False
        }

        return enhanced_result

    except Exception as e:
        print(f"❌ Error in parse_single_path_directions_enhanced_with_coordinates: {e}")
        return {
            'path': path_nodes,
            'path_triples': [],
            'formatted_path': '',
            'formatted_path_enhanced': '',
            'total_length_meters': 0,
            'total_direction_changes': 0,
            'path_analysis': {'error': str(e)},
            'success': False,
            'error': str(e),
            'gpu_acceleration_used': graph_processor.use_gpu if graph_processor else False
        }


def format_enhanced_path_with_coordinates(triples, recorded_coordinates):
    """
    Enhanced path formatting using recorded coordinates with proper first/last nearest/boundary handling

    Expected format:
    - First nearest/boundary: "(Bloom 45, nearest, W 45 St) (12.5m, 17°(N)) -> ..."
    - Last nearest/boundary: "... -> (transition_distance, transition_direction) -> (W 43 St, nearest, bella vita tranttoria) (0.6m, 198°(S))"
    """
    if not triples:
        return ""

    path_parts = []

    for i, triple in enumerate(triples):
        is_first = (i == 0)
        is_last = (i == len(triples) - 1)

        # Create the basic triple part
        source_name = triple.get('source_name', 'Unknown')
        middle_name = triple.get('middle_component', 'Unknown')
        target_name = triple.get('target_name', 'Unknown')

        triple_part = f"({source_name}, {middle_name}, {target_name})"

        # Get edge type and direction/distance info
        edge_type = triple.get('edge_type', '')
        length = triple.get('length')
        direction = triple.get('direction')

        # SPECIAL CASE 1: First node with nearest/boundary edge
        if is_first and 'nearest' in edge_type or 'boundary' in edge_type:
            direction_tuple = format_direction_distance_tuple(length, direction)
            path_parts.append(f"{triple_part} {direction_tuple}")

            # Add arrow for continuation (if not the only triple)
            if not is_last:
                path_parts.append("->")

        # SPECIAL CASE 2: Last node with nearest/boundary edge
        elif is_last and ('nearest' in edge_type or 'boundary' in edge_type or 'near' in edge_type):
            # First add the transition from previous coordinates to final destination
            if i > 0 and recorded_coordinates:
                prev_coords = recorded_coordinates.get(i - 1)

                if prev_coords:
                    prev_x, prev_y, _ = prev_coords

                    # Get target_info from the triple
                    target_info = triple.get('target_info', {})
                    edge_info = triple.get('edge_info', {})

                    # Calculate transition to the actual final destination
                    transition_direction, transition_distance = calculate_transition_to_final_destination(
                        (prev_x, prev_y), target_info, edge_info
                    )

                    if transition_direction and transition_distance:
                        transition_tuple = format_direction_distance_tuple(transition_distance, transition_direction)
                        path_parts.append(transition_tuple)
                        path_parts.append("->")

            # Then add the final triple with its own direction/distance tuple
            direction_tuple = format_direction_distance_tuple(length, direction)
            path_parts.append(f"{triple_part} {direction_tuple}")

        # NORMAL CASE: Middle nodes or first/last without nearest/boundary
        else:
            path_parts.append(triple_part)

            # Add direction/distance tuple and arrow if not the last triple
            if not is_last:
                direction_tuple = format_direction_distance_tuple(length, direction)
                path_parts.append("->")
                path_parts.append(direction_tuple)
                path_parts.append("->")

    # Join and clean up formatting
    result = " ".join(path_parts)

    # Clean up multiple consecutive arrows and extra spaces
    import re
    result = re.sub(r'\s*->\s*->\s*', ' -> ', result)
    result = re.sub(r'\s+', ' ', result)

    # Final check: ensure the last triple has distance/direction if available
    # This is a safety net for cases where the edge type detection failed
    if triples and len(triples) > 0:
        last_triple = triples[-1]
        last_triple_text = f"({last_triple.get('source_name', 'Unknown')}, {last_triple.get('middle_component', 'Unknown')}, {last_triple.get('target_name', 'Unknown')})"

        # Check if the last triple is missing its distance/direction tuple
        if last_triple_text in result and not result.strip().endswith(')'):
            # Try to add distance/direction from the triple data
            last_length = last_triple.get('length')
            last_direction = last_triple.get('direction')

            if last_length and last_direction:
                direction_tuple = format_direction_distance_tuple(last_length, last_direction)
                result = result.rstrip() + f" {direction_tuple}"
                print(f"🔧 Added missing distance/direction to last triple: {direction_tuple}")
            else:
                print(f"⚠️ Last triple missing distance/direction but no data available")
                print(f"   Last triple: {last_triple_text}")
                print(f"   Length: {last_length}, Direction: {last_direction}")

    return result.strip()


def get_shortest_paths_to_destinations_with_coordinates(center_node, destination_nodes, graph_processor, nodes_gdf,
                                                        edges_gdf):
    """
    Get exactly ONE shortest path from center to each destination using coordinate recording approach
    Updated to use the enhanced path parsing with coordinate recording and proper first/last node handling

    Args:
        center_node: Center node ID
        destination_nodes: List of destination node IDs
        graph_processor: GraphProcessor instance
        nodes_gdf: GeoDataFrame containing nodes
        edges_gdf: GeoDataFrame containing edges

    Returns:
        Dictionary with ONE shortest path to each destination using enhanced coordinate recording
    """

    shortest_paths = {}

    print(f"🛣️ Finding ONE shortest path from center {center_node} to {len(destination_nodes)} destinations")
    print("🚀 Using coordinate recording approach for enhanced accuracy")

    for dest_node in destination_nodes:
        try:
            # Get THE shortest path (only one)
            shortest_path = nx.shortest_path(graph_processor.nx_graph, center_node, dest_node)
            path_length = len(shortest_path) - 1

            print(f"  Shortest path to {dest_node}: {path_length} hops, {len(shortest_path)} nodes")

            # Parse this single path for detailed directions using coordinate recording
            try:
                parsed_result = parse_single_path_directions_enhanced_with_coordinates(
                    shortest_path, nodes_gdf, edges_gdf, graph_processor
                )

                if parsed_result.get('success', False):
                    path_data = {
                        'path_nodes': shortest_path,
                        'path_length': len(shortest_path),
                        'hop_distance': path_length,
                        # 'formatted_path_original': parsed_result.get('formatted_path', ''),  # Original format
                        'formatted_path_enhanced': parsed_result.get('formatted_path_enhanced', ''),  # Enhanced format
                        # 'formatted_path': parsed_result.get('formatted_path_enhanced', ''),  # Use enhanced as default
                        'total_length_meters': parsed_result.get('total_length_meters', 0),
                        'direction_changes': parsed_result.get('total_direction_changes', 0),
                        'path_triples': parsed_result.get('path_triples', []),
                        'recorded_coordinates': parsed_result.get('recorded_coordinates', {}),
                        'coordinates_recorded_count': len(parsed_result.get('recorded_coordinates', {})),
                        'parsing_successful': True,
                        'coordinate_recording_applied': True
                    }

                    print(f"  ✅ Coordinate recording parsing successful for {dest_node}")
                    print(f"     Coordinates recorded: {len(parsed_result.get('recorded_coordinates', {}))} positions")
                    print(f"     Enhanced format: {parsed_result.get('formatted_path_enhanced', '')[:100]}...")

                    # Check for special first/last cases
                    triples = parsed_result.get('path_triples', [])
                    if triples:
                        first_edge_type = triples[0].get('edge_type', '')
                        last_edge_type = triples[-1].get('edge_type', '')

                        if first_edge_type in ['nearest', 'boundary', 'boundary_intersect']:
                            path_data['first_node_special_case'] = first_edge_type
                            print(f"     🎯 First node special case: {first_edge_type}")

                        if last_edge_type in ['nearest', 'boundary', 'boundary_intersect']:
                            path_data['last_node_special_case'] = last_edge_type
                            print(f"     🎯 Last node special case: {last_edge_type}")

                else:
                    # Keep basic info if parsing fails
                    path_data = {
                        'path_nodes': shortest_path,
                        'path_length': len(shortest_path),
                        'hop_distance': path_length,
                        # 'formatted_path': ' → '.join(map(str, shortest_path)),
                        'formatted_path_enhanced': ' → '.join(map(str, shortest_path)),
                        # 'formatted_path_original': ' → '.join(map(str, shortest_path)),
                        'total_length_meters': 0,
                        'direction_changes': 0,
                        'path_triples': [],
                        'recorded_coordinates': {},
                        'coordinates_recorded_count': 0,
                        'parsing_successful': False,
                        'coordinate_recording_applied': False,
                        'parsing_error': parsed_result.get('error', 'Unknown error')
                    }
                    print(f"  ⚠️ Coordinate recording parsing failed for {dest_node}, using basic format")

            except Exception as parse_error:
                print(f"  ⚠️ Error in coordinate recording parsing for path to {dest_node}: {parse_error}")
                # Still keep basic path info
                path_data = {
                    'path_nodes': shortest_path,
                    'path_length': len(shortest_path),
                    'hop_distance': path_length,
                    # 'formatted_path': ' → '.join(map(str, shortest_path)),
                    'formatted_path_enhanced': ' → '.join(map(str, shortest_path)),
                    # 'formatted_path_original': ' → '.join(map(str, shortest_path)),
                    'total_length_meters': 0,
                    'direction_changes': 0,
                    'path_triples': [],
                    'recorded_coordinates': {},
                    'coordinates_recorded_count': 0,
                    'parsing_successful': False,
                    'coordinate_recording_applied': False,
                    'parsing_error': str(parse_error)
                }
                print(f"  ⚠️ Basic path to {dest_node} recorded (coordinate recording parsing failed)")

            shortest_paths[dest_node] = path_data

        except nx.NetworkXNoPath:
            print(f"  ❌ No path exists to {dest_node}")
            continue
        except Exception as e:
            print(f"  ❌ Error finding path to {dest_node}: {e}")
            continue

    print(f"📊 Successfully found shortest paths to {len(shortest_paths)}/{len(destination_nodes)} destinations")
    print(f"📊 Total paths recorded: {len(shortest_paths)} (exactly 1 per destination)")

    # Enhanced debug information for coordinate recording
    successful_coordinate_recording = sum(1 for path_data in shortest_paths.values()
                                          if path_data.get('coordinate_recording_applied', False))
    first_node_special_cases = sum(1 for path_data in shortest_paths.values()
                                   if 'first_node_special_case' in path_data)
    last_node_special_cases = sum(1 for path_data in shortest_paths.values()
                                  if 'last_node_special_case' in path_data)

    # print(f"🔍 DEBUG - Coordinate recording applied to {successful_coordinate_recording}/{len(shortest_paths)} paths")
    # print(f"🔍 DEBUG - First node special cases: {first_node_special_cases}")
    # print(f"🔍 DEBUG - Last node special cases: {last_node_special_cases}")
    # print(f"🔍 DEBUG - Recorded shortest_paths keys: {list(shortest_paths.keys())}")

    return shortest_paths


def find_distant_nodes_and_route_random_walk(subgraph_result, nodes_gdf, here_api_key,
                                             min_hop_distance=3, profile="pedestrian",
                                             max_candidates=5, distance_method='geodesic',
                                             auto_switch_distance_km=8.0, driving_profile="car",
                                             category=None, filter='similarity', diversity_config=None, max_paths=5,
                                             language='en', max_walks=100, max_walk_length=20):
    """
    Find distant nodes and routes using random walk approach.

    This function uses random walk algorithm to find paths between nodes instead of
    traditional shortest path algorithms. It returns the same format as
    find_distant_nodes_and_route_enhanced for compatibility with process_paths_with_gpu_optimization.

    Args:
        subgraph_result: Dictionary containing subgraph information
        nodes_gdf: GeoDataFrame containing node data
        here_api_key: HERE API key for routing
        min_hop_distance: Minimum hop distance required
        profile: Routing profile ('pedestrian', 'car', etc.)
        max_candidates: Maximum number of candidate pairs to consider
        distance_method: Method for calculating distances ('geodesic', 'euclidean')
        auto_switch_distance_km: Distance threshold for switching routing profiles
        driving_profile: Alternative routing profile for long distances
        category: Optional category filter for nodes
        filter: Filter method for path selection
        diversity_config: Configuration for path diversity
        max_paths: Maximum number of paths to find
        language: Language for route instructions
        max_walks: Maximum number of random walks to attempt
        max_walk_length: Maximum length of each random walk

    Returns:
        Dictionary with the same format as find_distant_nodes_and_route_enhanced
    """
    import random
    import networkx as nx
    from geopy.distance import geodesic

    print(f"🎯 Using random walk approach with max_walks={max_walks}, max_walk_length={max_walk_length}")

    if not subgraph_result.get('has_edges', True):
        print("❌ Subgraph has no edges - cannot find paths")
        return None

    if subgraph_result.get('total_edges', 0) == 0:
        print("❌ Subgraph has no edges - cannot find paths")
        return None

    # Create graph processor
    graph_processor = GraphProcessor(use_gpu=False)  # Use NetworkX for random walk
    graph_processor.nx_graph = subgraph_result['subgraph']

    all_nodes = subgraph_result['all_nodes']

    print(f"=== Finding paths using random walk ===")
    print(f"Subgraph nodes: {len(all_nodes)}")
    print(f"Subgraph edges: {graph_processor.nx_graph.number_of_edges()}")
    print(f"Min hop distance: {min_hop_distance}")
    print(f"Max paths: {max_paths}")

    # Helper function to get coordinates
    def get_coordinates(geom):
        if geom.geom_type == 'Point':
            return geom.x, geom.y
        else:
            centroid = geom.centroid
            return centroid.x, centroid.y

    # Helper function to calculate distance between nodes
    # Cache node coordinates to avoid repeated lookups
    node_coord_cache = {}
    node_name_cache = {}

    def get_node_name(node_id):
        """Get the display name of a node, with caching"""
        if node_id not in node_name_cache:
            node_info = get_node_info_from_gdf(node_id, nodes_gdf)
            if node_info:
                # Try different name fields that might exist
                name = (node_info.get('name') or
                        node_info.get('display_name') or
                        node_info.get('label') or
                        str(node_id))
                node_name_cache[node_id] = name
            else:
                node_name_cache[node_id] = str(node_id)
        return node_name_cache[node_id]

    def calculate_distance(node1, node2):
        # Use cached coordinates if available
        if node1 not in node_coord_cache:
            node1_info = get_node_info_from_gdf(node1, nodes_gdf)
            if not node1_info or 'geometry' not in node1_info:
                node_coord_cache[node1] = None
            else:
                coord1 = get_coordinates(node1_info['geometry'])
                node_coord_cache[node1] = coord1 if coord1 else None

        if node2 not in node_coord_cache:
            node2_info = get_node_info_from_gdf(node2, nodes_gdf)
            if not node2_info or 'geometry' not in node2_info:
                node_coord_cache[node2] = None
            else:
                coord2 = get_coordinates(node2_info['geometry'])
                node_coord_cache[node2] = coord2 if coord2 else None

        coord1 = node_coord_cache[node1]
        coord2 = node_coord_cache[node2]

        if not coord1 or not coord2:
            return float('inf')

        if distance_method == 'geodesic':
            return geodesic(coord1[::-1], coord2[::-1]).kilometers
        else:
            return ((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2) ** 0.5 * 111  # Approximate km

    # Random walk function
    def random_walk_to_target(graph, start_node, target_node, max_length):
        """Perform random walk from start_node to target_node"""
        path = [start_node]
        current = start_node

        for _ in range(max_length):
            if current == target_node:
                return path

            # Get neighbors
            neighbors = list(graph.neighbors(current))
            if not neighbors:
                break

            # Randomly select next node
            next_node = random.choice(neighbors)
            if next_node in path:  # Avoid cycles
                continue

            path.append(next_node)
            current = next_node

        return path if current == target_node else None

    # Find diverse paths using random walk for the same source-target pair
    def find_random_walk_paths(graph, source_id, target_id, min_hops, max_paths, max_walks=5):
        """Find diverse paths using random walk between the same source-target pair"""
        paths = []
        attempts = 0

        while len(paths) < max_paths and attempts < max_walks:
            # Perform random walk between the same source and target
            path = random_walk_to_target(graph, source_id, target_id, max_walk_length)

            if path and len(path) - 1 >= min_hops:
                # Check if path is diverse enough from existing paths
                is_diverse = True
                for existing_path in paths:
                    if len(set(path) & set(existing_path['nodes'])) / len(
                            set(path) | set(existing_path['nodes'])) > 0.7:
                        is_diverse = False
                        break

                if is_diverse:
                    path_info = {
                        'path_id': len(paths),
                        'nodes': path,
                        'length': len(path) - 1,
                        'distance': sum(graph[path[i]][path[i + 1]].get('weight', 1.0)
                                        for i in range(len(path) - 1)),
                        'has_required_category': category is None,
                        'is_shortest_path': False
                    }
                    paths.append(path_info)
                    print(f"✅ Found random walk path {len(paths)}: {len(path)} nodes, {len(path) - 1} hops")

            attempts += 1

        return paths

    # Get node coordinates for HERE API (reuse the cache we built)
    node_coordinates = {}
    for node_id in all_nodes:
        if node_id in node_coord_cache and node_coord_cache[node_id] is not None:
            node_coordinates[node_id] = node_coord_cache[node_id]
        else:
            # Fallback for nodes not in cache
            node_info = get_node_info_from_gdf(node_id, nodes_gdf)
            if node_info and 'geometry' in node_info:
                coord = get_coordinates(node_info['geometry'])
                node_coordinates[node_id] = coord

    # Find candidate pairs with minimum hop distance
    candidate_pairs = []
    graph = graph_processor.nx_graph

    # Calculate all pairs shortest path lengths
    try:
        all_pairs_lengths = dict(nx.all_pairs_shortest_path_length(graph))
    except nx.NetworkXError:
        print("❌ Error calculating shortest path lengths")
        return None

    # Optimized: Use itertools.combinations to avoid self-pairs and duplicates
    from itertools import combinations

    # Pre-filter nodes that have valid coordinates to avoid unnecessary calculations
    valid_nodes = []
    for node_id in all_nodes:
        node_info = get_node_info_from_gdf(node_id, nodes_gdf)
        if node_info and 'geometry' in node_info:
            coord = get_coordinates(node_info['geometry'])
            if coord:  # Only include nodes with valid coordinates
                valid_nodes.append(node_id)

    print(f"📊 Found {len(valid_nodes)} nodes with valid coordinates out of {len(all_nodes)} total nodes")

    # Use combinations to generate unique pairs efficiently
    for node1, node2 in combinations(valid_nodes, 2):
        # Filter out pairs with the same node name
        node1_name = get_node_name(node1)
        node2_name = get_node_name(node2)

        if node1_name == node2_name:
            continue  # Skip pairs with same name

        # Optional: Add hop distance filtering if needed
        # hop_distance = all_pairs_lengths.get(node1, {}).get(node2, float('inf'))
        # if hop_distance < min_hop_distance:
        #     continue

        geometric_distance = calculate_distance(node1, node2)
        if geometric_distance != float('inf'):  # Only add valid distances
            candidate_pairs.append({
                'node1': node1,
                'node2': node2,
                'node1_name': node1_name,
                'node2_name': node2_name,
                'geometric_distance_km': geometric_distance
            })

    print(f"🎯 Generated {len(candidate_pairs)} candidate pairs (filtered out pairs with same node names)")

    if not candidate_pairs:
        print("❌ No candidate pairs found after filtering (same names or invalid distances)")
        return None

    # Limit candidate pairs to avoid excessive computation
    max_candidate_pairs = 1000  # Adjust based on your needs
    if len(candidate_pairs) > max_candidate_pairs:
        print(f"⚠️ Limiting candidate pairs from {len(candidate_pairs)} to {max_candidate_pairs}")
        # Sort by distance and take the top pairs (farthest first)
        candidate_pairs.sort(key=lambda x: x['geometric_distance_km'], reverse=True)
        candidate_pairs = candidate_pairs[:max_candidate_pairs]

    # Sort by geometric distance (descending) to find farthest pairs
    candidate_pairs.sort(key=lambda x: x['geometric_distance_km'], reverse=True)

    # Select the farthest pair
    selected_pair = candidate_pairs[0]
    source_id = selected_pair['node1']
    target_id = selected_pair['node2']

    print(
        f"🎯 Selected pair: {source_id} ({selected_pair['node1_name']}) -> {target_id} ({selected_pair['node2_name']})")
    # print(f"   Hop distance: {selected_pair['hop_distance']}")  # Commented out since we're not using hop distance
    print(f"   Geometric distance: {selected_pair['geometric_distance_km']:.2f} km")

    # Find paths using random walk for the selected source-target pair
    # If we have category filtering, ensure both source and target have the required category
    if category:
        source_info = get_node_info_from_gdf(source_id, nodes_gdf)
        target_info = get_node_info_from_gdf(target_id, nodes_gdf)

        if not source_info or source_info.get('category') != category:
            print(f"⚠️ Source node {source_id} doesn't have required category '{category}'")
            return None

        if not target_info or target_info.get('category') != category:
            print(f"⚠️ Target node {target_id} doesn't have required category '{category}'")
            return None

        print(f"📋 Category filtering: Both source and target have category '{category}'")

    # Find diverse paths using random walk between the same source-target pair
    all_paths_info = {
        'paths': find_random_walk_paths(graph, source_id, target_id, min_hop_distance, max_paths, max_walks),
        'total_paths_found': 0,
        'method': 'random_walk'
    }

    all_paths_info['total_paths_found'] = len(all_paths_info['paths'])

    print(f"🎲 Found {all_paths_info['total_paths_found']} random walk paths")

    # Get shortest path for comparison
    try:
        shortest_path = nx.shortest_path(graph, source=source_id, target=target_id)
    except nx.NetworkXNoPath:
        print("❌ No shortest path found between selected nodes")
        shortest_path = None

    # Get coordinates for HERE API
    source_coord = node_coordinates.get(source_id)
    target_coord = node_coordinates.get(target_id)

    if not source_coord or not target_coord:
        print("❌ Could not get coordinates for HERE API")
        print('node_coordinates:', node_coordinates)
        print("source_id:", source_id)
        return None

    source_lat, source_lon = source_coord
    target_lat, target_lon = target_coord

    # Determine routing profile
    selected_distance_km = selected_pair['geometric_distance_km']
    profile_switched = False
    routing_profile = profile

    if selected_distance_km > auto_switch_distance_km and profile.lower() == "pedestrian":
        routing_profile = driving_profile
        profile_switched = True
        print(f"🔄 Switching from '{profile}' to '{driving_profile}' for long distance")

    # Call HERE API
    print(f"🌐 Calling HERE API for route...")
    here_response = get_here_route(
        source_lat, source_lon, target_lat, target_lon, here_api_key, routing_profile, language=language
    )

    # Create result in the same format as find_distant_nodes_and_route_enhanced
    result = {
        'origin_node': source_id,
        'destination_node': target_id,
        'hop_distance': None,  # Random walk doesn't use hop distance filtering
        'geometric_distance_km': selected_pair['geometric_distance_km'],
        'routing_profile_used': routing_profile,
        'profile_switched': profile_switched,
        'routes': here_response,
        'shortest_path': shortest_path,
        'all_paths': all_paths_info,
        'gpu_acceleration_used': False,  # Random walk doesn't use GPU
        'diversity_method_used': 'random_walk',
        'max_paths_requested': max_paths,
        'paths_found': all_paths_info.get('total_paths_found', 0)
    }

    # Add category filtering statistics if category was specified
    if category:
        result['category_filter'] = {
            'required_category': category,
            'candidate_pairs_with_category': len(candidate_pairs),
            'category_filtering_applied': True
        }

    # Add random walk statistics
    result['random_walk_stats'] = {
        'max_walks': max_walks,
        'max_walk_length': max_walk_length,
        'min_hop_distance': min_hop_distance,
        'method': 'random_walk'
    }

    print(f"✅ Found {result['paths_found']} random walk paths successfully")

    return result


def calculate_nearest_direction_enhanced(source_info, target_info, edge_info):
    """
    Enhanced nearest direction calculation for intersection-street relationships

    Rules:
    - Intersection node (Point) -> Street node (LineString/MultiLineString) = "nearest"
    - Street node (LineString/MultiLineString) -> Intersection node (Point) = "near"
    - Calculate bearing from intersection to nearest point on street
    - Calculate distance from intersection to nearest point on street
    """
    # print(f"🔍 calculate_nearest_direction_enhanced debug:")

    # Get geometry types
    source_geometry = source_info.get('geometry')
    target_geometry = target_info.get('geometry')

    source_is_point = is_point_geometry_original(source_geometry)
    target_is_point = is_point_geometry_original(target_geometry)

    # print(f"   - Source is point: {source_is_point}")
    # print(f"   - Target is point: {target_is_point}")
    # print(f"   - Source geometry type: {type(source_geometry).__name__ if source_geometry else 'None'}")
    # print(f"   - Target geometry type: {type(target_geometry).__name__ if target_geometry else 'None'}")

    # Determine the relationship
    if source_is_point and not target_is_point:
        # Intersection -> Street: "nearest"
        # print(f"   - Relationship: Intersection -> Street (nearest)")
        relationship = "nearest"
        intersection_geom = source_geometry
        street_geom = target_geometry
    elif target_is_point and not source_is_point:
        # Street -> Intersection: "near"
        # print(f"   - Relationship: Street -> Intersection (near)")
        relationship = "near"
        intersection_geom = target_geometry
        street_geom = source_geometry
    else:
        # Both points or both lines - fallback to original logic
        # print(f"   - Fallback: Both same geometry type")
        return calculate_nearest_direction_original(source_info, target_info, edge_info)

    # Calculate bearing and distance from intersection to nearest point on street
    try:
        from shapely.ops import nearest_points

        # Get intersection point coordinates
        if hasattr(intersection_geom, 'x') and hasattr(intersection_geom, 'y'):
            intersection_point = intersection_geom
        else:
            intersection_point = intersection_geom.centroid

        # Find nearest point on street to intersection
        # This works for LineString, MultiLineString, Polygon, MultiPolygon
        nearest_point_on_street = nearest_points(intersection_point, street_geom)[1]

        # print(f"   - Intersection point: ({intersection_point.x:.6f}, {intersection_point.y:.6f})")
        # print(f"   - Street geometry type: {street_geom.geom_type if hasattr(street_geom, 'geom_type') else 'Unknown'}")
        # print(f"   - Nearest point on street: ({nearest_point_on_street.x:.6f}, {nearest_point_on_street.y:.6f})")

        # Calculate bearing from intersection to nearest point on street
        bearing = calculate_bearing_between_points_original(intersection_point, nearest_point_on_street)

        # For "near" relationship (Street -> Intersection), reverse the bearing
        if relationship == "near":
            bearing = calculate_reverse_bearing_original(bearing)
            print(f"   - Reversed bearing for 'near' relationship: {bearing:.1f}°")

        # Calculate distance from intersection to nearest point on street
        from geopy.distance import geodesic
        distance = geodesic(
            (intersection_point.y, intersection_point.x),
            (nearest_point_on_street.y, nearest_point_on_street.x)
        ).meters

        # If distance is 0, try to calculate a meaningful distance
        if distance < 1e-10:
            # print(f"   - Warning: Distance is 0, trying alternative calculation")
            # Use a small default distance for identical points
            distance = 1.0  # 1 meter as fallback
            # print(f"   - Using fallback distance: {distance:.1f}m")

        # print(f"   - Calculated bearing: {bearing:.1f}°")
        # print(f"   - Calculated distance: {distance:.1f}m")
        # print(f"   - Relationship: {relationship}")

        # Convert bearing to direction
        direction = convert_bearing_to_direction_original(bearing)

        # Store calculated values in edge_info for later use
        edge_info['calculated_bearing'] = bearing
        edge_info['calculated_distance'] = distance
        edge_info['relationship'] = relationship

        return direction

    except Exception as e:
        # print(f"   - Error calculating nearest direction: {e}")
        # print(f"   - Falling back to original logic")
        # print(f"   - Source info keys: {list(source_info.keys())}")
        # print(f"   - Target info keys: {list(target_info.keys())}")
        # print(f"   - Edge info keys: {list(edge_info.keys())}")
        # Fallback to original logic
        return calculate_nearest_direction_original(source_info, target_info, edge_info)


def calculate_nearest_distance_enhanced(source_info, target_info, edge_info):
    """
    Enhanced distance calculation for nearest edges between intersection and street nodes
    """
    # print(f"🔍 calculate_nearest_distance_enhanced debug:")

    # Check if we already calculated the distance
    if 'calculated_distance' in edge_info:
        distance = edge_info['calculated_distance']
        if distance != 0:
            # print(f"   - Using pre-calculated distance: {distance:.1f}m")
            return distance
        else:
            # print(f"   - Pre-calculated distance is 0, falling back to original logic")
            # Fallback to original distance calculation
            return calculate_distance_from_coordinates(source_info, target_info, edge_info)

    # print(f"   - No pre-calculated distance found, falling back to original logic")
    # Fallback to original distance calculation
    return calculate_distance_from_coordinates(source_info, target_info, edge_info)


def calculate_distance_from_coordinates(source_info, target_info, edge_info):
    """
    Fallback distance calculation from coordinates
    """
    # print(f"   - Fallback: calculating distance from coordinates")

    # Try to get coordinates from various sources
    source_coords = None
    target_coords = None

    # Method 1: Try to get coordinates from geometry objects
    source_geometry = source_info.get('geometry')
    if source_geometry:
        try:
            if hasattr(source_geometry, 'x') and hasattr(source_geometry, 'y'):
                # Point geometry
                source_coords = (source_geometry.x, source_geometry.y)
                print(f"   - Got source coords from Point geometry: {source_coords}")
            elif hasattr(source_geometry, 'centroid'):
                # Complex geometry (LineString, MultiLineString, Polygon, MultiPolygon)
                if hasattr(source_geometry, 'geom_type'):
                    geom_type = source_geometry.geom_type
                    print(f"   - Source geometry type: {geom_type}")

                    if geom_type in ['LineString', 'MultiLineString']:
                        # For lines, use the first point as approximation
                        if hasattr(source_geometry, 'coords'):
                            coords = list(source_geometry.coords)
                            if coords:
                                source_coords = (coords[0][0], coords[0][1])
                                print(f"   - Got source coords from {geom_type} first point: {source_coords}")
                        else:
                            # Fallback to centroid
                            centroid = source_geometry.centroid
                            source_coords = (centroid.x, centroid.y)
                            print(f"   - Got source coords from {geom_type} centroid: {source_coords}")
                    elif geom_type in ['Polygon', 'MultiPolygon']:
                        # For polygons, use centroid
                        centroid = source_geometry.centroid
                        source_coords = (centroid.x, centroid.y)
                        print(f"   - Got source coords from {geom_type} centroid: {source_coords}")
                    else:
                        # Unknown geometry type, use centroid
                        centroid = source_geometry.centroid
                        source_coords = (centroid.x, centroid.y)
                        print(f"   - Got source coords from unknown geometry centroid: {source_coords}")
                else:
                    # No geom_type, use centroid
                    centroid = source_geometry.centroid
                    source_coords = (centroid.x, centroid.y)
                    print(f"   - Got source coords from geometry centroid: {source_coords}")
        except Exception as e:
            print(f"   - Error getting source coords from geometry: {e}")

    target_geometry = target_info.get('geometry')
    if target_geometry:
        try:
            if hasattr(target_geometry, 'x') and hasattr(target_geometry, 'y'):
                # Point geometry
                target_coords = (target_geometry.x, target_geometry.y)
                # print(f"   - Got target coords from Point geometry: {target_coords}")
            elif hasattr(target_geometry, 'centroid'):
                # Complex geometry (LineString, MultiLineString, Polygon, MultiPolygon)
                if hasattr(target_geometry, 'geom_type'):
                    geom_type = target_geometry.geom_type
                    # print(f"   - Target geometry type: {geom_type}")

                    if geom_type in ['LineString', 'MultiLineString']:
                        # For lines, use the first point as approximation
                        if hasattr(target_geometry, 'coords'):
                            coords = list(target_geometry.coords)
                            if coords:
                                target_coords = (coords[0][0], coords[0][1])
                                print(f"   - Got target coords from {geom_type} first point: {target_coords}")
                        else:
                            # Fallback to centroid
                            centroid = target_geometry.centroid
                            target_coords = (centroid.x, centroid.y)
                            # print(f"   - Got target coords from {geom_type} centroid: {target_coords}")
                    elif geom_type in ['Polygon', 'MultiPolygon']:
                        # For polygons, use centroid
                        centroid = target_geometry.centroid
                        target_coords = (centroid.x, centroid.y)
                        # print(f"   - Got target coords from {geom_type} centroid: {target_coords}")
                    else:
                        # Unknown geometry type, use centroid
                        centroid = target_geometry.centroid
                        target_coords = (centroid.x, centroid.y)
                        # print(f"   - Got target coords from unknown geometry centroid: {target_coords}")
                else:
                    # No geom_type, use centroid
                    centroid = target_geometry.centroid
                    target_coords = (centroid.x, centroid.y)
                    # print(f"   - Got target coords from geometry centroid: {target_coords}")
        except Exception as e:
            print(f"   - Error getting target coords from geometry: {e}")

    # Method 2: Try to get coordinates from node info fields
    if source_coords is None:
        source_x = source_info.get('x')
        source_y = source_info.get('y')
        if source_x is not None and source_y is not None:
            source_coords = (source_x, source_y)
            # print(f"   - Got source coords from node info: {source_coords}")

    if target_coords is None:
        target_x = target_info.get('x')
        target_y = target_info.get('y')
        if target_x is not None and target_y is not None:
            target_coords = (target_x, target_y)
            print(f"   - Got target coords from node info: {target_coords}")

    # Calculate distance if we have both coordinates
    if source_coords and target_coords:
        try:
            from geopy.distance import geodesic
            distance = geodesic((source_coords[1], source_coords[0]),
                                (target_coords[1], target_coords[0])).meters
            # print(f"   - Calculated distance from coordinates: {distance:.1f}m")
            return distance
        except Exception as e:
            print(f"   - Error calculating distance from coordinates: {e}")

    # print(f"   - Could not calculate distance, returning 0")
    # print(f"   - Source coords: {source_coords}")
    # print(f"   - Target coords: {target_coords}")
    return 0


def determine_nearest_relationship_enhanced(source_info, target_info, edge_info):
    """
    Determine the relationship string for nearest edges based on geometry types

    Returns:
    - "nearest" if intersection -> street
    - "near" if street -> intersection
    - "nearest" as fallback
    """
    source_geometry = source_info.get('geometry')
    target_geometry = target_info.get('geometry')

    source_is_point = is_point_geometry_original(source_geometry)
    target_is_point = is_point_geometry_original(target_geometry)

    if source_is_point and not target_is_point:
        return "nearest"  # Intersection -> Street
    elif target_is_point and not source_is_point:
        return "near"  # Street -> Intersection
    else:
        return "nearest"  # Fallback


def get_and_record_nearest_coordinates_enhanced(source_info, target_info, edge_info):
    """
    Enhanced coordinate recording for nearest edges between intersection and street nodes
    """
    # print(f"🔍 get_and_record_nearest_coordinates_enhanced debug:")

    try:
        from shapely.ops import nearest_points

        # Get geometry types
        source_geometry = source_info.get('geometry')
        target_geometry = target_info.get('geometry')

        source_is_point = is_point_geometry_original(source_geometry)
        target_is_point = is_point_geometry_original(target_geometry)

        # print(f"   - Source is point: {source_is_point}")
        # print(f"   - Target is point: {target_is_point}")

        # Determine which is intersection and which is street
        if source_is_point and not target_is_point:
            # Intersection -> Street
            intersection_geom = source_geometry
            street_geom = target_geometry
            relationship = "nearest"
        elif target_is_point and not source_is_point:
            # Street -> Intersection
            intersection_geom = target_geometry
            street_geom = source_geometry
            relationship = "near"
        else:
            # Both same type - fallback to original logic
            # print(f"   - Both same geometry type, using fallback")
            return get_and_record_nearest_coordinates(source_info, target_info, edge_info)

        # Get intersection point coordinates
        if hasattr(intersection_geom, 'x') and hasattr(intersection_geom, 'y'):
            intersection_point = intersection_geom
        else:
            intersection_point = intersection_geom.centroid

        # Find nearest point on street to intersection
        nearest_point_on_street = nearest_points(intersection_point, street_geom)[1]

        # print(f"   - Intersection point: ({intersection_point.x:.6f}, {intersection_point.y:.6f})")
        # print(f"   - Nearest point on street: ({nearest_point_on_street.x:.6f}, {nearest_point_on_street.y:.6f})")
        # print(f"   - Relationship: {relationship}")

        # Return the nearest point on street coordinates
        return nearest_point_on_street.x, nearest_point_on_street.y, f'nearest_point_{relationship}'

    except Exception as e:
        print(f"   - Error in enhanced coordinate recording: {e}")
        # Fallback to original logic
        return get_and_record_nearest_coordinates(source_info, target_info, edge_info)


def calculate_transition_between_intersections(current_triple, next_triple, recorded_coordinates):
    """
    Calculate transition distance and direction between two consecutive nearest/near triples.
    This handles the case: (intersection1, nearest, street) -> (street, near, intersection2)

    Args:
        current_triple: Current triple with nearest/near edge
        next_triple: Next triple with nearest/near edge
        recorded_coordinates: Dictionary of recorded coordinates

    Returns:
        tuple: (transition_distance, transition_direction) or (None, None) if cannot calculate
    """
    try:
        print(f"🔍 calculate_transition_between_intersections debug:")
        print(f"   - Current triple: {current_triple.get('source_name')} -> {current_triple.get('target_name')}")
        print(f"   - Next triple: {next_triple.get('source_name')} -> {next_triple.get('target_name')}")
        print(
            f"   - Recorded coordinates keys: {list(recorded_coordinates.keys()) if recorded_coordinates else 'None'}")

        # Get coordinates from current and next triples
        current_coords = None
        next_coords = None

        # Try to get coordinates from recorded_coordinates first
        if recorded_coordinates:
            current_pos = current_triple.get('position', 0)
            next_pos = next_triple.get('position', current_pos + 1)

            # print(f"   - Looking for positions: current={current_pos}, next={next_pos}")

            if current_pos in recorded_coordinates:
                current_coords = recorded_coordinates[current_pos][:2]  # (x, y)
                # print(f"   - Found current coords: {current_coords}")
            else:
                print(f"   - Current position {current_pos} not found in recorded_coordinates")

            if next_pos in recorded_coordinates:
                next_coords = recorded_coordinates[next_pos][:2]  # (x, y)
                # print(f"   - Found next coords: {next_coords}")
            else:
                print(f"   - Next position {next_pos} not found in recorded_coordinates")

        # If we don't have coordinates from recorded_coordinates, try to get them from triple info
        if not current_coords:
            current_source_info = current_triple.get('source_info', {})
            current_target_info = current_triple.get('target_info', {})

            # Try to get coordinates from source or target info
            if 'geometry' in current_source_info:
                geom = current_source_info['geometry']
                if hasattr(geom, 'x') and hasattr(geom, 'y'):
                    current_coords = (geom.x, geom.y)
                    print(f"   - Got current coords from source geometry: {current_coords}")
            elif 'geometry' in current_target_info:
                geom = current_target_info['geometry']
                if hasattr(geom, 'x') and hasattr(geom, 'y'):
                    current_coords = (geom.x, geom.y)
                    print(f"   - Got current coords from target geometry: {current_coords}")

        if not next_coords:
            next_source_info = next_triple.get('source_info', {})
            next_target_info = next_triple.get('target_info', {})

            # Try to get coordinates from source or target info
            if 'geometry' in next_source_info:
                geom = next_source_info['geometry']
                if hasattr(geom, 'x') and hasattr(geom, 'y'):
                    next_coords = (geom.x, geom.y)
                    print(f"   - Got next coords from source geometry: {next_coords}")
            elif 'geometry' in next_target_info:
                geom = next_target_info['geometry']
                if hasattr(geom, 'x') and hasattr(geom, 'y'):
                    next_coords = (geom.x, geom.y)
                    print(f"   - Got next coords from target geometry: {next_coords}")

        # If we have both coordinates, calculate transition
        if current_coords and next_coords:
            from geopy.distance import geodesic

            # Calculate distance
            distance = geodesic(
                (current_coords[1], current_coords[0]),  # (lat, lon)
                (next_coords[1], next_coords[0])  # (lat, lon)
            ).meters

            # Calculate bearing
            bearing = calculate_bearing_original(
                current_coords[0], current_coords[1],  # (lon, lat)
                next_coords[0], next_coords[1]  # (lon, lat)
            )

            # Convert bearing to direction
            # from generate_Here_routes1 import convert_bearing_to_direction_original
            direction = convert_bearing_to_direction_original(bearing)

            # print(f"🔍 Calculated transition: {distance:.1f}m, {direction}")
            return distance, direction

        print(f"⚠️ Could not calculate transition between triples")
        return None, None

    except Exception as e:
        print(f"❌ Error calculating transition between intersections: {e}")
        return None, None


def format_enhanced_path_with_coordinates_improved(triples, recorded_coordinates):
    """
    Enhanced path formatting with proper handling of consecutive nearest/near triples

    Expected format for consecutive nearest/near triples:
    "(intersection1, nearest, street) (150.2m, 45°(NE)) -> (200.5m, 90°(E)) -> (street, near, intersection2) (180.3m, 225°(SW))"

    Where:
    - First tuple: distance/direction from intersection1 to nearest point on street
    - Middle tuple: distance/direction from current position to next position (transition)
    - Last tuple: distance/direction from street to intersection2
    """
    if not triples:
        return ""

    path_parts = []

    for i, triple in enumerate(triples):
        is_first = (i == 0)
        is_last = (i == len(triples) - 1)
        # Check if this is a nearest/near/bounds/intersects type edge
        # These edges should have distance/direction tuples attached to them
        edge_type = triple.get('edge_type', '')
        middle_component = triple.get('middle_component', '')
        is_nearest_type = (edge_type in ['nearest', 'near', 'boundary', 'boundary_intersects', 'boundary_crossing'] or
                           middle_component in ['nearest', 'near', 'bounds', 'intersects'])

        # Debug: Print edge type for each triple
        # print(f"🔍 Triple {i}: {triple.get('source_name', 'Unknown')} -> {triple.get('target_name', 'Unknown')}")
        # print(f"   Edge type: {triple.get('edge_type', 'Unknown')}")
        # print(f"   Is nearest type: {is_nearest_type}")
        # print(f"   Length: {triple.get('length', 'Unknown')}")
        # print(f"   Direction: {triple.get('direction', 'Unknown')}")
        # if is_last:
        #     print(f"   ⭐ This is the LAST triple")

        # Create the basic triple part
        source_name = triple.get('source_name', 'Unknown')
        middle_name = triple.get('middle_component', 'Unknown')
        target_name = triple.get('target_name', 'Unknown')

        triple_part = f"({source_name}, {middle_name}, {target_name})"

        # Get direction/distance info (edge_type and middle_component already extracted above)
        length = triple.get('length')
        direction = triple.get('direction')

        # SPECIAL CASE 1: First node with nearest/boundary edge
        if is_first and is_nearest_type:
            # print(f"case1 - First node with nearest/boundary edge")
            # print(f"   - Triple: {triple_part}")
            # print(f"   - Length: {length}, Direction: {direction}")

            # Enhanced fallback for first triple too
            if not length or not direction:
                edge_info = triple.get('edge_info', {})
                if not length:
                    length = edge_info.get('length') or edge_info.get('distance') or edge_info.get(
                        'calculated_distance')
                if not direction:
                    direction = edge_info.get('direction') or edge_info.get('calculated_bearing')
                    if direction and isinstance(direction, (int, float)):
                        # from generate_Here_routes1 import convert_bearing_to_direction_original
                        direction = convert_bearing_to_direction_original(direction)

            direction_tuple = format_direction_distance_tuple(length, direction)
            path_parts.append(f"{triple_part} {direction_tuple}")

            # Add arrow for continuation (if not the only triple)
            if not is_last:
                # Calculate transition to next triple
                next_triple = triples[i + 1]
                # print(f"   - Calculating transition to next triple: {next_triple.get('source_name')} -> {next_triple.get('target_name')}")
                transition_distance, transition_direction = calculate_transition_between_intersections(
                    triple, next_triple, recorded_coordinates
                )

                if transition_distance and transition_direction:
                    # Add: -> transition_tuple ->
                    transition_tuple = format_direction_distance_tuple(transition_distance, transition_direction)
                    path_parts.append("->")
                    path_parts.append(transition_tuple)
                    path_parts.append("->")
                    # print(f"   - Added transition: {transition_tuple}")
                else:
                    # No transition, just add single arrow
                    # print(f"   - No transition calculated")
                    path_parts.append("->")

        # SPECIAL CASE 2: Last node with nearest/boundary edge
        elif is_last and is_nearest_type:
            # print(f"🔍 Processing last triple with nearest/near edge: {triple_part}")
            # print(f"   Edge type: {triple.get('edge_type', 'Unknown')}")
            # print(f"   Length: {length}, Direction: {direction}")
            # print(f"   Recorded coordinates available: {len(recorded_coordinates) if recorded_coordinates else 0}")
            #
            # # Enhanced debugging to see what's in the triple
            # print(f"   🔍 Triple raw data:")
            # print(f"      Length from triple: {triple.get('length')}")
            # print(f"      Direction from triple: {triple.get('direction')}")
            # print(f"      Edge info available: {triple.get('edge_info', {})}")
            if 'edge_info' in triple:
                edge_info = triple['edge_info']
                # print(f"      Edge info length: {edge_info.get('length')}")
                # print(f"      Edge info distance: {edge_info.get('distance')}")
                # print(f"      Edge info calculated_distance: {edge_info.get('calculated_distance')}")
                # print(f"      Edge info direction: {edge_info.get('direction')}")
                # print(f"      Edge info calculated_bearing: {edge_info.get('calculated_bearing')}")

            # For the last nearest-type triple, we don't add any transition tuple before it
            # because the previous triple (whether normal or nearest-type) would have already
            # handled the transition. We just add the triple with its own direction/distance tuple.
            
            # Add the final triple with its own direction/distance tuple
            if length and direction:
                direction_tuple = format_direction_distance_tuple(length, direction)
                path_parts.append(f"{triple_part} {direction_tuple}")
                print(f"   ✅ Added direction tuple: {direction_tuple}")
            else:
                print(f"   ⚠️ Missing length or direction for last triple")
                print(f"   Length: {length}, Direction: {direction}")

                # Enhanced fallback: try multiple sources from edge_info
                edge_info = triple.get('edge_info', {})
                print(f"   🔍 Attempting enhanced fallback from edge_info...")

                # Try multiple length sources
                fallback_length = (edge_info.get('length') or
                                   edge_info.get('distance') or
                                   edge_info.get('calculated_distance'))

                # Try multiple direction sources
                fallback_direction = edge_info.get('direction')
                if not fallback_direction:
                    # Try calculated_bearing and convert to direction
                    calculated_bearing = edge_info.get('calculated_bearing')
                    if calculated_bearing is not None:
                        # from generate_Here_routes1 import convert_bearing_to_direction_original
                        fallback_direction = convert_bearing_to_direction_original(calculated_bearing)
                    else:
                        # As a fallback, calculate direction and distance using geometry
                        try:
                            # from generate_Here_routes1 import calculate_nearest_direction_original
                            source_info = triple.get('source_info', {})
                            target_info = triple.get('target_info', {})
                            calculated_direction, calculated_distance = calculate_nearest_direction_original(
                                source_info, target_info, edge_info
                            )
                            if not fallback_direction and calculated_direction:
                                fallback_direction = calculated_direction
                            if not fallback_length and calculated_distance:
                                fallback_length = calculated_distance
                        except Exception as _e:
                            # Keep silent fallback if calculation fails
                            pass
                #         print(
                #             f"   🔄 Converted calculated_bearing {calculated_bearing} to direction: {fallback_direction}")
                #
                # print(f"   🔍 Fallback values - Length: {fallback_length}, Direction: {fallback_direction}")

                if fallback_length and fallback_direction:
                    fallback_tuple = format_direction_distance_tuple(fallback_length, fallback_direction)
                    path_parts.append(f"{triple_part} {fallback_tuple}")
                    print(f"   🔄 Used enhanced fallback: {fallback_tuple}")
                else:
                    path_parts.append(triple_part)
                    # print(f"   ❌ No fallback available, added triple without direction")

        # SPECIAL CASE 3: Middle node with nearest/boundary edge
        elif is_nearest_type and not is_first and not is_last:
            # print(f"case3")
            # Check if previous and next triples are also nearest/near type
            prev_triple = triples[i - 1]
            next_triple = triples[i + 1]
            # Use the same logic as is_nearest_type to check prev and next
            prev_edge_type = prev_triple.get('edge_type', '')
            prev_middle = prev_triple.get('middle_component', '')
            prev_is_nearest = (prev_edge_type in ['nearest', 'near', 'boundary', 'boundary_intersects', 'boundary_crossing'] or
                              prev_middle in ['nearest', 'near', 'bounds', 'intersects'])
            next_edge_type = next_triple.get('edge_type', '')
            next_middle = next_triple.get('middle_component', '')
            next_is_nearest = (next_edge_type in ['nearest', 'near', 'boundary', 'boundary_intersects', 'boundary_crossing'] or
                              next_middle in ['nearest', 'near', 'bounds', 'intersects'])

            # If we have consecutive nearest/near triples, calculate transition
            if prev_is_nearest or next_is_nearest:
                # print(f"🔍 Processing middle nearest triple with consecutive nearest/near triples")

                # Enhanced fallback for middle triple too
                if not length or not direction:
                    edge_info = triple.get('edge_info', {})
                    if not length:
                        length = edge_info.get('length') or edge_info.get('distance') or edge_info.get(
                            'calculated_distance')
                    if not direction:
                        direction = edge_info.get('direction') or edge_info.get('calculated_bearing')
                        if direction and isinstance(direction, (int, float)):
                            # from generate_Here_routes1 import convert_bearing_to_direction_original
                            direction = convert_bearing_to_direction_original(direction)

                # Add the triple with its own direction/distance tuple
                direction_tuple = format_direction_distance_tuple(length, direction)
                path_parts.append(f"{triple_part} {direction_tuple}")

                # Calculate and add transition to next triple
                transition_distance, transition_direction = calculate_transition_between_intersections(
                    triple, next_triple, recorded_coordinates
                )

                if transition_direction and transition_distance:
                    transition_tuple = format_direction_distance_tuple(transition_distance, transition_direction)
                    path_parts.append("->")
                    path_parts.append(transition_tuple)
                    path_parts.append("->")
                else:
                    # Fallback to normal arrow
                    path_parts.append("->")
            else:
                # Normal middle nearest triple

                # Enhanced fallback for normal middle triple too
                if not length or not direction:
                    edge_info = triple.get('edge_info', {})
                    if not length:
                        length = edge_info.get('length') or edge_info.get('distance') or edge_info.get(
                            'calculated_distance')
                    if not direction:
                        direction = edge_info.get('direction') or edge_info.get('calculated_bearing')
                        if direction and isinstance(direction, (int, float)):
                            # from generate_Here_routes1 import convert_bearing_to_direction_original
                            direction = convert_bearing_to_direction_original(direction)

                direction_tuple = format_direction_distance_tuple(length, direction)
                path_parts.append(f"{triple_part} {direction_tuple}")
                path_parts.append("->")

        # NORMAL CASE: Middle nodes without nearest/boundary or first/last without nearest/boundary
        else:
            # print(f"case4")
            path_parts.append(triple_part)

            # For normal triples, only add transition (not the triple's own direction/distance)
            if not is_last:
                # Calculate transition to next triple
                next_triple = triples[i + 1]
                transition_distance, transition_direction = calculate_transition_between_intersections(
                    triple, next_triple, recorded_coordinates
                )

                if transition_distance and transition_direction:
                    transition_tuple = format_direction_distance_tuple(transition_distance, transition_direction)
                    path_parts.append("->")
                    path_parts.append(transition_tuple)
                    path_parts.append("->")
                else:
                    # Fallback to normal arrow
                    path_parts.append("->")
            else:
                # This is the last triple but not a nearest/near type
                # Still try to add distance/direction if available
                if length and direction:
                    direction_tuple = format_direction_distance_tuple(length, direction)
                    path_parts.append(f" {direction_tuple}")
                    print(f"🔍 Added direction tuple to last non-nearest triple: {direction_tuple}")
                else:
                    print(f"🔍 Last triple has no distance/direction info: {triple_part}")

    # Join and clean up formatting
    result = " ".join(path_parts)

    # Clean up multiple consecutive arrows and extra spaces
    import re
    result = re.sub(r'\s*->\s*->\s*', ' -> ', result)
    result = re.sub(r'\s+', ' ', result)

    return result.strip()