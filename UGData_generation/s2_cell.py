import geopandas as gpd
import pandas as pd
import s2sphere
import numpy as np
import os
import json
from shapely.geometry import Point, LineString, Polygon, MultiPoint, MultiLineString, MultiPolygon

# points = gpd.read_file("./data/geo/points.geojson")
# polygons = gpd.read_file("./data/geo/polygons.geojson")
# lines = gpd.read_file("./data/geo/all_lines.geojson")
# nodes = pd.concat([points,polygons,lines])
# nodes = gpd.read_file("data/geo/NewYorkWhole/nodes1.geojson")

def get_centroid(geom):
    """
    Safely get centroid for different geometry types
    """
    try:
        # If it's already a point, return it
        if isinstance(geom, Point):
            return geom

        # Handle different geometry types
        if isinstance(geom, (LineString, MultiLineString)):
            return geom.centroid

        if isinstance(geom, (Polygon, MultiPolygon)):
            return geom.centroid

        if isinstance(geom, MultiPoint):
            # For MultiPoint, use the centroid of all points
            return geom.centroid

        # If nothing else works, try to get the centroid
        return geom.centroid

    except Exception as e:
        print(f"Error getting centroid: {e}")
        return None


def split_nodes_by_s2_cells(nodes_gdf, level=24, output_dir='./s2_cell_splits'):
    """
    Split nodes into S2 cells with improved error handling for various geometry types
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Ensure the GDF is in WGS84
    if nodes_gdf.crs != "EPSG:4326":
        nodes_gdf = nodes_gdf.to_crs("EPSG:4326")

    # Initialize cell dictionary
    cells_dict = {}

    # Process each node
    for _, row in nodes_gdf.iterrows():
        try:
            # Get geometry
            geom = row.geometry

            # Skip invalid geometries
            if geom is None or geom.is_empty:
                continue

            # Get centroid
            centroid = get_centroid(geom)

            # Skip if no valid centroid
            if centroid is None:
                continue

            # Get coordinates from centroid
            lat, lon = centroid.y, centroid.x

            # Get S2 cell for this point
            cell = s2sphere.CellId.from_lat_lng(s2sphere.LatLng.from_degrees(lat, lon)).parent(level)
            cell_id = str(cell.id())

            # Add to cells dictionary
            if cell_id not in cells_dict:
                cells_dict[cell_id] = []
            cells_dict[cell_id].append(row['id'])

        except Exception as e:
            print(f"Error processing node {row.get('id', 'Unknown ID')}: {e}")
            continue

    # Save results
    _save_cell_splits(cells_dict, output_dir)

    return cells_dict


def _save_cell_splits(cells_dict, output_dir):
    """
    Save cell splits with metadata
    """
    # Metadata about the split
    metadata = {
        'total_cells': len(cells_dict),
        'total_features': sum(len(ids) for ids in cells_dict.values()),
    }

    # Save metadata
    with open(os.path.join(output_dir, 'cell_split_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    # Save cell splits as JSON
    with open(os.path.join(output_dir, 'cell_splits.json'), 'w') as f:
        json.dump(cells_dict, f)

    # Print summary
    print(f"Total number of S2 cells: {len(cells_dict)}")
    print("Top 10 cells by number of nodes:")
    sorted_cells = sorted(cells_dict.items(), key=lambda x: len(x[1]), reverse=True)
    for cell, nodes_list in sorted_cells[:10]:
        print(f"Cell {cell}: {len(nodes_list)} nodes")

import argparse
import pandas as pd
import pickle
def main():

    parser = argparse.ArgumentParser(description="Merge OSM batch Parquet files into a single file.")
    parser.add_argument("--place",default='ilede', help="Directory containing the batch Parquet files.")
    parser.add_argument("--level",default='18',help="s2_cell level")

    args = parser.parse_args()
    """Reading parquet data"""
    input_folder = f"./data/geo/SR/osm_data/{args.place}"
    # Usage
    # nodes =  gpd.read_file(f'./data/geo/SR/osm_data/{args.place}/nodes_with_districts.geojson')
    # mapillary_nodes = gpd.read_file(f'./data/geo/SR/osm_data/{args.place}/nodes_mapillary_with_districts.geojson', driver = 'GeoJSON')
    nodes = gpd.read_file(f'./data/geo/SR/osm_data/{args.place}/nodes.geojson')
    mapillary_nodes = gpd.read_file(f'./data/geo/SR/osm_data/{args.place}/nodes_mapillary.geojson',
                                    driver='GeoJSON')
    nodes = pd.concat([nodes, mapillary_nodes])
    # print(type(mapillary_nodes['id'].iloc[0])==str)
    # if os.path.exists(input_folder+'/nodes_all.geojson'):
    #     nodes=  gpd.read_file(input_folder+'/nodes_all.geojson')
    # if os.path.exists(input_folder+'/nodes.pkl'):
    #     with open(input_folder+'/nodes.pkl', 'rb') as f:
    #         nodes = pickle.load(f)

    # streets = gpd.read_file('/root/lanyun-fs/UrbanKG/data/geo/SR/spatial/lines_with_directions.geojson')
    # nodes = pd.concat([poi_aois_crossings,streets])
    # nodes.to_file('/root/lanyun-fs/UrbanKG/data/geo/SR/spatial/nodes1.geojson')
    try:
        # Split nodes into S2 cells
        print("Splitting nodes into S2 cells...")
        # Convert level to integer
        level_int = int(args.level)
        cell_to_nodes = split_nodes_by_s2_cells(nodes, level=level_int)

    except Exception as e:
        print(f"Error splitting nodes into S2 cells: {e}")
        import traceback

        traceback.print_exc()
    # with open("./data/geo/SR/s2cell2nodes.json", "w") as json_file:
    #     json.dump(cell_to_nodes, json_file)
    # print(cell_to_nodes)
    with open(input_folder+f"/s2cell2nodes_{args.level}_mapillary.json", "w") as json_file:

        # with open(f"/root/lanyun-fs/UrbanKG/data/geo/SR/spatial/s2cell2nodes_{level}.json", "w") as json_file:
        json.dump(cell_to_nodes, json_file)
    # print(len(cell_to_nodes.keys()))
    print('Finished')


if __name__ == "__main__":
    main()