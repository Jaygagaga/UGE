# import geopandas as gpd

import numpy as np
from shapely.geometry import Point, LineString, MultiLineString, MultiPoint, Polygon, MultiPolygon
from collections import defaultdict

# from networkx_graph import create_and_save_network_complete

import pandas as pd
import requests
import time
from shapely.geometry import Point, LineString
import geopandas as gpd

def reverse_geocode_nominatim(lat, lng, max_retries=3, delay=1):
    """
    Reverse geocode coordinates using Nominatim API with retry logic
    
    Parameters:
    -----------
    lat : float
        Latitude
    lng : float
        Longitude
    max_retries : int
        Maximum number of retry attempts
    delay : float
        Delay between retries in seconds
    
    Returns:
    --------
    dict : Dictionary containing city and street information
    """
    url = f"https://nominatim.openstreetmap.org/reverse?lat={lat}&lon={lng}&format=json&addressdetails=1"
    headers = {'User-Agent': 'UrbanKG-Geocoding/1.0'}  # Required by Nominatim
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if 'address' in data:
                address = data['address']
                return {
                    'city': address.get('city') or address.get('town') or address.get('village') or address.get('municipality'),
                    'street': address.get('road') or address.get('street') or address.get('residential'),
                    'postcode': address.get('postcode'),
                    'country': address.get('country'),
                    'state': address.get('state'),
                    'county': address.get('county')
                }
            return {'city': None, 'street': None, 'postcode': None, 'country': None, 'state': None, 'county': None}
            
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                print(f"Geocoding attempt {attempt + 1} failed: {e}. Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= 2  # Exponential backoff
            else:
                print(f"Geocoding failed after {max_retries} attempts: {e}")
                return {'city': None, 'street': None, 'postcode': None, 'country': None, 'state': None, 'county': None}
    
    return {'city': None, 'street': None, 'postcode': None, 'country': None, 'state': None, 'county': None}


def get_geometry_center(geometry):
    """
    Get the center point of a geometry (Point, LineString, Polygon, etc.)
    For MultiLineString and MultiPolygon, uses the first item
    
    Parameters:
    -----------
    geometry : shapely.geometry
        Geometry object
    
    Returns:
    --------
    tuple : (lat, lng) coordinates
    """
    if geometry is None or geometry.is_empty:
        return None, None
    
    try:
        if isinstance(geometry, Point):
            return geometry.y, geometry.x  # lat, lng
        elif isinstance(geometry, (LineString, Polygon)):
            # For single LineString or Polygon, get the centroid
            centroid = geometry.centroid
            return centroid.y, centroid.x  # lat, lng
        elif isinstance(geometry, MultiLineString):
            # For MultiLineString, use the first LineString
            if len(geometry.geoms) > 0:
                first_line = list(geometry.geoms)[0]
                centroid = first_line.centroid
                return centroid.y, centroid.x  # lat, lng
            else:
                return None, None
        elif isinstance(geometry, MultiPolygon):
            # For MultiPolygon, use the first Polygon
            if len(geometry.geoms) > 0:
                first_polygon = list(geometry.geoms)[0]
                centroid = first_polygon.centroid
                return centroid.y, centroid.x  # lat, lng
            else:
                return None, None
        elif hasattr(geometry, 'geoms'):
            # For other multi-part geometries, use the first item
            if len(geometry.geoms) > 0:
                first_geom = list(geometry.geoms)[0]
                if hasattr(first_geom, 'centroid'):
                    centroid = first_geom.centroid
                    return centroid.y, centroid.x  # lat, lng
                elif hasattr(first_geom, 'coords'):
                    # For geometries with coords (like Point)
                    coords = list(first_geom.coords)
                    if len(coords) > 0:
                        return coords[0][1], coords[0][0]  # lat, lng
            return None, None
        elif hasattr(geometry, 'centroid'):
            # For other geometry types with centroid
            centroid = geometry.centroid
            return centroid.y, centroid.x  # lat, lng
        elif hasattr(geometry, 'coords'):
            # For geometries with coords
            coords = list(geometry.coords)
            if len(coords) > 0:
                return coords[0][1], coords[0][0]  # lat, lng
        else:
            print(f"Warning: Unsupported geometry type: {type(geometry)}")
            return None, None
            
    except Exception as e:
        print(f"Error getting geometry center: {e}")
        return None, None

def add_geocoding_info(df, enable_geocoding=True, geocode_missing_only=True, max_geocoding_requests=None,
                       sleep_time=1.0, batch_size=50, max_time_hours=24, resume_file=None, 
                       prioritize_by_distance=True, max_distance_priority=5000):
    """
    Add city and street information using geocoding for missing data with enhanced features
    
    Parameters:
    -----------
    df : GeoDataFrame
        Input GeoDataFrame
    enable_geocoding : bool
        Whether to enable geocoding
    geocode_missing_only : bool
        Only geocode items missing city/street information
    max_geocoding_requests : int, optional
        Maximum number of geocoding requests to make (None for unlimited)
    sleep_time : float
        Sleep time between requests in seconds (default: 1.0 for Nominatim compliance)
    batch_size : int
        Number of requests to process before saving progress
    max_time_hours : float
        Maximum time to spend geocoding in hours
    resume_file : str
        File path to save/load progress for resuming interrupted geocoding
    prioritize_by_distance : bool
        Whether to prioritize items by distance from a reference point
    max_distance_priority : float
        Maximum distance for priority items (in meters)
    
    Returns:
    --------
    GeoDataFrame : Updated GeoDataFrame with geocoded information
    """
    import os
    
    if not enable_geocoding:
        return df
    
    df_updated = df.copy()
    
    # Ensure required columns exist
    if 'city' not in df_updated.columns:
        df_updated['city'] = None
    if 'street' not in df_updated.columns:
        df_updated['street'] = None
    
    # Count missing information
    missing_city = df_updated['city'].isna() | (df_updated['city'] == '')
    missing_street = df_updated['street'].isna() | (df_updated['street'] == '')
    
    if geocode_missing_only:
        # Only geocode items missing either city or street
        items_to_geocode = missing_city | missing_street
    else:
        # Geocode all items
        items_to_geocode = pd.Series([True] * len(df_updated), index=df_updated.index)
    
    # Get indices of items that need geocoding
    items_to_geocode_idx = items_to_geocode[items_to_geocode].index.tolist()
    print(f"Items to geocode: {len(items_to_geocode_idx)}")
    
    if len(items_to_geocode_idx) == 0:
        print("No items need geocoding")
        return df_updated
    
    # Load progress if resume file exists
    processed_indices = set()
    if resume_file and os.path.exists(resume_file):
        try:
            with open(resume_file, 'r') as f:
                processed_indices = set(int(line.strip()) for line in f)
            print(f"Loaded {len(processed_indices)} previously processed items from {resume_file}")
        except Exception as e:
            print(f"Warning: Could not load resume file {resume_file}: {e}")
    
    # Create priority queue for items to geocode
    items_queue = []
    for idx in items_to_geocode_idx:
        if idx in processed_indices:
            continue
            
        row = df_updated.loc[idx]
        needs_city = missing_city.loc[idx]
        needs_street = missing_street.loc[idx]
        
        # Get coordinates from geometry
        lat, lng = get_geometry_center(row.geometry)
        if lat is None or lng is None:
            continue
        
        # Calculate priority score
        priority_score = 0
        if prioritize_by_distance:
            # Prioritize items closer to center of dataset
            center_lat = df_updated.geometry.centroid.y.mean()
            center_lng = df_updated.geometry.centroid.x.mean()
            distance = ((lat - center_lat) ** 2 + (lng - center_lng) ** 2) ** 0.5
            if distance * 111320 <= max_distance_priority:  # Convert to meters
                priority_score = 1 / (1 + distance)
        
        # Add to queue with priority
        items_queue.append((priority_score, idx, lat, lng, needs_city, needs_street))
    
    # Sort by priority (highest first)
    items_queue.sort(key=lambda x: x[0], reverse=True)
    
    # Apply request limit if specified
    if max_geocoding_requests is not None and len(items_queue) > max_geocoding_requests:
        print(f"Limiting geocoding to {max_geocoding_requests} items (out of {len(items_queue)})")
        items_queue = items_queue[:max_geocoding_requests]
    elif max_geocoding_requests is None:
        print(f"Processing all {len(items_queue)} items (no request limit)")
    else:
        print(f"Processing {len(items_queue)} items (within limit of {max_geocoding_requests})")
    
    # Initialize counters and timers
    geocoded_count = 0
    success_count = 0
    start_time = time.time()
    max_time_seconds = max_time_hours * 3600
    
    print(f"Starting geocoding with {len(items_queue)} items")
    print(f"Batch size: {batch_size}, Sleep time: {sleep_time}s, Max time: {max_time_hours}h")
    
    # Process items in batches
    for batch_start in range(0, len(items_queue), batch_size):
        batch_end = min(batch_start + batch_size, len(items_queue))
        batch_items = items_queue[batch_start:batch_end]
        
        print(f"Processing batch {batch_start//batch_size + 1}/{(len(items_queue)-1)//batch_size + 1} "
              f"({len(batch_items)} items)")
        
        # Process each item in the batch
        for priority_score, idx, lat, lng, needs_city, needs_street in batch_items:
            # Check time limit
            if time.time() - start_time > max_time_seconds:
                print(f"Time limit reached ({max_time_hours}h). Stopping geocoding.")
                break
            
            # Perform geocoding
            geocoding_result = reverse_geocode_nominatim(lat, lng)
            geocoded_count += 1
            
            # Update missing information
            if needs_city and geocoding_result['city']:
                df_updated.at[idx, 'city'] = geocoding_result['city']
                success_count += 1
                
            if needs_street and geocoding_result['street']:
                df_updated.at[idx, 'street'] = geocoding_result['street']
                success_count += 1
            
            # Add additional geocoding information if not present
            for field in [ 'country', 'state', 'county']:
                if field not in df_updated.columns:
                    df_updated[field] = None
                if geocoding_result[field]:
                    df_updated.at[idx, field] = geocoding_result[field]
            
            # Mark as processed
            processed_indices.add(idx)
            
            # Progress reporting
            if geocoded_count % 10 == 0:
                elapsed_time = time.time() - start_time
                rate = geocoded_count / elapsed_time if elapsed_time > 0 else 0
                print(f"Geocoded {geocoded_count}/{len(items_queue)} items "
                      f"({success_count} successful, {rate:.2f} req/s)")
            
            # Rate limiting
            time.sleep(sleep_time)
        
        # Save progress after each batch
        if resume_file:
            try:
                with open(resume_file, 'w') as f:
                    for idx in processed_indices:
                        f.write(f"{idx}\n")
                print(f"Progress saved to {resume_file}")
            except Exception as e:
                print(f"Warning: Could not save progress to {resume_file}: {e}")
        
        # Check time limit after batch
        if time.time() - start_time > max_time_seconds:
            print(f"Time limit reached ({max_time_hours}h). Stopping geocoding.")
            break
    
    # Final summary
    elapsed_time = time.time() - start_time
    print(f"Geocoding completed:")
    print(f"  Total requests: {geocoded_count}")
    print(f"  Successful updates: {success_count}")
    print(f"  Elapsed time: {elapsed_time/3600:.2f} hours")
    print(f"  Average rate: {geocoded_count/elapsed_time:.2f} requests/second")
    
    return df_updated

def find_road_crossings_simplified(processed_streets, line_id_col='id', tolerance=1e-8):
    """
    简化版道路交叉点查找函数，基于您提供的代码修改
    只专注于找到交叉点坐标和相关的街道ID信息

    参数:
    processed_streets : GeoDataFrame
        处理过的街道GeoDataFrame，包含方向信息
    line_id_col : str
        街道ID列名 (default: 'id')
    tolerance : float
        坐标容差 (default: 1e-8)

    返回:
    GeoDataFrame: 包含交叉点信息的GeoDataFrame
    """

    print(f"开始查找道路交叉点...")
    print(f"总计 {len(processed_streets)} 条街道")

    # 验证ID列是否存在
    if line_id_col not in processed_streets.columns:
        raise ValueError(f"'{line_id_col}' 列在街道数据中不存在")

    # 创建副本并重置索引
    lines_gdf = processed_streets.copy().reset_index(drop=True)

    # 创建空间索引
    lines_sindex = lines_gdf.sindex

    # 结果列表
    results = []

    # 用于跟踪交叉点和连接道路的字典
    intersection_points = defaultdict(set)

    # 处理每条线
    processed_pairs = set()  # 避免重复处理

    for idx1, line1_row in lines_gdf.iterrows():
        line1_id = line1_row[line_id_col]
        line1_geom = line1_row.geometry
        line1_name = line1_row.get('name', f'Street_{line1_id}')

        # 跳过无效几何
        if not line1_geom.is_valid or line1_geom.is_empty:
            continue

        # 跳过非线几何
        if not isinstance(line1_geom, (LineString, MultiLineString)):
            continue

        # 获取潜在相交的线
        possible_matches_idx = list(lines_sindex.intersection(line1_geom.bounds))

        # 移除自身
        if idx1 in possible_matches_idx:
            possible_matches_idx.remove(idx1)

        for idx2 in possible_matches_idx:
            line2_row = lines_gdf.iloc[idx2]
            line2_id = line2_row[line_id_col]
            line2_geom = line2_row.geometry
            line2_name = line2_row.get('name', f'Street_{line2_id}')

            # 避免重复处理同一对线段
            pair_key = tuple(sorted([line1_id, line2_id]))
            if pair_key in processed_pairs:
                continue
            processed_pairs.add(pair_key)

            # 跳过无效几何
            if not line2_geom.is_valid or line2_geom.is_empty:
                continue

            # 跳过非线几何
            if not isinstance(line2_geom, (LineString, MultiLineString)):
                continue

            # 检查是否相交
            if line1_geom.intersects(line2_geom):
                # 计算交点
                intersection = line1_geom.intersection(line2_geom)

                # 处理不同类型的交点
                intersection_points_list = []

                if isinstance(intersection, Point):
                    intersection_points_list = [intersection]
                elif isinstance(intersection, MultiPoint):
                    intersection_points_list = list(intersection.geoms)
                elif hasattr(intersection, 'geoms'):
                    # 从复合几何中提取点
                    for geom in intersection.geoms:
                        if isinstance(geom, Point):
                            intersection_points_list.append(geom)

                # 为每个交点创建记录
                for point in intersection_points_list:
                    # 创建交叉点记录
                    result = {
                        'geometry': point,
                        'line1_id': line1_id,
                        'line2_id': line2_id,
                        'line1_name': line1_name,
                        'line2_name': line2_name,
                        'x': point.x,
                        'y': point.y,
                        'type': 'crossing'
                    }

                    results.append(result)

                    # 跟踪交叉点
                    point_key = f"{point.x:.10f},{point.y:.10f}"
                    intersection_points[point_key].add(line1_id)
                    intersection_points[point_key].add(line2_id)

        # 进度报告
        if (idx1 + 1) % 100 == 0:
            print(f"已处理 {idx1 + 1}/{len(lines_gdf)} 条街道，找到 {len(results)} 个交叉点")

    print(f"完成！总共找到 {len(results)} 个交叉点")

    if len(results) == 0:
        print("没有找到交叉点，返回空的GeoDataFrame")
        empty_df = pd.DataFrame(
            columns=['line1_id', 'line2_id', 'line1_name', 'line2_name', 'x', 'y', 'type', 'geometry'])
        return gpd.GeoDataFrame(empty_df, crs=processed_streets.crs)

    # 创建GeoDataFrame
    crossings_gdf = gpd.GeoDataFrame(results, crs=processed_streets.crs)

    return crossings_gdf


def remove_duplicate_crossings_enhanced(crossings_gdf, tolerance=1e-6):
    """
    增强版去重函数，处理在同一点的多个交叉

    参数:
    crossings_gdf : GeoDataFrame
        交叉点GeoDataFrame
    tolerance : float
        坐标容差

    返回:
    GeoDataFrame: 去重后的交叉点数据
    """
    if len(crossings_gdf) == 0:
        return crossings_gdf

    print(f"去重前: {len(crossings_gdf)} 个交叉点")

    # 按坐标分组，处理重复点
    crossings_with_key = crossings_gdf.copy()
    crossings_with_key['coord_key'] = crossings_with_key.apply(
        lambda row: f"{row.x:.8f},{row.y:.8f}", axis=1
    )

    unique_crossings = []

    # 按坐标键分组
    grouped = crossings_with_key.groupby('coord_key')

    for coord_key, group in grouped:
        if len(group) == 1:
            # 单个交叉点，直接添加
            crossing = group.iloc[0].to_dict()
            del crossing['coord_key']  # 移除临时键
            unique_crossings.append(crossing)
        else:
            # 多个交叉点在同一位置，需要合并
            # 取第一个作为基础
            base_crossing = group.iloc[0].to_dict()
            del base_crossing['coord_key']

            # 收集所有涉及的线段ID
            all_line_ids = set()
            all_line_names = set()

            for _, row in group.iterrows():
                all_line_ids.add(row.line1_id)
                all_line_ids.add(row.line2_id)
                all_line_names.add(row.line1_name)
                all_line_names.add(row.line2_name)

            # 如果超过2条线在此相交，标记为复杂交叉点
            if len(all_line_ids) > 2:
                base_crossing['type'] = 'complex_crossing'
                base_crossing['intersecting_lines_count'] = len(all_line_ids)
                base_crossing['intersecting_lines'] = list(all_line_ids)
                base_crossing['intersecting_names'] = list(all_line_names)

            unique_crossings.append(base_crossing)

    if len(unique_crossings) == 0:
        return crossings_gdf.iloc[0:0].copy()

    result_gdf = gpd.GeoDataFrame(unique_crossings, crs=crossings_gdf.crs)
    print(f"去重后: {len(result_gdf)} 个交叉点")

    return result_gdf


def add_crossings_to_nodes_enhanced(nodes_gdf, crossings_gdf):
    """
    增强版将交叉点添加到节点GeoDataFrame中
    同时为crossings_gdf添加crossing_id列

    参数:
    nodes_gdf : GeoDataFrame
        原始节点GeoDataFrame
    crossings_gdf : GeoDataFrame
        交叉点GeoDataFrame

    返回:
    tuple: (合并后的节点数据, 更新后的交叉点数据)
    """
    if len(crossings_gdf) == 0:
        print("没有交叉点需要添加")
        # 为空的crossings_gdf添加crossing_id列
        crossings_updated = crossings_gdf.copy()
        if 'crossing_id' not in crossings_updated.columns:
            crossings_updated['crossing_id'] = None
        return nodes_gdf.copy(), crossings_updated

    print(f"添加 {len(crossings_gdf)} 个交叉点到节点数据中")

    # 创建crossings_gdf的副本以添加crossing_id
    crossings_updated = crossings_gdf.copy()

    # 准备交叉点数据以匹配节点GDF的结构
    crossing_nodes = []
    crossing_ids = []  # 存储分配的crossing_id

    # 获取下一个可用的ID
    if 'id' in nodes_gdf.columns:
        if pd.api.types.is_numeric_dtype(nodes_gdf['id']):
            next_id = int(nodes_gdf['id'].max()) + 1
        else:
            next_id = len(nodes_gdf) + 1
    else:
        next_id = len(nodes_gdf) + 1

    for idx, crossing in crossings_gdf.iterrows():
        # 创建新的节点记录
        node_data = {
            'geometry': crossing.geometry,
            'type': crossing.get('type', 'crossing'),
            'line1_id': crossing.line1_id,
            'line2_id': crossing.line2_id,
            'x': crossing.x,
            'y': crossing.y
        }

        # 分配ID并记录crossing_id
        assigned_id = next_id
        crossing_ids.append(assigned_id)

        if 'id' in nodes_gdf.columns:
            node_data['id'] = assigned_id

        # 添加名称
        if 'name' in nodes_gdf.columns:
            if crossing.get('type') == 'complex_crossing':
                # 复杂交叉点使用所有街道名称
                all_names = crossing.get('intersecting_names', [crossing.line1_name, crossing.line2_name])
                # 过滤掉None值并限制名称长度
                valid_names = [name for name in all_names if name is not None and str(name).strip() != '']
                if valid_names:
                    node_data['name'] = f"Complex_Crossing_{'_'.join(valid_names[:3])}"  # 限制名称长度
                else:
                    node_data['name'] = f"Complex_Crossing_{crossing.line1_id}_{crossing.line2_id}"
            else:
                # 简单交叉点
                name1 = crossing.line1_name if crossing.line1_name is not None else f"Street_{crossing.line1_id}"
                name2 = crossing.line2_name if crossing.line2_name is not None else f"Street_{crossing.line2_id}"
                print(f"Crossing_{name1}_{name2}")
                node_data['name'] = f"Crossing_{name1}_{name2}"


        # 添加地址（如果原节点有这个字段）
        if 'address' in nodes_gdf.columns:
            if crossing.get('type') == 'complex_crossing':
                node_data['address'] = f"Complex intersection"
            else:
                name1 = crossing.line1_name if crossing.line1_name is not None else f"Street_{crossing.line1_id}"
                name2 = crossing.line2_name if crossing.line2_name is not None else f"Street_{crossing.line2_id}"
                node_data['address'] = f"Intersection of {name1} and {name2}"

        # 添加复杂交叉点的额外信息
        if crossing.get('type') == 'complex_crossing':
            node_data['intersecting_lines_count'] = crossing.get('intersecting_lines_count', 2)
            if 'intersecting_lines' in crossing:
                node_data['intersecting_lines'] = str(crossing['intersecting_lines'])  # 转为字符串存储

        # 填充其他可能存在的列
        for col in nodes_gdf.columns:
            if col not in node_data and col != 'geometry':
                node_data[col] = None

        crossing_nodes.append(node_data)
        next_id += 1

    # 为crossings_gdf添加crossing_id列
    crossings_updated['crossing_id'] = crossing_ids

    # 创建交叉点节点的GeoDataFrame
    if len(crossing_nodes) > 0:
        crossing_nodes_gdf = gpd.GeoDataFrame(crossing_nodes, crs=nodes_gdf.crs)

        # 确保列顺序一致
        for col in nodes_gdf.columns:
            if col not in crossing_nodes_gdf.columns:
                crossing_nodes_gdf[col] = None

        # 重新排列列顺序
        crossing_nodes_gdf = crossing_nodes_gdf[nodes_gdf.columns.tolist()]

        # 合并数据
        combined_nodes_gdf = pd.concat([nodes_gdf, crossing_nodes_gdf], ignore_index=True)
        combined_nodes_gdf = gpd.GeoDataFrame(combined_nodes_gdf, crs=nodes_gdf.crs)
    else:
        combined_nodes_gdf = nodes_gdf.copy()

    print(f"节点总数: {len(nodes_gdf)} -> {len(combined_nodes_gdf)}")
    print(f"新增交叉点节点: {len(combined_nodes_gdf) - len(nodes_gdf)}")
    print(
        f"crossings_gdf已添加crossing_id列，ID范围: {min(crossing_ids) if crossing_ids else 'N/A'} - {max(crossing_ids) if crossing_ids else 'N/A'}")

    return combined_nodes_gdf, crossings_updated


def process_road_crossings_complete(processed_streets, nodes_gdf, line_id_col='id', tolerance=1e-6):
    """
    完整的道路交叉点处理流程

    参数:
    processed_streets : GeoDataFrame
        处理过的街道GeoDataFrame（包含方向信息）
    nodes_gdf : GeoDataFrame
        原始节点GeoDataFrame
    line_id_col : str
        街道ID列名
    tolerance : float
        去重容差

    返回:
    tuple: (更新后的节点GDF, 更新后的交叉点GDF)
    """
    print("=== 开始完整的道路交叉点处理 ===")

    # 1. 找到所有交叉点
    print("\n步骤 1: 查找道路交叉点")
    crossings_gdf = find_road_crossings_simplified(processed_streets, line_id_col)

    # 2. 去除重复的交叉点
    print("\n步骤 2: 去除重复交叉点")
    if len(crossings_gdf) > 0:
        unique_crossings_gdf = remove_duplicate_crossings_enhanced(crossings_gdf, tolerance)
    else:
        unique_crossings_gdf = crossings_gdf

    # 3. 将交叉点添加到节点数据中，并为crossings_gdf添加crossing_id
    print("\n步骤 3: 添加交叉点到节点数据并分配crossing_id")
    updated_nodes_gdf, updated_crossings_gdf = add_crossings_to_nodes_enhanced(nodes_gdf, unique_crossings_gdf)

    print("\n=== 道路交叉点处理完成 ===")
    print(f"crossings_gdf现在包含crossing_id列，与nodes_gdf中的交叉点节点ID对应")

    return updated_nodes_gdf, updated_crossings_gdf


def verify_crossing_id_mapping(updated_nodes_gdf, updated_crossings_gdf):
    """
    验证crossing_id与nodes_gdf中交叉点的ID映射关系

    参数:
    updated_nodes_gdf : GeoDataFrame
        更新后的节点GDF
    updated_crossings_gdf : GeoDataFrame
        更新后的交叉点GDF

    返回:
    bool: 映射是否正确
    """
    print("=== 验证crossing_id映射 ===")

    if 'crossing_id' not in updated_crossings_gdf.columns:
        print("错误: crossings_gdf中没有crossing_id列")
        return False

    # 获取crossings中的所有crossing_id
    crossing_ids = updated_crossings_gdf['crossing_id'].dropna().tolist()

    if not crossing_ids:
        print("警告: 没有有效的crossing_id")
        return True  # 空的情况算作正确

    # 检查这些ID是否在nodes_gdf中存在
    if 'id' not in updated_nodes_gdf.columns:
        print("错误: nodes_gdf中没有id列")
        return False

    nodes_ids = set(updated_nodes_gdf['id'].tolist())
    missing_ids = [cid for cid in crossing_ids if cid not in nodes_ids]

    if missing_ids:
        print(f"错误: 以下crossing_id在nodes_gdf中不存在: {missing_ids}")
        return False

    # 检查这些节点是否确实是交叉点类型
    crossing_nodes = updated_nodes_gdf[updated_nodes_gdf['id'].isin(crossing_ids)]

    if 'type' in crossing_nodes.columns:
        non_crossing_types = crossing_nodes[~crossing_nodes['type'].str.contains('crossing', na=False)]
        if len(non_crossing_types) > 0:
            print(f"警告: {len(non_crossing_types)} 个节点的type不包含'crossing'")

    print(f"验证成功:")
    print(f"  crossings_gdf中的crossing_id数量: {len(crossing_ids)}")
    print(f"  对应的nodes中的交叉点数量: {len(crossing_nodes)}")
    print(f"  ID范围: {min(crossing_ids)} - {max(crossing_ids)}")

    return True


def analyze_crossings_detailed(crossings_gdf, processed_streets):
    """
    详细分析交叉点数据
    """
    if len(crossings_gdf) == 0:
        print("没有交叉点数据可分析")
        return

    print("=== 详细交叉点分析 ===")
    print(f"总交叉点数: {len(crossings_gdf)}")

    # 统计交叉点类型
    if 'type' in crossings_gdf.columns:
        type_counts = crossings_gdf['type'].value_counts()
        print(f"\n交叉点类型分布:")
        for crossing_type, count in type_counts.items():
            print(f"  {crossing_type}: {count}")

    # 统计复杂交叉点
    complex_crossings = crossings_gdf[crossings_gdf.get('type') == 'complex_crossing']
    if len(complex_crossings) > 0:
        print(f"\n复杂交叉点详情:")
        print(f"  复杂交叉点数量: {len(complex_crossings)}")
        if 'intersecting_lines_count' in complex_crossings.columns:
            max_lines = complex_crossings['intersecting_lines_count'].max()
            print(f"  最多线段相交数: {max_lines}")

    # 统计涉及的街道
    all_line_ids = set()
    for _, crossing in crossings_gdf.iterrows():
        all_line_ids.add(crossing.line1_id)
        all_line_ids.add(crossing.line2_id)

    print(f"\n涉及的街道数: {len(all_line_ids)}")
    print(f"街道总数: {len(processed_streets)}")
    print(f"有交叉点的街道比例: {len(all_line_ids) / len(processed_streets) * 100:.1f}%")

    # 统计每条街道的交叉点数量
    line_crossing_counts = {}
    for _, crossing in crossings_gdf.iterrows():
        line1_id = crossing.line1_id
        line2_id = crossing.line2_id

        line_crossing_counts[line1_id] = line_crossing_counts.get(line1_id, 0) + 1
        line_crossing_counts[line2_id] = line_crossing_counts.get(line2_id, 0) + 1

    # 找到交叉点最多的街道
    if line_crossing_counts:
        max_crossings = max(line_crossing_counts.values())
        busiest_lines = [line_id for line_id, count in line_crossing_counts.items() if count == max_crossings]
        print(f"\n交叉点最多的街道ID: {busiest_lines} (各有 {max_crossings} 个交叉点)")

# #
# # # 使用示例
# # if __name__ == "__main__":
# #     print("=== 道路交叉点处理系统（带crossing_id映射）===")
# #     print("""
# # 使用方法:
# #
# # # 处理道路交叉点（完整流程）
# # updated_nodes_gdf, updated_crossings_gdf = process_road_crossings_complete(
# #     processed_streets,  # 您处理过的街道数据（包含方向信息）
# #     nodes_gdf,         # 原始节点数据
# #     line_id_col='id',  # 街道ID列名
# #     tolerance=1e-6     # 去重容差
# # )
# #
# # # 验证crossing_id映射关系
# # mapping_correct = verify_crossing_id_mapping(updated_nodes_gdf, updated_crossings_gdf)
# #
# # # 分析结果
# # analyze_crossings_detailed(updated_crossings_gdf, processed_streets)
# #
# # # 查看crossings_gdf中的crossing_id列
# # print("crossings_gdf示例:")
# # print(updated_crossings_gdf[['line1_id', 'line2_id', 'line1_name', 'line2_name', 'crossing_id', 'x', 'y']].head())
# #
# # # 查看对应的nodes_gdf中的交叉点节点
# # crossing_nodes = updated_nodes_gdf[updated_nodes_gdf['type'].str.contains('crossing', na=False)]
# # print("\\n对应的交叉点节点:")
# # print(crossing_nodes[['id', 'name', 'type', 'line1_id', 'line2_id', 'x', 'y']].head())
# #
# # # 验证ID对应关系
# # crossing_ids_from_crossings = set(updated_crossings_gdf['crossing_id'].dropna())
# # crossing_ids_from_nodes = set(crossing_nodes['id'])
# # print(f"\\nID映射验证:")
# # print(f"crossings_gdf中的crossing_id: {len(crossing_ids_from_crossings)} 个")
# # print(f"nodes_gdf中的交叉点ID: {len(crossing_ids_from_nodes)} 个")
# # print(f"ID完全匹配: {crossing_ids_from_crossings == crossing_ids_from_nodes}")
# #
# # # 保存结果
# # updated_nodes_gdf.to_file('updated_nodes_with_crossings.gpkg', driver='GPKG')
# # updated_crossings_gdf.to_file('road_crossings_with_ids.gpkg', driver='GPKG')
# #     """)
#
# import geopandas as gpd
# import numpy as np
# from shapely.geometry import LineString
from geopy.distance import geodesic
# import pandas as pd
#
#
def analyze_and_update_streets(geojson_file =None,
                               straightness_threshold=0.8,
                               gdf = None,
                               max_turns=2,
                               max_vertices=10,
                               min_length_meters=50):
    """
    Load streets GeoJSON and add analysis results to the GeoDataFrame
    """

    # Load the streets data
    # if not gdf:
    #     print("Loading streets.geojson...")
    #     gdf = gpd.read_file(geojson_file)
    gdf['id'] = range(0, len(gdf))

    # Ensure CRS is EPSG:4326
    if gdf.crs is None:
        gdf = gdf.set_crs('EPSG:4326')
    elif gdf.crs != 'EPSG:4326':
        gdf = gdf.to_crs('EPSG:4326')

    print(f"Loaded {len(gdf)} streets")

    # Initialize new columns with default values
    gdf['bearing'] = np.nan
    gdf['direction'] = None
    gdf['straightness'] = np.nan
    gdf['turn_count'] = np.nan
    gdf['length_meters'] = np.nan
    gdf['is_simple'] = False

    simple_count = 0
    complex_count = 0

    # Analyze each street
    for idx, row in gdf.iterrows():
        geometry = row.geometry

        if isinstance(geometry, LineString):
            # Calculate metrics using geodesic methods
            straightness = calculate_straightness_geodesic(geometry)
            turn_count = count_significant_turns_geodesic(geometry)
            vertex_count = len(list(geometry.coords))
            street_length = calculate_length_geodesic(geometry)
            bearing = calculate_bearing_4326(geometry)
            direction = bearing_to_direction_english(bearing)

            # Update the GeoDataFrame with calculated values
            gdf.at[idx, 'bearing'] = bearing
            gdf.at[idx, 'direction'] = direction
            gdf.at[idx, 'straightness'] = straightness
            gdf.at[idx, 'turn_count'] = turn_count
            gdf.at[idx, 'length_meters'] = street_length

            # Check if street is simple
            is_simple = (straightness > straightness_threshold and
                         turn_count <= max_turns and
                         vertex_count <= max_vertices and
                         street_length > min_length_meters)

            gdf.at[idx, 'is_simple'] = is_simple

            if is_simple:
                simple_count += 1
            else:
                complex_count += 1
        else:
            # For non-LineString geometries, set appropriate values
            gdf.at[idx, 'turn_count'] = 0
            gdf.at[idx, 'length_meters'] = 0
            gdf.at[idx, 'straightness'] = 0

    print(f"Analysis completed:")
    print(f"  Simple streets: {simple_count}")
    print(f"  Complex streets: {complex_count}")

    return gdf


def calculate_straightness_geodesic(linestring_4326):
    """Calculate straightness using geodesic distance"""
    coords = list(linestring_4326.coords)
    if len(coords) < 2:
        return 0

    start_point = coords[0]  # [lon, lat]
    end_point = coords[-1]  # [lon, lat]

    # Geodesic distance (meters) - note geopy uses (lat, lon) format
    straight_distance = geodesic((start_point[1], start_point[0]),
                                 (end_point[1], end_point[0])).meters

    # Calculate actual path length
    actual_length = calculate_length_geodesic(linestring_4326)

    if actual_length == 0:
        return 0

    return straight_distance / actual_length


def calculate_length_geodesic(linestring_4326):
    """Calculate total length using geodesic distance"""
    coords = list(linestring_4326.coords)
    if len(coords) < 2:
        return 0

    total_length = 0
    for i in range(len(coords) - 1):
        p1 = coords[i]  # [lon, lat]
        p2 = coords[i + 1]  # [lon, lat]

        # Note: geodesic uses (lat, lon) format
        segment_distance = geodesic((p1[1], p1[0]), (p2[1], p2[0])).meters
        total_length += segment_distance

    return total_length


def count_significant_turns_geodesic(linestring_4326, angle_threshold=30):
    """Count significant turns using geodesic calculations"""
    coords = list(linestring_4326.coords)
    if len(coords) < 3:
        return 0

    turn_count = 0

    for i in range(1, len(coords) - 1):
        p1, p2, p3 = coords[i - 1], coords[i], coords[i + 1]

        # Calculate bearings between three points
        bearing1 = calculate_bearing_between_points(p1, p2)
        bearing2 = calculate_bearing_between_points(p2, p3)

        # Calculate angle difference
        angle_diff = abs(bearing2 - bearing1)
        if angle_diff > 180:
            angle_diff = 360 - angle_diff

        if angle_diff > angle_threshold:
            turn_count += 1

    return turn_count


def calculate_bearing_between_points(point1, point2):
    """Calculate bearing between two points"""
    lon1, lat1 = np.radians(point1[0]), np.radians(point1[1])
    lon2, lat2 = np.radians(point2[0]), np.radians(point2[1])

    dlon = lon2 - lon1

    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)

    bearing = np.degrees(np.arctan2(y, x))
    bearing = (bearing + 360) % 360

    return bearing


def calculate_bearing_4326(linestring):
    """Calculate overall bearing of linestring"""
    coords = list(linestring.coords)
    start_point = coords[0]
    end_point = coords[-1]

    return calculate_bearing_between_points(start_point, end_point)


def bearing_to_direction_english(bearing):
    """Convert bearing to English direction description"""
    if bearing is None or np.isnan(bearing):
        return None

    directions = [
        "N", "NNE", "NE", "ENE",
        "E", "ESE", "SE", "SSE",
        "S", "SSW", "SW", "WSW",
        "W", "WNW", "NW", "NNW"
    ]

    index = round(bearing / 22.5) % 16
    return directions[index]


def display_results(gdf):
    """Display analysis results"""
    simple_streets = gdf[gdf['is_simple'] == True]
    complex_streets = gdf[gdf['is_simple'] == False]

    print(f"\n=== SIMPLE STREETS ANALYSIS ===")
    print(f"Found {len(simple_streets)} simple streets:\n")

    # Display simple streets with their properties
    for idx, street in simple_streets.iterrows():
        name = street.get('name', f'Street_{idx}')
        direction = street['direction']
        bearing = street['bearing']
        straightness = street['straightness']
        length = street['length_meters']
        turns = street['turn_count']

        print(f"{name}: {direction} ({bearing:.1f}°)")
        print(f"  - Straightness: {straightness:.2f}")
        print(f"  - Length: {length:.0f}m")
        print(f"  - Turns: {int(turns)}")
        print()

    print(f"=== SUMMARY ===")
    print(f"Total streets: {len(gdf)}")
    print(f"Simple streets: {len(simple_streets)}")
    print(f"Complex streets: {len(complex_streets)}")

    # Show some statistics
    if len(simple_streets) > 0:
        print(f"\nSimple streets statistics:")
        print(f"  Average length: {simple_streets['length_meters'].mean():.1f}m")
        print(f"  Average straightness: {simple_streets['straightness'].mean():.2f}")
        print(f"  Average turns: {simple_streets['turn_count'].mean():.1f}")

        # Direction distribution
        direction_counts = simple_streets['direction'].value_counts()
        print(f"\nDirection distribution:")
        for direction, count in direction_counts.items():
            print(f"  {direction}: {count} streets")


def save_results(gdf, output_file='streets_analyzed.geojson'):
    """Save the analyzed streets to a new file"""
    # Save the complete analyzed dataset
    gdf.to_file(output_file, driver='GeoJSON')
    print(f"\nResults saved to {output_file}")

    # Also save just the simple streets
    simple_streets = gdf[gdf['is_simple'] == True]
    if len(simple_streets) > 0:
        simple_output = output_file.replace('.geojson', '_simple_only.geojson')
        simple_streets.to_file(simple_output, driver='GeoJSON')
        print(f"Simple streets saved to {simple_output}")


def find_crossing_pairs_on_same_street(updated_crossings_gdf, processed_streets=None):
    """
    找到在同一条街道上的交叉点对

    参数:
    updated_crossings_gdf : GeoDataFrame
        包含crossing_id的交叉点GeoDataFrame
    processed_streets : GeoDataFrame, optional
        处理过的街道数据，用于获取街道名称

    返回:
    GeoDataFrame: 包含交叉点对信息的GeoDataFrame
    """

    print("=== 查找同一街道上的交叉点对 ===")

    if len(updated_crossings_gdf) < 2:
        print("交叉点数量少于2个，无法形成对")
        return gpd.GeoDataFrame()

    # 创建结果列表
    crossing_pairs = []

    # 获取所有唯一的line_id
    all_line_ids = set()
    for _, row in updated_crossings_gdf.iterrows():
        all_line_ids.add(row['line1_id'])
        all_line_ids.add(row['line2_id'])

    print(f"总共涉及 {len(all_line_ids)} 条不同的街道")

    # 为每个line_id找到包含它的所有交叉点
    for line_id in all_line_ids:
        # 找到所有包含此line_id的交叉点
        crossings_with_line = updated_crossings_gdf[
            (updated_crossings_gdf['line1_id'] == line_id) |
            (updated_crossings_gdf['line2_id'] == line_id)
            ].copy()

        if len(crossings_with_line) < 2:
            continue  # 这条街道上少于2个交叉点，跳过

        # 获取街道名称
        street_name = None
        if processed_streets is not None:
            street_info = processed_streets[processed_streets['id'] == line_id]
            if not street_info.empty:
                street_name = street_info.iloc[0].get('name', f'Street_{line_id}')

        if street_name is None:
            street_name = f'Street_{line_id}'

        print(f"街道 '{street_name}' (ID: {line_id}) 上有 {len(crossings_with_line)} 个交叉点")

        # 创建所有可能的交叉点对
        crossings_list = crossings_with_line.to_dict('records')

        for i in range(len(crossings_list)):
            for j in range(i + 1, len(crossings_list)):
                crossing1 = crossings_list[i]
                crossing2 = crossings_list[j]

                # 计算两个交叉点之间的距离
                point1 = crossing1['geometry']
                point2 = crossing2['geometry']
                distance = point1.distance(point2)  # 度为单位

                # 转换为米（近似）
                distance_meters = distance * 111320  # 1度约等于111320米

                # 确定每个交叉点在这条街道上与哪条街道相交
                # crossing1与街道的关系
                if crossing1['line1_id'] == line_id:
                    crossing1_other_line = crossing1['line2_id']
                    crossing1_other_name = crossing1['line2_name']
                else:
                    crossing1_other_line = crossing1['line1_id']
                    crossing1_other_name = crossing1['line1_name']

                # crossing2与街道的关系
                if crossing2['line1_id'] == line_id:
                    crossing2_other_line = crossing2['line2_id']
                    crossing2_other_name = crossing2['line2_name']
                else:
                    crossing2_other_line = crossing2['line1_id']
                    crossing2_other_name = crossing2['line1_name']

                # 创建交叉点对记录
                pair_data = {
                    'common_line_id': line_id,
                    'common_street_name': street_name,

                    'crossing1_id': crossing1['crossing_id'],
                    'crossing1_x': crossing1['x'],
                    'crossing1_y': crossing1['y'],
                    'crossing1_other_line_id': crossing1_other_line,
                    'crossing1_other_street_name': crossing1_other_name,

                    'crossing2_id': crossing2['crossing_id'],
                    'crossing2_x': crossing2['x'],
                    'crossing2_y': crossing2['y'],
                    'crossing2_other_line_id': crossing2_other_line,
                    'crossing2_other_street_name': crossing2_other_name,

                    'distance_degrees': distance,
                    'distance_meters': distance_meters,

                    # 创建连接两个交叉点的线段几何
                    'geometry': LineString([point1, point2])
                }

                crossing_pairs.append(pair_data)

    # 创建结果GeoDataFrame
    if crossing_pairs:
        crossing_pairs_gdf = gpd.GeoDataFrame(
            crossing_pairs,
            crs=updated_crossings_gdf.crs
        )

        print(f"\n找到 {len(crossing_pairs_gdf)} 个交叉点对")

        # 统计信息
        print(f"涉及的街道数: {crossing_pairs_gdf['common_line_id'].nunique()}")
        print(f"平均距离: {crossing_pairs_gdf['distance_meters'].mean():.1f} 米")
        print(f"最短距离: {crossing_pairs_gdf['distance_meters'].min():.1f} 米")
        print(f"最长距离: {crossing_pairs_gdf['distance_meters'].max():.1f} 米")

        return crossing_pairs_gdf
    else:
        print("未找到任何交叉点对")
        return gpd.GeoDataFrame()

import math
def meters_to_degrees(meters, latitude):
    """
    将米转换为度（在给定纬度处的近似值）

    参数:
    meters: 距离（米）
    latitude: 纬度（度）

    返回:
    degrees: 对应的度数
    """
    # 地球半径（米）
    earth_radius = 6378137.0

    # 纬度方向：1度 ≈ 111320米
    lat_degrees = meters / 111320.0

    # 经度方向：在给定纬度处，1度的距离会变化
    lon_degrees = meters / (111320.0 * math.cos(math.radians(latitude)))

    # 返回较大的值作为缓冲区半径（保守估计）
    return max(lat_degrees, lon_degrees)
def calculate_geodesic_distance(point1, point2):
    """
    计算两点间的大圆距离（米）

    参数:
    point1, point2: shapely Point对象 (经度, 纬度)

    返回:
    distance: 距离（米）
    """
    if not isinstance(point1, Point) or not isinstance(point2, Point):
        return float('inf')

    try:
        coord1 = (point1.y, point1.x)  # geopy使用 (lat, lon)
        coord2 = (point2.y, point2.x)
        return geodesic(coord1, coord2).meters
    except:
        return float('inf')
def find_nearest_point_on_line_4326(point, line):
    """
    在线段上找到最接近给定点的点（适用于EPSG:4326）

    参数:
    point: shapely Point对象
    line: shapely LineString对象

    返回:
    nearest_point: 线段上最近的点
    distance: 距离（米）
    """
    if not isinstance(line, LineString):
        return None, float('inf')

    try:
        # 使用shapely的project方法找到最近点
        nearest_point = line.interpolate(line.project(point))
        distance = calculate_geodesic_distance(point, nearest_point)
        return nearest_point, distance
    except:
        return None, float('inf')

"""Nearest line to each point"""
def find_nearest_lines_4326(points_gdf, lines_gdf, max_distance=100.0,
                            point_id_col='id', line_id_col='id'):
    """
    为points_gdf中的每个点找到lines_gdf中最近的线段
    专门适配EPSG:4326坐标系

    参数:
    points_gdf : GeoDataFrame
        包含点几何的GeoDataFrame
    lines_gdf : GeoDataFrame
        包含线几何的GeoDataFrame
    max_distance : float, optional
        最大搜索距离（米） (default: 100.0)
    point_id_col : str, optional
        点标识符的列名 (default: index)
    line_id_col : str, optional
        线标识符的列名 (default: index)

    返回:
    GeoDataFrame
        包含point_id, line_id, distance, bearing和最近点的GeoDataFrame
    """

    # 确保相同的CRS
    if points_gdf.crs != lines_gdf.crs:
        lines_gdf = lines_gdf.to_crs(points_gdf.crs)

    # 验证是否为EPSG:4326
    if points_gdf.crs.to_string() != 'EPSG:4326':
        print(f"警告: 当前CRS为{points_gdf.crs}，建议使用EPSG:4326")

    print(f"开始处理 {len(points_gdf)} 个点和 {len(lines_gdf)} 条线段")
    print(f"最大搜索距离: {max_distance} 米")

    # 为线段创建空间索引
    lines_sindex = lines_gdf.sindex

    # 结果列表
    results = []
    processed_count = 0

    def calculate_bearing_4326(point1, point2):
        """
        计算两点间的方位角 (适用于EPSG:4326)

        参数:
        point1, point2: shapely Point对象 (经度, 纬度)

        返回:
        bearing: 方位角（度），北为0度，顺时针增加
        """
        if not isinstance(point1, Point) or not isinstance(point2, Point):
            return None

        lon1, lat1 = point1.x, point1.y
        lon2, lat2 = point2.x, point2.y

        # 转换为弧度
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        dlon_rad = math.radians(lon2 - lon1)

        # 计算方位角
        y = math.sin(dlon_rad) * math.cos(lat2_rad)
        x = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon_rad)

        bearing_rad = math.atan2(y, x)
        bearing_deg = math.degrees(bearing_rad)
        bearing_deg = (bearing_deg + 360) % 360

        return bearing_deg

    # 处理每个点
    for idx, point_row in points_gdf.iterrows():
        # 获取点ID
        if point_id_col and point_id_col in point_row:
            point_id = point_row[point_id_col]
        else:
            point_id = idx

        point_geom = point_row.geometry

        # 跳过无效几何
        if point_geom is None or not isinstance(point_geom, Point) or point_geom.is_empty:
            continue

        # 将米转换为度来创建缓冲区
        latitude = point_geom.y
        buffer_degrees = meters_to_degrees(max_distance, latitude)

        # 创建缓冲区查找附近线段
        buffer = point_geom.buffer(buffer_degrees)
        buffer_bounds = buffer.bounds

        # 使用空间索引查找缓冲区内的线段
        possible_matches_idx = list(lines_sindex.intersection(buffer_bounds))

        if not possible_matches_idx:
            continue

        # 获取候选线段
        candidates = lines_gdf.iloc[possible_matches_idx]

        # 找到最近的线段
        min_distance = float('inf')
        nearest_line_idx = None
        nearest_point_on_line = None

        for cand_idx in possible_matches_idx:
            line_row = lines_gdf.iloc[cand_idx]
            line_geom = line_row.geometry

            if isinstance(line_geom, LineString):
                nearest_point, distance = find_nearest_point_on_line_4326(point_geom, line_geom)
            elif isinstance(line_geom, MultiLineString):
                # 对于MultiLineString，检查所有组成部分
                min_dist_multi = float('inf')
                nearest_point_multi = None

                for line_part in line_geom.geoms:
                    if isinstance(line_part, LineString):
                        point_on_line, dist = find_nearest_point_on_line_4326(point_geom, line_part)
                        if dist < min_dist_multi:
                            min_dist_multi = dist
                            nearest_point_multi = point_on_line

                nearest_point = nearest_point_multi
                distance = min_dist_multi
            else:
                continue

            # 检查距离是否在最大距离内
            if distance <= max_distance and distance < min_distance:
                min_distance = distance
                nearest_line_idx = cand_idx
                nearest_point_on_line = nearest_point

        # 如果找到了符合条件的线段
        if nearest_line_idx is not None:
            nearest_line = lines_gdf.iloc[nearest_line_idx]

            # 获取线段ID
            if line_id_col and line_id_col in nearest_line:
                line_id = nearest_line[line_id_col]
            else:
                line_id = nearest_line_idx

            # 计算从点到线段的方位角
            bearing = None
            if nearest_point_on_line:
                bearing = calculate_bearing_4326(point_geom, nearest_point_on_line)

            # 添加到结果中
            results.append({
                'point_id': point_id,
                'line_id': line_id,
                'distance': min_distance,
                'bearing': bearing,
                'point_geometry': point_geom,
                'nearest_line_point': nearest_point_on_line,
                'point_x': point_geom.x,
                'point_y': point_geom.y,
                'nearest_x': nearest_point_on_line.x if nearest_point_on_line else None,
                'nearest_y': nearest_point_on_line.y if nearest_point_on_line else None
            })

        processed_count += 1
        if processed_count % 1000 == 0:
            print(f"已处理 {processed_count}/{len(points_gdf)} 个点")

    print(f"处理完成，找到 {len(results)} 个点-线段匹配")

    # 创建结果GeoDataFrame
    if results:
        result_df = pd.DataFrame(results)
        result_gdf = gpd.GeoDataFrame(
            result_df,
            geometry='point_geometry',
            crs=points_gdf.crs
        )
        return result_gdf
    else:
        return gpd.GeoDataFrame(
            columns=['point_id', 'line_id', 'distance', 'bearing',
                     'point_geometry', 'nearest_line_point',
                     'point_x', 'point_y', 'nearest_x', 'nearest_y'],
            geometry='point_geometry',
            crs=points_gdf.crs
        )


from difflib import SequenceMatcher
import re


def enhance_nearest_streets_with_names(nearest_streets_gdf, points_gdf, lines_gdf,
                                       point_street_col='street',
                                       point_id_col='point_id',
                                       line_id_col='id',
                                       line_name_col='name'):
    """
    为最近街道结果添加街道名称，并与POI原始街道名称进行比较

    参数:
    nearest_streets_gdf : GeoDataFrame
        find_nearest_lines_4326函数的输出结果
    points_gdf : GeoDataFrame
        包含POI和原始街道名称的GeoDataFrame（街道列可选）
    lines_gdf : GeoDataFrame
        包含街道ID和名称的线段GeoDataFrame
    point_street_col : str
        POI中街道名称的列名（如果不存在会跳过相关处理）
    point_id_col : str
        最近街道结果中point_id列名
    line_id_col : str
        线段GDF中ID列名
    line_name_col : str
        线段GDF中名称列名

    返回:
    GeoDataFrame: 增强后的最近街道结果，包含名称比较（如果可能）
    """

    print("=== 增强最近街道结果并进行名称比较 ===")
    print(f"最近街道记录数: {len(nearest_streets_gdf)}")

    # 检查输入数据
    print(f"Points GDF 列名: {list(points_gdf.columns)}")
    print(f"Lines GDF 列名: {list(lines_gdf.columns)}")
    print(f"Nearest streets GDF 列名: {list(nearest_streets_gdf.columns)}")

    # 创建增强版的结果GDF
    enhanced_gdf = nearest_streets_gdf.copy()

    # 1. 从lines_gdf获取找到的最近街道名称
    if line_name_col in lines_gdf.columns:
        line_name_dict = dict(zip(lines_gdf[line_id_col], lines_gdf[line_name_col]))
        enhanced_gdf['nearest_street_name'] = enhanced_gdf['line_id'].map(line_name_dict)
        print(f"添加了最近街道名称: {enhanced_gdf['nearest_street_name'].notna().sum()} 条记录")
    else:
        print(f"警告: lines_gdf中没有找到列 '{line_name_col}'")
        enhanced_gdf['nearest_street_name'] = None

    # 2. 检查POI是否有街道名称列
    has_street_col = point_street_col in points_gdf.columns

    if has_street_col:
        print(f"找到POI街道名称列: '{point_street_col}'")
        # 从points_gdf获取POI的原始街道名称
        if 'id' in points_gdf.columns:
            point_street_dict = dict(zip(points_gdf['id'], points_gdf[point_street_col]))
        else:
            point_street_dict = dict(zip(points_gdf.index, points_gdf[point_street_col]))

        enhanced_gdf['original_street_name'] = enhanced_gdf[point_id_col].map(point_street_dict)
        print(f"添加了原始街道名称: {enhanced_gdf['original_street_name'].notna().sum()} 条记录")
    else:
        print(f"注意: points_gdf中没有找到街道名称列 '{point_street_col}'")
        enhanced_gdf['original_street_name'] = None
        print("将跳过原始街道名称的映射")

    # 3. 添加POI名称（如果存在）
    if 'name' in points_gdf.columns:
        if 'id' in points_gdf.columns:
            poi_name_dict = dict(zip(points_gdf['id'], points_gdf['name']))
        else:
            poi_name_dict = dict(zip(points_gdf.index, points_gdf['name']))
        enhanced_gdf['poi_name'] = enhanced_gdf[point_id_col].map(poi_name_dict)
        print(f"添加了POI名称: {enhanced_gdf['poi_name'].notna().sum()} 条记录")
    else:
        print("注意: points_gdf中没有找到 'name' 列")
        enhanced_gdf['poi_name'] = None

    # 4. 如果有街道名称，进行比较分析
    if has_street_col and 'nearest_street_name' in enhanced_gdf.columns:
        # 创建名称比较标志
        enhanced_gdf['street_names_match'] = False
        enhanced_gdf['street_name_similarity'] = 0.0

        # 进行精确匹配
        exact_matches = (enhanced_gdf['original_street_name'].notna() &
                         enhanced_gdf['nearest_street_name'].notna() &
                         (enhanced_gdf['original_street_name'] == enhanced_gdf['nearest_street_name']))
        enhanced_gdf.loc[exact_matches, 'street_names_match'] = True
        enhanced_gdf.loc[exact_matches, 'street_name_similarity'] = 1.0

        # 计算相似度（使用简单的字符串相似度）
        try:
            from difflib import SequenceMatcher

            def calculate_similarity(str1, str2):
                if pd.isna(str1) or pd.isna(str2):
                    return 0.0
                return SequenceMatcher(None, str(str1).lower(), str(str2).lower()).ratio()

            # 对非精确匹配的记录计算相似度
            non_exact = ~exact_matches & enhanced_gdf['original_street_name'].notna() & enhanced_gdf[
                'nearest_street_name'].notna()
            if non_exact.any():
                similarities = enhanced_gdf.loc[non_exact].apply(
                    lambda row: calculate_similarity(row['original_street_name'], row['nearest_street_name']),
                    axis=1
                )
                enhanced_gdf.loc[non_exact, 'street_name_similarity'] = similarities

                # 设置相似度阈值为匹配
                similarity_threshold = 0.8
                high_similarity = non_exact & (enhanced_gdf['street_name_similarity'] >= similarity_threshold)
                enhanced_gdf.loc[high_similarity, 'street_names_match'] = True

        except ImportError:
            print("注意: 无法导入difflib，跳过相似度计算")

        # 报告匹配统计
        total_with_both_names = enhanced_gdf['original_street_name'].notna() & enhanced_gdf[
            'nearest_street_name'].notna()
        if total_with_both_names.any():
            matches = enhanced_gdf.loc[total_with_both_names, 'street_names_match'].sum()
            total = total_with_both_names.sum()
            match_rate = matches / total * 100
            print(f"街道名称匹配统计: {matches}/{total} ({match_rate:.1f}%)")

            if 'street_name_similarity' in enhanced_gdf.columns:
                avg_similarity = enhanced_gdf.loc[total_with_both_names, 'street_name_similarity'].mean()
                print(f"平均相似度: {avg_similarity:.3f}")
    else:
        print("跳过街道名称比较（缺少必要的列）")

    # 5. 添加其他可能有用的POI属性
    other_poi_cols = ['type', 'category', 'address', 'city', 'postcode', 'country']
    for col in other_poi_cols:
        if col in points_gdf.columns:
            if 'id' in points_gdf.columns:
                col_dict = dict(zip(points_gdf['id'], points_gdf[col]))
            else:
                col_dict = dict(zip(points_gdf.index, points_gdf[col]))
            enhanced_gdf[f'poi_{col}'] = enhanced_gdf[point_id_col].map(col_dict)
            non_null_count = enhanced_gdf[f'poi_{col}'].notna().sum()
            if non_null_count > 0:
                print(f"添加了POI {col}: {non_null_count} 条记录")

    # 6. 生成摘要报告
    print("\n=== 增强结果摘要 ===")
    print(f"总记录数: {len(enhanced_gdf)}")
    print(f"有最近街道名称的记录: {enhanced_gdf['nearest_street_name'].notna().sum()}")
    if has_street_col:
        print(f"有原始街道名称的记录: {enhanced_gdf['original_street_name'].notna().sum()}")
        if 'street_names_match' in enhanced_gdf.columns:
            print(f"街道名称匹配的记录: {enhanced_gdf['street_names_match'].sum()}")
    print(f"有POI名称的记录: {enhanced_gdf['poi_name'].notna().sum()}")

    return enhanced_gdf
# def enhance_nearest_streets_with_names(nearest_streets_gdf, points_gdf, lines_gdf,
#                                        point_street_col='street',
#                                        point_id_col='point_id',
#                                        line_id_col='id',
#                                        line_name_col='name'):
#     """
#     为最近街道结果添加街道名称，并与POI原始街道名称进行比较
#
#     参数:
#     nearest_streets_gdf : GeoDataFrame
#         find_nearest_lines_4326函数的输出结果
#     points_gdf : GeoDataFrame
#         包含POI和原始街道名称的GeoDataFrame
#     lines_gdf : GeoDataFrame
#         包含街道ID和名称的线段GeoDataFrame
#     point_street_col : str
#         POI中街道名称的列名
#     point_id_col : str
#         最近街道结果中point_id列名
#     line_id_col : str
#         线段GDF中ID列名
#     line_name_col : str
#         线段GDF中名称列名
#
#     返回:
#     GeoDataFrame: 增强后的最近街道结果，包含名称比较
#     """
#
#     print("=== 增强最近街道结果并进行名称比较 ===")
#     print(f"最近街道记录数: {len(nearest_streets_gdf)}")
#
#     # 创建增强版的结果GDF
#     enhanced_gdf = nearest_streets_gdf.copy()
#
#     # 1. 从lines_gdf获取找到的最近街道名称
#     line_name_dict = dict(zip(lines_gdf[line_id_col], lines_gdf[line_name_col]))
#     enhanced_gdf['nearest_street_name'] = enhanced_gdf['line_id'].map(line_name_dict)
#
#     # 2. 从points_gdf获取POI的原始街道名称
#     if 'id' in points_gdf.columns:
#         point_street_dict = dict(zip(points_gdf['id'], points_gdf[point_street_col]))
#     else:
#         point_street_dict = dict(zip(points_gdf.index, points_gdf[point_street_col]))
#
#     enhanced_gdf['original_street_name'] = enhanced_gdf[point_id_col].map(point_street_dict)
#
#     # 3. 添加POI名称（如果存在）
#     if 'name' in points_gdf.columns:
#         if 'id' in points_gdf.columns:
#             poi_name_dict = dict(zip(points_gdf['id'], points_gdf['name']))
#         else:
#             poi_name_dict = dict(zip(points_gdf.index, points_gdf['name']))
#         enhanced_gdf['poi_name'] = enhanced_gdf[point_id_col].map(poi_name_dict)
#
#     print(f"添加了最近街道名称: {enhanced_gdf['nearest_street_name'].notna().sum()} 条记录")
#     print(f"添加了原始街道名称: {enhanced_gdf['original_street_name'].notna().sum()} 条记录")
#
#     return enhanced_gdf


def calculate_name_similarity(name1, name2):
    """
    计算两个街道名称的相似度

    参数:
    name1, name2 : str
        待比较的街道名称

    返回:
    float: 相似度 (0-1)
    """
    if pd.isna(name1) or pd.isna(name2):
        return 0.0

    # 标准化名称
    def normalize_name(name):
        name = str(name).lower().strip()
        # 移除常见的街道后缀进行比较
        suffixes = ['street', 'st', 'avenue', 'ave', 'road', 'rd', 'drive', 'dr',
                    'boulevard', 'blvd', 'lane', 'ln', 'court', 'ct', 'place', 'pl']
        words = name.split()
        if words and words[-1] in suffixes:
            name = ' '.join(words[:-1])
        return name

    norm_name1 = normalize_name(name1)
    norm_name2 = normalize_name(name2)

    # 使用序列匹配器计算相似度
    similarity = SequenceMatcher(None, norm_name1, norm_name2).ratio()

    return similarity


def classify_street_name_differences(enhanced_gdf, similarity_threshold=0.8):
    """
    对街道名称差异进行分类

    参数:
    enhanced_gdf : GeoDataFrame
        增强后的最近街道结果
    similarity_threshold : float
        相似度阈值，超过此值认为是匹配的

    返回:
    GeoDataFrame: 添加了差异分类的结果
    """

    print("=== 分析街道名称差异 ===")

    # 创建副本
    result_gdf = enhanced_gdf.copy()

    # 计算名称相似度
    result_gdf['name_similarity'] = result_gdf.apply(
        lambda row: calculate_name_similarity(row['original_street_name'], row['nearest_street_name']),
        axis=1
    )

    # 分类差异类型
    def classify_difference(row):
        original = row['original_street_name']
        nearest = row['nearest_street_name']
        similarity = row['name_similarity']
        distance = row['distance']

        # 缺失数据
        if pd.isna(original) and pd.isna(nearest):
            return 'both_missing'
        elif pd.isna(original):
            return 'original_missing'
        elif pd.isna(nearest):
            return 'nearest_missing'

        # 精确匹配
        if original.lower().strip() == nearest.lower().strip():
            return 'exact_match'

        # 高相似度匹配
        if similarity >= similarity_threshold:
            return 'high_similarity_match'

        # 基于距离和相似度的分类
        if distance <= 50:  # 50米内
            if similarity >= 0.5:
                return 'close_location_similar_name'
            else:
                return 'close_location_different_name'
        elif distance <= 200:  # 200米内
            if similarity >= 0.3:
                return 'nearby_location_somewhat_similar'
            else:
                return 'nearby_location_different_name'
        else:  # 200米外
            if similarity >= 0.3:
                return 'distant_location_similar_name'
            else:
                return 'distant_location_different_name'

    result_gdf['difference_type'] = result_gdf.apply(classify_difference, axis=1)

    # 统计各类型数量
    type_counts = result_gdf['difference_type'].value_counts()
    print(f"\n差异类型统计:")
    for diff_type, count in type_counts.items():
        percentage = count / len(result_gdf) * 100
        print(f"  {diff_type}: {count} ({percentage:.1f}%)")

    return result_gdf


def analyze_location_vs_semantic_differences(classified_gdf):
    """
    分析位置差异 vs 语义差异

    参数:
    classified_gdf : GeoDataFrame
        已分类的街道差异数据
    """

    if len(classified_gdf) == 0:
        print("\n=== 位置差异 vs 语义差异分析 ===")
        print("无可分析的数据（classified_gdf为空），跳过此步骤")
        return

    print("\n=== 位置差异 vs 语义差异分析 ===")

    # 定义位置差异类别（主要由距离驱动）
    location_driven = [
        'close_location_different_name',
        'nearby_location_different_name',
        'distant_location_different_name'
    ]

    # 定义语义差异类别（主要由名称差异驱动）
    semantic_driven = [
        'close_location_similar_name',
        'nearby_location_somewhat_similar',
        'distant_location_similar_name'
    ]

    # 统计
    location_issues = classified_gdf[classified_gdf['difference_type'].isin(location_driven)]
    semantic_issues = classified_gdf[classified_gdf['difference_type'].isin(semantic_driven)]
    exact_matches = classified_gdf[classified_gdf['difference_type'] == 'exact_match']
    high_sim_matches = classified_gdf[classified_gdf['difference_type'] == 'high_similarity_match']

    print(f"精确匹配: {len(exact_matches)} ({len(exact_matches) / len(classified_gdf) * 100:.1f}%)")
    print(f"高相似度匹配: {len(high_sim_matches)} ({len(high_sim_matches) / len(classified_gdf) * 100:.1f}%)")
    print(f"位置驱动的差异: {len(location_issues)} ({len(location_issues) / len(classified_gdf) * 100:.1f}%)")
    print(f"语义驱动的差异: {len(semantic_issues)} ({len(semantic_issues) / len(classified_gdf) * 100:.1f}%)")

    # 分析位置差异
    if len(location_issues) > 0:
        print(f"\n位置差异详情:")
        print(f"  平均距离: {location_issues['distance'].mean():.1f} 米")
        print(f"  平均名称相似度: {location_issues['name_similarity'].mean():.3f}")

        print(f"\n位置差异示例 (前5个):")
        for _, row in location_issues.head().iterrows():
            poi_name = row.get('poi_name', 'Unknown POI')
            print(f"    POI: {poi_name}")
            print(f"      原始街道: '{row['original_street_name']}'")
            print(f"      最近街道: '{row['nearest_street_name']}'")
            print(f"      距离: {row['distance']:.1f}m, 相似度: {row['name_similarity']:.3f}")
            print()

    # 分析语义差异
    if len(semantic_issues) > 0:
        print(f"\n语义差异详情:")
        print(f"  平均距离: {semantic_issues['distance'].mean():.1f} 米")
        print(f"  平均名称相似度: {semantic_issues['name_similarity'].mean():.3f}")

        print(f"\n语义差异示例 (前5个):")
        for _, row in semantic_issues.head().iterrows():
            poi_name = row.get('poi_name', 'Unknown POI')
            print(f"    POI: {poi_name}")
            print(f"      原始街道: '{row['original_street_name']}'")
            print(f"      最近街道: '{row['nearest_street_name']}'")
            print(f"      距离: {row['distance']:.1f}m, 相似度: {row['name_similarity']:.3f}")
            print()


def generate_street_matching_report(classified_gdf, output_file=None):
    """
    生成街道匹配报告

    参数:
    classified_gdf : GeoDataFrame
        已分类的结果数据
    output_file : str, optional
        输出文件路径
    """

    report = []
    report.append("# 街道匹配分析报告")
    report.append("=" * 50)

    # 总体统计
    total_records = len(classified_gdf)
    report.append(f"\n## 总体统计")
    report.append(f"总记录数: {total_records}")

    # 匹配质量统计
    exact_matches = len(classified_gdf[classified_gdf['difference_type'] == 'exact_match'])
    high_sim = len(classified_gdf[classified_gdf['difference_type'] == 'high_similarity_match'])

    report.append(f"\n## 匹配质量")
    report.append(f"精确匹配: {exact_matches} ({exact_matches / total_records * 100:.1f}%)")
    report.append(f"高相似度匹配: {high_sim} ({high_sim / total_records * 100:.1f}%)")
    report.append(f"总体良好匹配率: {(exact_matches + high_sim) / total_records * 100:.1f}%")

    # 距离统计
    distances = classified_gdf['distance'].describe()
    report.append(f"\n## 距离统计 (米)")
    report.append(f"平均距离: {distances['mean']:.1f}")
    report.append(f"中位数距离: {distances['50%']:.1f}")
    report.append(f"最大距离: {distances['max']:.1f}")

    # 相似度统计
    similarities = classified_gdf['name_similarity'].describe()
    report.append(f"\n## 名称相似度统计")
    report.append(f"平均相似度: {similarities['mean']:.3f}")
    report.append(f"中位数相似度: {similarities['50%']:.3f}")

    # 差异类型分布
    report.append(f"\n## 差异类型分布")
    type_counts = classified_gdf['difference_type'].value_counts()
    for diff_type, count in type_counts.items():
        percentage = count / total_records * 100
        report.append(f"{diff_type}: {count} ({percentage:.1f}%)")

    # 问题案例
    problem_cases = classified_gdf[
        ~classified_gdf['difference_type'].isin(['exact_match', 'high_similarity_match'])
    ]

    if len(problem_cases) > 0:
        report.append(f"\n## 需要关注的案例 ({len(problem_cases)} 个)")
        report.append("(距离 > 100米 或 相似度 < 0.5 的案例)")

        attention_cases = problem_cases[
            (problem_cases['distance'] > 100) | (problem_cases['name_similarity'] < 0.5)
            ]

        for _, row in attention_cases.head(10).iterrows():
            poi_name = row.get('poi_name', 'Unknown')
            report.append(f"  POI: {poi_name}")
            report.append(f"    原始: {row['original_street_name']}")
            report.append(f"    最近: {row['nearest_street_name']}")
            report.append(f"    距离: {row['distance']:.1f}m, 相似度: {row['name_similarity']:.3f}")
            report.append("")

    report_text = "\n".join(report)

    # 打印报告
    print(report_text)

    # 保存到文件
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report_text)
        print(f"\n报告已保存到: {output_file}")

    return report_text


def complete_street_comparison_analysis(nearest_streets_gdf, points_gdf, lines_gdf,
                                        point_street_col='street',
                                        similarity_threshold=0.8):
    """
    完整的街道比较分析流程

    参数:
    nearest_streets_gdf : GeoDataFrame
        find_nearest_lines_4326的输出结果
    points_gdf : GeoDataFrame
        包含POI的原始数据
    lines_gdf : GeoDataFrame
        街道线段数据
    point_street_col : str
        POI中街道名称列名
    similarity_threshold : float
        相似度阈值

    返回:
    GeoDataFrame: 完整分析结果
    """

    print("=== 开始完整的街道比较分析 ===")

    # 1. 增强最近街道结果
    enhanced_gdf = enhance_nearest_streets_with_names(
        nearest_streets_gdf, points_gdf, lines_gdf, point_street_col
    )

    if len(enhanced_gdf) == 0:
        print("增强结果为空，跳过名称对比分析，直接返回最近街道结果")
        return enhanced_gdf

    if 'name_similarity' not in enhanced_gdf.columns:
        enhanced_gdf['name_similarity'] = np.nan

    has_street_col = (
        point_street_col in points_gdf.columns
        and points_gdf[point_street_col].notna().any()
    )
    if not has_street_col:
        print(f"points_gdf 缺少有效的街道列 '{point_street_col}'，仅返回空间最近结果")
        return enhanced_gdf

    # 2. 分类差异
    classified_gdf = classify_street_name_differences(enhanced_gdf, similarity_threshold)

    if len(classified_gdf) == 0:
        print("分类结果为空，跳过后续分析和报告")
        return classified_gdf

    # 3. 分析位置vs语义差异
    analyze_location_vs_semantic_differences(classified_gdf)

    # 4. 生成报告
    generate_street_matching_report(classified_gdf)

    return classified_gdf


import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import split
import warnings

warnings.filterwarnings('ignore')

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import split
import warnings

warnings.filterwarnings('ignore')


def find_aoi_boundary_roads(streets_gdf, polygons_gdf,
                            buffer_distance=0.0001,  # ~10 meters in degrees
                            overlap_threshold=0.3):
    """
    Find roads that form boundaries of AOIs (Area of Interest)

    Parameters:
    - streets_gdf: GeoDataFrame with street geometries (EPSG:4326)
    - polygons_gdf: GeoDataFrame with AOI polygons (EPSG:4326)
    - buffer_distance: Distance in degrees for boundary matching
    - overlap_threshold: Minimum overlap ratio to consider a road as boundary

    Returns:
    - boundary_roads_gdf: GeoDataFrame with boundary roads and AOI information
    """

    print("Finding AOI boundary roads...")
    print(f"Streets: {len(streets_gdf)}")
    print(f"AOI polygons: {len(polygons_gdf)}")

    # Ensure both GDFs are in EPSG:4326
    if streets_gdf.crs != 'EPSG:4326':
        streets_gdf = streets_gdf.to_crs('EPSG:4326')
    if polygons_gdf.crs != 'EPSG:4326':
        polygons_gdf = polygons_gdf.to_crs('EPSG:4326')

    boundary_roads = []
    processed_count = 0

    for poly_idx, poly_row in polygons_gdf.iterrows():
        aoi_polygon = poly_row.geometry
        aoi_name = poly_row.get('name', f'AOI_{poly_idx}')

        processed_count += 1
        if processed_count % 100 == 0:
            print(f"Processed {processed_count}/{len(polygons_gdf)} AOIs...")

        try:
            # Skip invalid geometries
            if aoi_polygon is None or aoi_polygon.is_empty:
                print(f"Skipping invalid polygon: {aoi_name}")
                continue

            # Get polygon boundary
            boundary = aoi_polygon.boundary

            # Create buffer around polygon boundary for matching
            boundary_buffer = boundary.buffer(buffer_distance)

            # Find streets that intersect with the boundary buffer
            intersecting_streets = streets_gdf[streets_gdf.intersects(boundary_buffer)]

            for street_idx, street_row in intersecting_streets.iterrows():
                street_geom = street_row.geometry

                # Skip invalid street geometries
                if street_geom is None or street_geom.is_empty:
                    continue

                # Calculate overlap metrics
                overlap_info = calculate_boundary_overlap(
                    street_geom, boundary, aoi_polygon, buffer_distance
                )

                if overlap_info['overlap_ratio'] >= overlap_threshold:
                    boundary_road = {
                        'street_id': street_idx,
                        'aoi_id': poly_idx,
                        'aoi_name': aoi_name,
                        'street_name': street_row.get('name', f'Street_{street_idx}'),
                        'overlap_ratio': overlap_info['overlap_ratio'],
                        'overlap_length_meters': overlap_info['overlap_length_meters'],
                        'total_street_length_meters': overlap_info['total_street_length_meters'],
                        'boundary_type': overlap_info['boundary_type'],
                        'geometry': street_geom
                    }

                    # Add existing street attributes if available
                    for col in street_row.index:
                        if col not in ['geometry'] and col not in boundary_road:
                            boundary_road[f'street_{col}'] = street_row[col]

                    # Add polygon attributes
                    for col in poly_row.index:
                        if col not in ['geometry'] and col not in boundary_road:
                            boundary_road[f'aoi_{col}'] = poly_row[col]

                    boundary_roads.append(boundary_road)

        except Exception as e:
            print(f"Error processing AOI {aoi_name}: {e}")
            continue

    # Create GeoDataFrame
    if boundary_roads:
        boundary_roads_gdf = gpd.GeoDataFrame(boundary_roads, crs='EPSG:4326')
        print(f"Found {len(boundary_roads_gdf)} boundary road segments")
        return boundary_roads_gdf
    else:
        print("No boundary roads found")
        return gpd.GeoDataFrame()


def calculate_boundary_overlap(street_geom, aoi_boundary, aoi_polygon, buffer_distance):
    """
    Calculate how much a street overlaps with AOI boundary
    """
    from geopy.distance import geodesic
    from shapely.geometry import LineString, MultiLineString

    try:
        # Create buffer around boundary for intersection
        boundary_buffer = aoi_boundary.buffer(buffer_distance)

        # Find intersection between street and boundary buffer
        intersection = street_geom.intersection(boundary_buffer)

        # Calculate lengths using geodesic distance
        total_street_length = calculate_geodesic_length(street_geom)

        if intersection.is_empty:
            overlap_length = 0
        else:
            overlap_length = calculate_geodesic_length(intersection)

        # Calculate overlap ratio
        overlap_ratio = overlap_length / total_street_length if total_street_length > 0 else 0

        # Determine boundary type
        boundary_type = determine_boundary_type(street_geom, aoi_polygon, buffer_distance)

        return {
            'overlap_ratio': overlap_ratio,
            'overlap_length_meters': overlap_length,
            'total_street_length_meters': total_street_length,
            'boundary_type': boundary_type
        }

    except Exception as e:
        print(f"Warning: Error calculating overlap for geometry: {e}")
        return {
            'overlap_ratio': 0,
            'overlap_length_meters': 0,
            'total_street_length_meters': 0,
            'boundary_type': 'error'
        }


def calculate_geodesic_length(geom):
    """Calculate geodesic length of a geometry (handles multi-part geometries)"""
    from geopy.distance import geodesic
    from shapely.geometry import LineString, MultiLineString

    if geom is None or geom.is_empty:
        return 0

    total_length = 0

    # Handle different geometry types
    if isinstance(geom, LineString):
        # Single LineString
        coords = list(geom.coords)
        if len(coords) < 2:
            return 0

        for i in range(len(coords) - 1):
            p1 = coords[i]
            p2 = coords[i + 1]
            distance = geodesic((p1[1], p1[0]), (p2[1], p2[0])).meters
            total_length += distance

    elif isinstance(geom, MultiLineString):
        # Multiple LineStrings
        for line in geom.geoms:
            total_length += calculate_geodesic_length(line)

    elif hasattr(geom, 'geoms'):
        # Other multi-part geometries
        for sub_geom in geom.geoms:
            if isinstance(sub_geom, (LineString, MultiLineString)):
                total_length += calculate_geodesic_length(sub_geom)

    elif hasattr(geom, 'coords'):
        # Single geometry with coords
        coords = list(geom.coords)
        if len(coords) < 2:
            return 0

        for i in range(len(coords) - 1):
            p1 = coords[i]
            p2 = coords[i + 1]
            distance = geodesic((p1[1], p1[0]), (p2[1], p2[0])).meters
            total_length += distance

    return total_length


def determine_boundary_type(street_geom, aoi_polygon, buffer_distance):
    """
    Determine the type of boundary relationship
    """
    try:
        # Check if street is completely outside
        if not street_geom.intersects(aoi_polygon.buffer(buffer_distance)):
            return 'external'

        # Check if street is completely inside
        if aoi_polygon.contains(street_geom):
            return 'internal'

        # Check if street crosses the boundary
        if street_geom.intersects(aoi_polygon.boundary):
            return 'boundary_crossing'

        # Default case
        return 'boundary_adjacent'

    except Exception as e:
        print(f"Warning: Error determining boundary type: {e}")
        return 'unknown'


def analyze_boundary_roads(boundary_roads_gdf):
    """
    Analyze the boundary roads results
    """
    if len(boundary_roads_gdf) == 0:
        print("No boundary roads to analyze")
        return

    print("=== BOUNDARY ROADS ANALYSIS ===")

    # Overall statistics
    total_roads = len(boundary_roads_gdf)
    unique_aois = boundary_roads_gdf['aoi_name'].nunique()
    unique_streets = boundary_roads_gdf['street_name'].nunique()

    print(f"Total boundary road segments: {total_roads}")
    print(f"Number of AOIs with boundary roads: {unique_aois}")
    print(f"Number of unique streets involved: {unique_streets}")

    # Average overlap statistics
    avg_overlap = boundary_roads_gdf['overlap_ratio'].mean()
    avg_length = boundary_roads_gdf['overlap_length_meters'].mean()

    print(f"Average overlap ratio: {avg_overlap:.2f}")
    print(f"Average overlap length: {avg_length:.1f} meters")

    # Boundary type distribution
    print("\nBoundary type distribution:")
    boundary_types = boundary_roads_gdf['boundary_type'].value_counts()
    for btype, count in boundary_types.items():
        print(f"  {btype}: {count}")

    # AOI-wise analysis
    print("\nPer AOI analysis:")
    aoi_stats = boundary_roads_gdf.groupby('aoi_name').agg({
        'street_name': 'count',
        'overlap_ratio': 'mean',
        'overlap_length_meters': 'sum'
    }).round(2)
    aoi_stats.columns = ['num_boundary_roads', 'avg_overlap_ratio', 'total_boundary_length_m']
    print(aoi_stats)

    return aoi_stats


def filter_boundary_roads(boundary_roads_gdf,
                          min_overlap_ratio=0.5,
                          min_overlap_length=50,
                          boundary_types=['boundary_crossing', 'boundary_adjacent']):
    """
    Filter boundary roads based on criteria
    """
    filtered = boundary_roads_gdf[
        (boundary_roads_gdf['overlap_ratio'] >= min_overlap_ratio) &
        (boundary_roads_gdf['overlap_length_meters'] >= min_overlap_length) &
        (boundary_roads_gdf['boundary_type'].isin(boundary_types))
        ].copy()

    print(f"Filtered from {len(boundary_roads_gdf)} to {len(filtered)} boundary roads")
    return filtered






def find_pois_inside_aois(points_gdf, polygons_gdf,
                          buffer_distance=0,
                          include_boundary=False):
    """
    Find POIs (points) that are completely inside AOIs (polygons)

    Parameters:
    - points_gdf: GeoDataFrame with POI points (EPSG:4326)
    - polygons_gdf: GeoDataFrame with AOI polygons (EPSG:4326)
    - buffer_distance: Buffer distance in degrees (default 0 for exact containment)
    - include_boundary: Whether to include points on polygon boundaries

    Returns:
    - pois_inside_gdf: GeoDataFrame with POIs inside AOIs and AOI information
    """

    print("Finding POIs inside AOIs...")
    print(f"POI points: {len(points_gdf)}")
    print(f"AOI polygons: {len(polygons_gdf)}")

    # Ensure both GDFs are in EPSG:4326
    if points_gdf.crs != 'EPSG:4326':
        points_gdf = points_gdf.to_crs('EPSG:4326')
    if polygons_gdf.crs != 'EPSG:4326':
        polygons_gdf = polygons_gdf.to_crs('EPSG:4326')

    # Prepare results list
    pois_inside = []
    processed_points = 0

    for point_idx, point_row in points_gdf.iterrows():
        point_geom = point_row.geometry
        poi_name = point_row.get('name', f'POI_{point_idx}')

        processed_points += 1
        if processed_points % 1000 == 0:
            print(f"Processed {processed_points}/{len(points_gdf)} points...")

        try:
            # Skip invalid geometries
            if point_geom is None or point_geom.is_empty:
                continue

            # Find which AOIs contain this point
            containing_aois = find_containing_aois(
                point_geom, polygons_gdf, buffer_distance, include_boundary
            )

            # Add entry for each containing AOI
            for aoi_info in containing_aois:
                poi_inside = {
                    'poi_id': point_idx,
                    'aoi_id': aoi_info['aoi_id'],
                    'poi_name': poi_name,
                    'aoi_name': aoi_info['aoi_name'],
                    'containment_type': aoi_info['containment_type'],
                    'distance_to_boundary': aoi_info['distance_to_boundary'],
                    'geometry': point_geom
                }

                # Add existing POI attributes
                for col in point_row.index:
                    if col not in ['geometry'] and col not in poi_inside:
                        poi_inside[f'poi_{col}'] = point_row[col]

                # Add AOI attributes
                aoi_row = polygons_gdf.loc[aoi_info['aoi_id']]
                for col in aoi_row.index:
                    if col not in ['geometry'] and col not in poi_inside:
                        poi_inside[f'aoi_{col}'] = aoi_row[col]

                pois_inside.append(poi_inside)

        except Exception as e:
            print(f"Error processing POI {poi_name}: {e}")
            continue

    # Create GeoDataFrame
    if pois_inside:
        pois_inside_gdf = gpd.GeoDataFrame(pois_inside, crs='EPSG:4326')
        print(f"Found {len(pois_inside_gdf)} POI-AOI containment relationships")
        return pois_inside_gdf
    else:
        print("No POIs found inside AOIs")
        return gpd.GeoDataFrame()


def find_containing_aois(point_geom, polygons_gdf, buffer_distance=0, include_boundary=False):
    """
    Find all AOIs that contain a given point
    """
    from geopy.distance import geodesic

    containing_aois = []

    for poly_idx, poly_row in polygons_gdf.iterrows():
        aoi_polygon = poly_row.geometry
        aoi_name = poly_row.get('name', f'AOI_{poly_idx}')

        try:
            # Skip invalid polygons
            if aoi_polygon is None or aoi_polygon.is_empty:
                continue

            # Apply buffer if specified
            if buffer_distance != 0:
                test_polygon = aoi_polygon.buffer(buffer_distance)
            else:
                test_polygon = aoi_polygon

            # Check containment
            is_inside = False
            containment_type = None

            if include_boundary:
                # Check if point is within or on boundary
                if test_polygon.contains(point_geom) or test_polygon.touches(point_geom):
                    is_inside = True
                    if aoi_polygon.touches(point_geom):
                        containment_type = 'on_boundary'
                    else:
                        containment_type = 'inside'
            else:
                # Check strict containment (not on boundary)
                if test_polygon.contains(point_geom):
                    is_inside = True
                    containment_type = 'inside'

            if is_inside:
                # Calculate distance to boundary
                distance_to_boundary = calculate_distance_to_boundary(point_geom, aoi_polygon)

                containing_aois.append({
                    'aoi_id': poly_idx,
                    'aoi_name': aoi_name,
                    'containment_type': containment_type,
                    'distance_to_boundary': distance_to_boundary
                })

        except Exception as e:
            print(f"Warning: Error checking containment for AOI {aoi_name}: {e}")
            continue

    return containing_aois


def calculate_distance_to_boundary(point_geom, polygon_geom):
    """
    Calculate distance from point to polygon boundary in meters
    """
    from geopy.distance import geodesic

    try:
        # Get point coordinates
        point_coords = (point_geom.y, point_geom.x)  # (lat, lon)

        # Get boundary of polygon
        boundary = polygon_geom.boundary

        # Find closest point on boundary
        closest_point_on_boundary = boundary.interpolate(boundary.project(point_geom))
        closest_coords = (closest_point_on_boundary.y, closest_point_on_boundary.x)

        # Calculate geodesic distance
        distance_meters = geodesic(point_coords, closest_coords).meters

        return distance_meters

    except Exception as e:
        print(f"Warning: Error calculating distance to boundary: {e}")
        return np.nan


def analyze_pois_inside_aois(pois_inside_gdf):
    """
    Analyze the POIs inside AOIs results
    """
    if len(pois_inside_gdf) == 0:
        print("No POIs inside AOIs to analyze")
        return

    print("=== POIs INSIDE AOIs ANALYSIS ===")

    # Overall statistics
    total_relationships = len(pois_inside_gdf)
    unique_pois = pois_inside_gdf['poi_name'].nunique()
    unique_aois = pois_inside_gdf['aoi_name'].nunique()

    print(f"Total POI-AOI relationships: {total_relationships}")
    print(f"Number of unique POIs inside AOIs: {unique_pois}")
    print(f"Number of AOIs containing POIs: {unique_aois}")

    # Containment type distribution
    print(f"\nContainment type distribution:")
    containment_types = pois_inside_gdf['containment_type'].value_counts()
    for ctype, count in containment_types.items():
        print(f"  {ctype}: {count}")

    # Distance to boundary statistics
    valid_distances = pois_inside_gdf['distance_to_boundary'].dropna()
    if len(valid_distances) > 0:
        print(f"\nDistance to boundary statistics (meters):")
        print(f"  Average: {valid_distances.mean():.1f}")
        print(f"  Median: {valid_distances.median():.1f}")
        print(f"  Min: {valid_distances.min():.1f}")
        print(f"  Max: {valid_distances.max():.1f}")

    # AOI-wise analysis
    print(f"\nPer AOI analysis:")
    aoi_stats = pois_inside_gdf.groupby('aoi_name').agg({
        'poi_name': 'count',
        'distance_to_boundary': ['mean', 'min', 'max']
    }).round(1)
    aoi_stats.columns = ['num_pois', 'avg_distance_to_boundary', 'min_distance', 'max_distance']
    print(aoi_stats.head(10))

    # POI type analysis (if available)
    if 'poi_type' in pois_inside_gdf.columns:
        print(f"\nPOI type distribution:")
        poi_types = pois_inside_gdf['poi_type'].value_counts()
        for ptype, count in poi_types.head(10).items():
            print(f"  {ptype}: {count}")

    return aoi_stats


def filter_pois_inside_aois(pois_inside_gdf,
                            min_distance_to_boundary=0,
                            max_distance_to_boundary=float('inf'),
                            containment_types=['inside'],
                            exclude_multiple_aois=False):
    """
    Filter POIs inside AOIs based on criteria
    """
    filtered = pois_inside_gdf.copy()

    # Filter by distance to boundary
    if min_distance_to_boundary > 0 or max_distance_to_boundary < float('inf'):
        distance_mask = (
                (filtered['distance_to_boundary'] >= min_distance_to_boundary) &
                (filtered['distance_to_boundary'] <= max_distance_to_boundary)
        )
        filtered = filtered[distance_mask]

    # Filter by containment type
    if containment_types:
        filtered = filtered[filtered['containment_type'].isin(containment_types)]

    # Exclude POIs that are in multiple AOIs
    if exclude_multiple_aois:
        poi_counts = filtered['poi_id'].value_counts()
        single_aoi_pois = poi_counts[poi_counts == 1].index
        filtered = filtered[filtered['poi_id'].isin(single_aoi_pois)]

    print(f"Filtered from {len(pois_inside_gdf)} to {len(filtered)} POI-AOI relationships")
    return filtered


# Alternative: Simple usage function
def simple_find_pois_inside_aois(points_gdf, polygons_gdf):
    """
    Simplified function for basic POI-inside-AOI finding
    """
    return find_pois_inside_aois(
        points_gdf,
        polygons_gdf,
        buffer_distance=0,
        include_boundary=False
    )


from pathlib import Path


def read_osm_data(input_folder):
    """
    Read OSM data from parquet or geojson files in the specified folder structure.

    Parameters:
    -----------
    input_folder : str
        Path to the folder containing the OSM data files
        Expected structure: input_folder/
                            ├── area_pois.parquet or area_pois.geojson (Polygon geometries)
                            ├── point_pois.parquet or point_pois.geojson (Point geometries)
                            └── roads.parquet or roads.geojson (LineString geometries)

    Returns:
    --------
    tuple : (area_pois_gdf, poi_pois_gdf, roads_gdf)
        Three GeoDataFrames containing the OSM data
    """

    input_path = Path(input_folder)

    # Check if input folder exists
    if not input_path.exists():
        raise FileNotFoundError(f"Input folder not found: {input_folder}")

    print(f"Reading OSM data from: {input_folder}")

    # Read area POIs (Polygons)
    area_pois_path = input_path / "area_pois.parquet"
    area_pois_gdf = None
    if area_pois_path.exists():
        print(f"Reading area POIs from: {area_pois_path}")
        area_pois_gdf = gpd.read_parquet(area_pois_path)
        print(f"Area POIs loaded: {len(area_pois_gdf)} records")
        print(f"Columns: {list(area_pois_gdf.columns)}")
        print(f"Geometry types: {area_pois_gdf.geometry.geom_type.unique()}")
    else:
        area_pois_geojson_path = input_path / "area_pois.geojson"
        if area_pois_geojson_path.exists():
            print(f"Reading area POIs from: {area_pois_geojson_path}")
            area_pois_gdf = gpd.read_file(area_pois_geojson_path)
            print(f"Area POIs loaded: {len(area_pois_gdf)} records")
            print(f"Columns: {list(area_pois_gdf.columns)}")
            print(f"Geometry types: {area_pois_gdf.geometry.geom_type.unique()}")
        else:
            print(f"Warning: area_pois.parquet and area_pois.geojson not found at {input_path}")

    # Read POI POIs (Points)
    poi_pois_path = input_path / "point_pois.parquet"
    poi_pois_gdf = None
    if poi_pois_path.exists():
        print(f"Reading POI POIs from: {poi_pois_path}")
        poi_pois_gdf = gpd.read_parquet(poi_pois_path)
        print(f"POI POIs loaded: {len(poi_pois_gdf)} records")
        print(f"Columns: {list(poi_pois_gdf.columns)}")
        print(f"Geometry types: {poi_pois_gdf.geometry.geom_type.unique()}")
    else:
        poi_pois_geojson_path = input_path / "point_pois.geojson"
        if poi_pois_geojson_path.exists():
            print(f"Reading POI POIs from: {poi_pois_geojson_path}")
            poi_pois_gdf = gpd.read_file(poi_pois_geojson_path)
            print(f"POI POIs loaded: {len(poi_pois_gdf)} records")
            print(f"Columns: {list(poi_pois_gdf.columns)}")
            print(f"Geometry types: {poi_pois_gdf.geometry.geom_type.unique()}")
        else:
            print(f"Warning: point_pois.parquet and point_pois.geojson not found at {input_path}")

    # Read roads (LineStrings)
    roads_path = input_path / "roads.parquet"
    roads_gdf = None
    if roads_path.exists():
        print(f"Reading roads from: {roads_path}")
        roads_gdf = gpd.read_parquet(roads_path)
        print(f"Roads loaded: {len(roads_gdf)} records")
        print(f"Columns: {list(roads_gdf.columns)}")
        print(f"Geometry types: {roads_gdf.geometry.geom_type.unique()}")
    else:
        roads_geojson_path = input_path / "roads.geojson"
        if roads_geojson_path.exists():
            print(f"Reading roads from: {roads_geojson_path}")
            roads_gdf = gpd.read_file(roads_geojson_path)
            print(f"Roads loaded: {len(roads_gdf)} records")
            print(f"Columns: {list(roads_gdf.columns)}")
            print(f"Geometry types: {roads_gdf.geometry.geom_type.unique()}")
        else:
            print(f"Warning: roads.parquet and roads.geojson not found at {input_path}")

    return area_pois_gdf, poi_pois_gdf, roads_gdf


def fill_name_from_other_columns(df):
    """Fill NA values in name column from other name-related columns."""

    # List of name-related columns to check in order of preference
    name_columns = [
        'name', 'official_name', 'alt_name', 'short_name', 'loc_name',
        'name_en', 'name_zh', 'name_de', 'name_fr', 'name_es', 'name_it',
        'name_pt', 'name_ko', 'name_ru', 'name_ja', 'name_zh_Hans', 'name_zh_Hant'
    ]

    df_cleaned = df.copy()

    # Fill NA values in name column
    for idx, row in df_cleaned.iterrows():
        if pd.isna(row['name']) or row['name'] == '':
            # Try to find a non-NA value from other name columns
            for col in name_columns[1:]:  # Skip 'name' itself
                if col in df_cleaned.columns and not pd.isna(row[col]) and row[col] != '':
                    df_cleaned.at[idx, 'name'] = row[col]
                    break

    return df_cleaned


def create_category(df):
    """Create a road category column from highway, bicycle, sidewalk, railway, cycleway."""

    df_cleaned = df.copy()
    df_cleaned['category'] = 'other'

    # Highway categories
    highway_categories = {
        'motorway': 'motorway',
        'trunk': 'major_road',
        'primary': 'major_road',
        'secondary': 'major_road',
        'tertiary': 'major_road',
        'residential': 'residential',
        'service': 'service_road',
        'unclassified': 'minor_road',
        'living_street': 'residential',
        'pedestrian': 'pedestrian',
        'footway': 'pedestrian',
        'path': 'path',
        'track': 'track',
        'bridleway': 'path',
        'steps': 'pedestrian'
    }

    # Assign highway categories
    if 'highway' in df_cleaned.columns:
        for highway_type, category in highway_categories.items():
            mask = df_cleaned['highway'] == highway_type
            df_cleaned.loc[mask, 'category'] = category

    # Bicycle categories
    if 'bicycle' in df_cleaned.columns:
        bicycle_mask = df_cleaned['bicycle'].isin(['yes', 'designated', 'permitted'])
        df_cleaned.loc[bicycle_mask, 'category'] = 'bicycle_road'

    # Cycleway categories
    if 'cycleway' in df_cleaned.columns:
        cycleway_mask = df_cleaned['cycleway'].notna()
        df_cleaned.loc[cycleway_mask, 'category'] = 'cycleway'

    # Sidewalk categories
    if 'sidewalk' in df_cleaned.columns:
        sidewalk_mask = df_cleaned['sidewalk'].isin(['both', 'left', 'right', 'separate'])
        df_cleaned.loc[sidewalk_mask, 'category'] = 'sidewalk'

    # Railway categories
    if 'railway' in df_cleaned.columns:
        railway_mask = df_cleaned['railway'].notna()
        df_cleaned.loc[railway_mask, 'category'] = 'railway'

    # Foot access
    if 'foot' in df_cleaned.columns:
        foot_mask = df_cleaned['foot'].isin(['yes', 'designated', 'permitted'])
        df_cleaned.loc[foot_mask, 'category'] = 'foot_path'

    return df_cleaned


def clean_roads(df, enable_geocoding=True, geocode_missing_only=True, max_geocoding_requests=None,
                sleep_time=1.0, batch_size=50, max_time_hours=24, resume_file=None,
                prioritize_by_distance=True, max_distance_priority=5000):
    """
    Clean roads data and add geocoding information for missing city/street data
    
    Parameters:
    -----------
    df : GeoDataFrame
        Input roads GeoDataFrame
    enable_geocoding : bool
        Whether to enable geocoding for missing information
    geocode_missing_only : bool
        Only geocode items missing city/street information
    max_geocoding_requests : int
        Maximum number of geocoding requests to make
    sleep_time : float
        Sleep time between requests in seconds (default: 1.0 for Nominatim compliance)
    batch_size : int
        Number of requests to process before saving progress
    max_time_hours : float
        Maximum time to spend geocoding in hours
    resume_file : str
        File path to save/load progress for resuming interrupted geocoding
    prioritize_by_distance : bool
        Whether to prioritize items by distance from a reference point
    max_distance_priority : float
        Maximum distance for priority items (in meters)
    
    Returns:
    --------
    GeoDataFrame : Cleaned roads data with geocoded information
    """
    print("=== Cleaning roads data ===")
    
    # Step 1: Fill name column
    print("Step 1: Filling name column from other name-related columns...")
    df = fill_name_from_other_columns(df)
    
    # Step 2: Create category column
    print("Step 2: Creating category column...")
    df = create_category(df)
    
    # Step 3: Add geocoding information for missing city/street data
    if enable_geocoding:
        print("Step 3: Adding geocoding information for missing city/street data...")
        df = add_geocoding_info(df, enable_geocoding=True, 
                               geocode_missing_only=geocode_missing_only,
                               max_geocoding_requests=max_geocoding_requests,
                               sleep_time=sleep_time, batch_size=batch_size,
                               max_time_hours=max_time_hours, resume_file=resume_file,
                               prioritize_by_distance=prioritize_by_distance,
                               max_distance_priority=max_distance_priority)
    else:
        print("Step 3: Geocoding disabled, skipping...")

    # Step 4: Reorder columns
    print("Step 4: Reorganizing columns...")
    important_cols = ['osm_id', 'geometry', 'name', 'category',
                      'highway', 'length_m', 'city', 'street', 'postcode', 'country', 'state', 'county']
    common_cols = [c for c in important_cols if c in df.columns]

    df = df[common_cols]
    df.rename(columns = {'length_m':'length', 'highway':'type'}, inplace=True)
    
    # Print summary statistics
    print(f"\nRoads cleaning completed:")
    print(f"- Total roads: {len(df)}")
    print(f"- Roads with names: {df['name'].notna().sum()}")
    print(f"- Roads with categories: {df['category'].notna().sum()}")
    print(f"- Roads with city info: {df['city'].notna().sum()}")
    print(f"- Roads with street info: {df['street'].notna().sum()}")
    
    return df

def create_category_column(df):
    """Create a category column from common type categories."""

    # Define category mapping based on column names
    category_mapping = {
        'leisure': 'leisure',
        'school': 'education',
        'police': 'emergency',
        'community_centre': 'community',
        'car_rental': 'transport',
        'parking': 'transport',
        'museum': 'culture',
        'healthcare': 'healthcare',
        'shop': 'commercial',
        'amenity': 'amenity',
        'tourism': 'tourism',
        'office': 'business',
        'historic': 'historic',
        'building': 'building',
        'natural': 'natural',
        'sport': 'sport',
        'emergency': 'emergency',
        'public_transport': 'transport',
        'railway': 'transport',
        'aeroway': 'transport',
        'military': 'military',
        'craft': 'craft',
        'emergency_shelter': 'emergency',
        'zoo': 'tourism',
        'hospital_level_CN': 'healthcare',
        'cemetery': 'funeral',
        'place': 'place',
        'government': 'government',
        'diplomatic': 'diplomatic',
        'embassy': 'diplomatic',
        'consulate': 'diplomatic',
        'castle_type': 'historic',
        'heritage': 'historic',
        'archaeological_site': 'historic',
        'memorial': 'historic',
        'tomb': 'historic',
        'ruins': 'historic',
        'theatre': 'culture',
        'theatre_genre': 'culture',
        'theatre_type': 'culture',
        'cinema': 'culture',
        'library': 'education',
        'university': 'education',
        'college': 'education',
        'kindergarten': 'education',
        'preschool': 'education',
        'bank': 'financial',
        'atm': 'financial',
        'post_office': 'service',
        'fire_station': 'emergency',
        'ambulance': 'emergency',
        'pharmacy': 'healthcare',
        'clinic': 'healthcare',
        'dentist': 'healthcare',
        'veterinary': 'healthcare',
        'restaurant': 'food',
        'cafe': 'food',
        'bar': 'food',
        'pub': 'food',
        'fast_food': 'food',
        'cuisine': 'food',
        'hotel': 'accommodation',
        'hostel': 'accommodation',
        'motel': 'accommodation',
        'guest_house': 'accommodation',
        'camp_site': 'accommodation',
        'caravan_site': 'accommodation',
        'club': 'recreation'  # Changed from 'club' to 'recreation' to reduce clubs
    }

    df_cleaned = df.copy()

    # Check if category column already exists
    if 'category' in df_cleaned.columns:
        print("Category column already exists. Filling NA values from other columns...")
        # Only fill NA values, preserve existing non-NA values
        na_mask = df_cleaned['category'].isna() | (df_cleaned['category'] == '')

        # For rows with NA category, assign categories based on non-null values in category columns
        for col, category in category_mapping.items():
            if col in df_cleaned.columns:
                # Only update rows where category is NA and the specific column has a value
                mask = na_mask & df_cleaned[col].notna() & (df_cleaned[col] != '')
                df_cleaned.loc[mask, 'category'] = category
    else:
        print("Creating new category column...")
        df_cleaned['category'] = 'other'  # Default category

        # Assign categories based on non-null values in category columns
        for col, category in category_mapping.items():
            if col in df_cleaned.columns:
                mask = df_cleaned[col].notna() & (df_cleaned[col] != '')
                df_cleaned.loc[mask, 'category'] = category

    return df_cleaned


def create_housenumber_column(df):
    """Create a housenumber column from house number related columns."""

    housenumber_columns = [
        'addr_housenumber', 'addr_housename', 'housenumber', 'house_number',
        'building_number', 'number', 'addr_number'
    ]

    df_cleaned = df.copy()
    df_cleaned['housenumber'] = None

    # Fill housenumber from available columns
    for idx, row in df_cleaned.iterrows():
        for col in housenumber_columns:
            if col in df_cleaned.columns and not pd.isna(row[col]) and row[col] != '':
                df_cleaned.at[idx, 'housenumber'] = str(row[col])
                break

    return df_cleaned


def clean_area_pois(df, enable_geocoding=True, geocode_missing_only=True, max_geocoding_requests=1000,
                    sleep_time=1.0, batch_size=50, max_time_hours=24, resume_file=None,
                    prioritize_by_distance=True, max_distance_priority=5000):
    """
    Main function to clean area_pois data and add geocoding information
    
    Parameters:
    -----------
    df : GeoDataFrame
        Input area POIs GeoDataFrame
    enable_geocoding : bool
        Whether to enable geocoding for missing information
    geocode_missing_only : bool
        Only geocode items missing city/street information
    max_geocoding_requests : int
        Maximum number of geocoding requests to make
    sleep_time : float
        Sleep time between requests in seconds (default: 1.0 for Nominatim compliance)
    batch_size : int
        Number of requests to process before saving progress
    max_time_hours : float
        Maximum time to spend geocoding in hours
    resume_file : str
        File path to save/load progress for resuming interrupted geocoding
    prioritize_by_distance : bool
        Whether to prioritize items by distance from a reference point
    max_distance_priority : float
        Maximum distance for priority items (in meters)
    
    Returns:
    --------
    GeoDataFrame : Cleaned area POIs data with geocoded information
    """
    print("=== Cleaning area POIs data ===")
    print(f"Original columns: {len(df.columns)}")

    # Step 1: Fill name column
    print("\nStep 1: Filling name column from other name-related columns...")
    df = fill_name_from_other_columns(df)

    # Step 2: Create category column
    print("Step 2: Creating category column...")
    df = create_category_column(df)

    # Step 3: Create housenumber column
    print("Step 3: Creating housenumber column...")
    df = create_housenumber_column(df)
    
    # Step 4: Add geocoding information for missing city/street data
    if enable_geocoding:
        print("Step 4: Adding geocoding information for missing city/street data...")
        df = add_geocoding_info(df, enable_geocoding=True, 
                               geocode_missing_only=geocode_missing_only,
                               max_geocoding_requests=max_geocoding_requests,
                               sleep_time=sleep_time, batch_size=batch_size,
                               max_time_hours=max_time_hours, resume_file=resume_file,
                               prioritize_by_distance=prioritize_by_distance,
                               max_distance_priority=max_distance_priority)
    else:
        print("Step 4: Geocoding disabled, skipping...")

    # Step 5: Select and reorder important columns
    print("Step 5: Reorganizing columns...")
    cols = ['osm_id', 'geometry', 'name', 'type', 'category', 'housenumber', 
            'address', 'street', 'city', 'postcode', 'country', 'state', 'county']
    common_cols = [c for c in cols if c in df.columns]

    df = df[common_cols]

    print(f"Cleaned columns: {len(df.columns)}")

    # Print summary statistics
    print("\nSummary:")
    print(f"- Total POIs: {len(df)}")
    print(f"- POIs with names: {df['name'].notna().sum()}")
    print(f"- POIs with categories: {df['category'].notna().sum()}")
    print(f"- POIs with housenumbers: {df['housenumber'].notna().sum()}")
    print(f"- POIs with city info: {df['city'].notna().sum()}")
    print(f"- POIs with street info: {df['street'].notna().sum()}")

    # Category distribution
    print("\nCategory distribution:")
    category_counts = df['category'].value_counts()
    for category, count in category_counts.head(10).items():
        print(f"  {category}: {count}")

    return df
    
import argparse
import pandas as pd
import geopandas as gpd
def main():

    parser = argparse.ArgumentParser(description="Merge OSM batch Parquet files into a single file.")
    parser.add_argument("--place", default='singapore', help="Directory containing the batch Parquet files.")
    parser.add_argument("--enable_geocoding", action='store_true', default=True, 
                       help="Enable geocoding for missing city/street information")
    parser.add_argument("--geocode_missing_only", action='store_true', default=True,
                       help="Only geocode items missing city/street information")
    parser.add_argument("--max_geocoding_requests", type=int, default=1000,
                       help="Maximum number of geocoding requests to make")
    parser.add_argument("--max_roads_geocoding", type=int, default=500,
                       help="Maximum number of geocoding requests for roads")

    args = parser.parse_args()
    
    print(f"=== OSM Data Processing with Geocoding ===")
    print(f"Place: {args.place}")
    print(f"Geocoding enabled: {args.enable_geocoding}")
    print(f"Geocode missing only: {args.geocode_missing_only}")
    print(f"Max geocoding requests (POIs): {args.max_geocoding_requests}")
    print(f"Max geocoding requests (roads): {args.max_roads_geocoding}")
    
    """Reading parquet data"""

    input_folder = f"./data/geo/SR/osm_data/{args.place}"
    import geopandas as gpd
    # area_pois_gdf, poi_pois_gdf, roads_gdf = read_osm_data(input_folder)
    #
    # # Clean data with geocoding
    # print("\n=== Cleaning POI data ===")
    # points = clean_area_pois(poi_pois_gdf,
    #                         enable_geocoding=args.enable_geocoding,
    #                         geocode_missing_only=args.geocode_missing_only,
    #                         max_geocoding_requests=args.max_geocoding_requests)
    #
    # print("\n=== Cleaning area POI data ===")
    # polygons = clean_area_pois(area_pois_gdf,
    #                           enable_geocoding=args.enable_geocoding,
    #                           geocode_missing_only=args.geocode_missing_only,
    #                           max_geocoding_requests=args.max_geocoding_requests)
    #
    # print("\n=== Cleaning roads data ===")
    # roads_gdf = clean_roads(roads_gdf,
    #                        enable_geocoding=args.enable_geocoding,
    #                        geocode_missing_only=args.geocode_missing_only,
    #                        max_geocoding_requests=args.max_roads_geocoding)

    """Get street directions"""

    # streets_gdf = analyze_and_update_streets(
    #         # './data/geo/NewYorkWhole/lines1.geojson',
    #         gdf = roads_gdf,
    #         straightness_threshold=0.75,  # Straightness threshold
    #         max_turns=3,  # Maximum turns
    #         max_vertices=15,  # Maximum vertices
    #         min_length_meters=100  # Minimum length (meters)
    #     )
    # print(streets_gdf.columns)
    # streets_gdf = streets_gdf[['id', 'segment_count', 'original_segment_ids', 'name',
    #        'bearing', 'is_complex', 'complexity_type', 'direction',
    #        'opposite_direction', 'total_angle_change', 'max_angle_change',
    #        'direction_changes', 'geometry', 'straightness', 'turn_count',
    #        'length_meters']]
    # streets_gdf.to_file(f'./data/geo/SR/osm_data/{args.place}/lines_with_directions.geojson')
    # points = gpd.read_file(f'./data/geo/SR/osm_data/{args.place}/points.geojson')
    # print(points['id'].max())
    #
    # polygons = gpd.read_file(f'./data/geo/SR/osm_data/{args.place}/polygons.geojson')
    # print(polygons['id'].max())
    # streets_gdf = gpd.read_file(f'./data/geo/SR/osm_data/{args.place}/lines_with_directions.geojson')
    # streets_gdf =gpd.read_file('./data/geo/SR/spatial/lines_with_directions.geojson')
    # streets_gdf = streets_gdf[(streets_gdf.name.isnull()==False) & (streets_gdf.name.str.contains('非成熟')==False)]
    # streets_gdf.to_file(f'./data/geo/SR/osm_data/{args.place}/lines_with_directions.geojson')
    # print(streets_gdf['id'].max())
    # new_nodes =gpd.read_file( "/root/lanyun-fs/UrbanKG/data/geo/SR/osm_data/newyork/filtered_entities_from_jsonl_osm_attributes.geojson")
    # nodes = gpd.read_file(f'./data/geo/SR/osm_data/{args.place}/nodes_all_add2.geojson')
    nodes = gpd.read_file(f'./data/geo/SR/osm_data/{args.place}/nodes.geojson')
    # existing_names = set(nodes.name.str.lower().dropna().unique())
    # new_nodes = new_nodes[~new_nodes.name.str.lower().isin(existing_names)]
    # new_nodes['id'] = range(max(nodes['id'])+1, max(nodes['id'])+1+len(new_nodes))
    streets_gdf_ =  nodes[nodes.geometry.geom_type.isin(['LineString', 'MultiLineString'])]
    # nodes = nodes.to_crs(epsg=4326)

    import pickle
    # with open(f'./data/geo/SR/osm_data/{args.place}/nodes.pkl', 'rb') as f:
    #     nodes = pickle.load(f)
    # nodes = nodes[(nodes.name.isnull()==False) & (nodes.name != '') ]
    # polygons = nodes[nodes.geometry.geom_type.isin(['Polygon', 'MultiPolygon'])]
    # streets_gdf = nodes[nodes.geometry.geom_type.isin(['LineString', 'MultiLineString'])]
    # nodes_add  = gpd.read_file("/root/lanyun-fs/UrbanKG/data/geo/SR/osm_data/newyork/add_nodes.geojson")
    # all_nodes_add = gpd.read_file("/root/lanyun-fs/UrbanKG/data/geo/SR/osm_data/newyork/all_nodes_add.geojson")
    # poi_aois = all_nodes_add[all_nodes_add.geometry.geom_type.isin(['Point', 'Polygon'])]
    # poi_aois = poi_aois.drop_duplicates()
    # # Filter poi_aois to the same bounds [-75.00,39.00,-72.00,42.00]
    # poi_aois = poi_aois.cx[-75.00:-72.00, 39.00:42.00]
    # new_streets = nodes_add[nodes_add.geometry.geom_type.isin(['LineString', 'MultiLineString'])]
    # existing_names = set(streets_gdf.name.str.lower().dropna().unique())
    # new_streets = new_streets[~new_streets.name.str.lower().isin(existing_names)]
    # new_streets_ = analyze_and_update_streets(
    #         # './data/geo/NewYorkWhole/lines1.geojson',
    #         gdf = new_streets,
    #         straightness_threshold=0.75,  # Straightness threshold
    #         max_turns=3,  # Maximum turns
    #         max_vertices=15,  # Maximum vertices
    #         min_length_meters=100  # Minimum length (meters)
    #     )
    #
    # import pandas as pd
    # streets_gdf_ = pd.concat([streets_gdf,new_streets_])
    # streets_gdf_ = streets_gdf_.drop_duplicates()
    # streets_gdf_['id'] =  range(max(streets_gdf_['id'])+1, max(streets_gdf_['id'])+1+len(streets_gdf_))
    # # Filter out nodes not within bounds [-75.00,39.00,-72.00,42.00]
    # bounds = [-75.00, 39.00, -72.00, 42.00]  # [min_lon, min_lat, max_lon, max_lat]
    # streets_gdf_ = streets_gdf_.cx[bounds[0]:bounds[2], bounds[1]:bounds[3]]
    # points = nodes[nodes.geometry.geom_type.isin(['Point'])]
    # import pandas as pd
    # poi_aois = pd.concat([points,polygons])
    # poi_aois['id'] =  range(max(streets_gdf_['id'])+1, max(streets_gdf_['id'])+1+len(poi_aois))
    # points = poi_aois[poi_aois.geometry.geom_type.isin(['Point'])]
    # polygons = poi_aois[poi_aois.geometry.geom_type.isin(['Polygon', 'MultiPolygon'])]
    # #
    # # points.to_file(f'./data/geo/SR/osm_data/{args.place}/points.geojson')
    # # polygons.to_file(f'./data/geo/SR/osm_data/{args.place}/polygons.geojson')
    # updated_nodes_gdf, updated_crossings_gdf = process_road_crossings_complete(
    #     streets_gdf_,  # 您处理过的街道数据（包含方向信息）
    #     poi_aois,         # 原始节点数据
    #     line_id_col='id',  # 街道ID列名
    #     tolerance=1e-6     # 去重容差
    # )
    # updated_nodes_gdf.to_file(f'./data/geo/SR/osm_data/{args.place}/poi_aois_crossings_add.geojson',  driver = 'GeoJSON')
    # updated_crossings_gdf[['geometry', 'line1_id', 'line2_id', 'line1_name', 'line2_name', 'x',
    #        'y', 'type','crossing_id']].to_file(f'./data/geo/SR/osm_data/{args.place}/street_crossing_edges_add.geojson',  driver = 'GeoJSON')
    # crossing_pairs_gdf = find_crossing_pairs_on_same_street(updated_crossings_gdf)
    # crossing_pairs_gdf_ = crossing_pairs_gdf.merge(streets_gdf[['name', 'id']], how = 'left', left_on= 'common_line_id', right_on='id')
    # crossing_pairs_gdf_ = crossing_pairs_gdf_[['crossing1_id', 'crossing1_x',
    #        'crossing1_y', 'crossing1_other_line_id', 'crossing1_other_street_name',
    #        'crossing2_id', 'crossing2_x', 'crossing2_y', 'crossing2_other_line_id',
    #        'crossing2_other_street_name', 'distance_degrees', 'distance_meters',
    #        'geometry','id','name']]
    # crossing_pairs_gdf['type'] = 'on_same_street'
    # crossing_pairs_gdf_= crossing_pairs_gdf_.rename(columns = {'crossing1_id':'id1', 'crossing2_id':'id2','distance_degrees':'crossing_distance_degrees',
    #                                       'distance_meters':'crossing_distance_meters','id':'common_street_id', 'name':'common_street_name'})
    # crossing_pairs_gdf_.to_file(f'./data/geo/SR/osm_data/{args.place}/crossing_pairs_add.geojson', driver = 'GeoJSON')
    # #
    #
    def convert_to_geodataframe(schools_df,lon,lat):
        """
        Convert schools DataFrame to GeoDataFrame with coordinates

        Parameters:
        -----------
        schools_df : pandas.DataFrame
            DataFrame with latitude and longitude columns

        Returns:
        --------
        geopandas.GeoDataFrame
            GeoDataFrame with point geometries
        """
        # Create point geometries
        schools_df['geometry'] = schools_df.apply(
            lambda row: Point(row[lon], row[lat])
            if pd.notna(row[lon]) and pd.notna(row[lat])
            else None,
            axis=1
        )

        # Convert to GeoDataFrame with specified CRS
        gdf = gpd.GeoDataFrame(
            schools_df,
            geometry='geometry',
            crs="EPSG:4326"  # New York Long Island CRS
        )

        return gdf
    import pickle
    # streets_gdf = gpd.read_file('./data/geo/SR/spatial/lines_with_directions.geojson')
    # with open("./data/geo/NewYorkWhole/spatial/nodes.pkl", 'rb') as f:
    #     nodes = pickle.load(f)
    # with open("./data/geo/NewYorkWhole/spatial/edges.pkl", 'rb') as f:
    #     edges = pickle.load(f)
    # bike_trip = pd.read_csv("./data/geo/JC-202408-citibike-tripdata.csv")
    # bike_trip_edges = bike_trip.groupby(['start_station_name', 'end_station_name']).size().reset_index(name='trip_count')
    # station_start = pd.merge(bike_trip_edges[['start_station_name']], bike_trip[['start_station_name','start_lng','start_lat']], how='left', on = 'start_station_name').rename(columns= {'start_station_name':'station', 'start_lng': 'lon', 'start_lat':'lat'})
    # station_end = pd.merge(bike_trip_edges[['end_station_name']], bike_trip[['end_station_name','end_lng','end_lat']], how='left', on = 'end_station_name').rename(columns= {'end_station_name':'station', 'end_lng': 'lon', 'end_lat':'lat'})
    # bike_trip_stations = pd.concat([station_start,station_end])
    # bike_trip_stations = bike_trip_stations.drop_duplicates()
    # bike_trip_stations = convert_to_geodataframe(bike_trip_stations,'lon','lat')
    # # bike_trip_stations = bike_trip_stations.to_crs("EPSG:32118")
    # bike_trip_stations=bike_trip_stations[['station', 'geometry']]
    # bike_trip_stations['id'] = range(max(nodes['id'])+1, max(nodes['id'])+1+len(bike_trip_stations))
    # bike_trip_stations['type'] = 'bike station'
    #
    # # 步骤 1: 构建 station -> id 的映射字典（可选）
    # station_to_id = dict(zip(bike_trip_stations['station'], bike_trip_stations['id']))
    #
    # # 步骤 2: 将 start_station_name 映射为 id1
    # edges_with_id1 = pd.merge(
    #     bike_trip_edges,
    #     bike_trip_stations[['station', 'id']],
    #     left_on='start_station_name',
    #     right_on='station',
    #     how='left'
    # ).rename(columns={'id': 'id1'}).drop(columns=['station'])
    #
    # # 步骤 3: 将 end_station_name 映射为 id2
    # edges_with_id1_id2 = pd.merge(
    #     edges_with_id1,
    #     bike_trip_stations[['station', 'id']],
    #     left_on='end_station_name',
    #     right_on='station',
    #     how='left'
    # ).rename(columns={'id': 'id2'}).drop(columns=['station'])
    #
    # edges_with_id1_id2 = pd.merge(
    #     edges_with_id1_id2,
    #     bike_trip_stations[[ 'id','geometry']],
    #     left_on='id1',
    #     right_on='id',
    #     how='left'
    # )
    # subway_trip = pd.read_csv('./data/geo/NewYorkWhole/MTA_Subway_Origin-Destination_Ridership_Estimate__Beginning_2025_20250605.csv')
    import pandas as pd
    import geopandas as gpd
    from shapely.geometry import Point
    import pandas as pd

    from shapely import wkt  # Use shapely.wkt to parse WKT strings


    def create_subway_stations_gdf(subway_trip_df):
        """
        Creates a GeoDataFrame of unique subway stations with station names and geometry (Point).

        Parameters:
            subway_trip_df (pd.DataFrame): DataFrame containing trip data with origin/destination station names and POINT geometries as WKT strings or Point objects.

        Returns:
            gpd.GeoDataFrame: GeoDataFrame of unique subway stations.
        """

        # Convert WKT strings to Shapely Point objects if they are strings
        if subway_trip_df['Origin Point'].dtype == 'object':
            # Check if the first non-null value is a string (WKT format)
            first_valid = subway_trip_df['Origin Point'].dropna().iloc[0]
            if isinstance(first_valid, str):
                subway_trip_df['Origin Point'] = subway_trip_df['Origin Point'].apply(wkt.loads)

        if subway_trip_df['Destination Point'].dtype == 'object':
            # Check if the first non-null value is a string (WKT format)
            first_valid = subway_trip_df['Destination Point'].dropna().iloc[0]
            if isinstance(first_valid, str):
                subway_trip_df['Destination Point'] = subway_trip_df['Destination Point'].apply(wkt.loads)

        # Extract origin and destination station info
        origins = subway_trip_df[[
            'Origin Station Complex Name',
            'Origin Point'
        ]].rename(columns={
            'Origin Station Complex Name': 'name',
            'Origin Point': 'geometry'
        })

        destinations = subway_trip_df[[
            'Destination Station Complex Name',
            'Destination Point'
        ]].rename(columns={
            'Destination Station Complex Name': 'name',  # Fixed: changed from 'name' to 'station_name'
            'Destination Point': 'geometry'
        })

        # Concatenate and drop duplicates
        combined = pd.concat([origins, destinations], ignore_index=True)
        combined.drop_duplicates(subset=['name', 'geometry'],
                                 inplace=True)  # Fixed: changed from 'name' to 'station_name'

        # Convert to GeoDataFrame
        subway_stations_gdf = gpd.GeoDataFrame(combined, geometry='geometry', crs='EPSG:4326')
        subway_stations_gdf['type'] = 'subway station'

        return subway_stations_gdf

    def convert_mapillary_jsonl_to_geopandas(jsonl_file_path, output_file_path=None, place=None):
        """
        Convert mapillary_results_cleaned.jsonl to a GeoPandas DataFrame.
        
        Args:
            jsonl_file_path (str): Path to the input JSONL file
            output_file_path (str): Path to save the output GeoPackage file
        
        Returns:
            gpd.GeoDataFrame: The converted GeoPandas DataFrame
        """
        
        # Read JSONL file
        import json
        data = []
        skipped_count = 0
        # city_dict = {'beijing':'beijing',
        #              'chubu':'tokyo',
        #              'ilede':'paris'}
        
        with open(jsonl_file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():  # Skip empty lines

                    try:
                        record = json.loads(line)
                        if place != None and record['city'] != place:
                            continue
                        # Validate geometry structure
                        if 'geometry' not in record or record['url'] == '' or record['url'] == None and record['city'] != place:
                            print(f"Warning: Line {line_num} missing 'geometry' field, skipping")
                            skipped_count += 1
                            continue

                        geometry = record['geometry']

                        # Handle different geometry formats
                        if isinstance(geometry, dict) and 'coordinates' in geometry:
                            # Standard GeoJSON format
                            coords = geometry['coordinates']
                            if isinstance(coords, list) and len(coords) >= 2:
                                data.append(record)
                            else:
                                print(f"Warning: Line {line_num} has invalid coordinates format, skipping")
                                skipped_count += 1
                                continue
                        elif isinstance(geometry, (int, float)):
                            # Handle case where geometry might be a numeric value
                            print(f"Warning: Line {line_num} has geometry as {type(geometry).__name__}, skipping")
                            skipped_count += 1
                            continue
                        else:
                            print(f"Warning: Line {line_num} has unexpected geometry format: {type(geometry)}, skipping")
                            skipped_count += 1
                            continue
                            
                    except json.JSONDecodeError as e:
                        print(f"Warning: Line {line_num} has invalid JSON, skipping: {e}")
                        skipped_count += 1
                        continue
                    except Exception as e:
                        print(f"Warning: Line {line_num} caused error, skipping: {e}")
                        skipped_count += 1
                        continue
        
        print(f"Loaded {len(data)} valid records from {jsonl_file_path}")
        if skipped_count > 0:
            print(f"Skipped {skipped_count} invalid records")
        
        if len(data) == 0:
            raise ValueError("No valid records found in the JSONL file")
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Create geometry column from coordinates
        geometries = []
        for _, row in df.iterrows():
            try:
                coords = row['geometry']['coordinates']
                # GeoJSON coordinates are [longitude, latitude]
                point = Point(coords[0], coords[1])
                geometries.append(point)
            except Exception as e:
                print(f"Warning: Could not create geometry for row {row.get('mapillary_id', 'unknown')}: {e}")
                # Create a default point if geometry creation fails
                geometries.append(Point(0, 0))
        
        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(
            df.drop('geometry', axis=1),  # Remove the original geometry dict
            geometry=geometries,
            crs='EPSG:4326'  # WGS84 coordinate system
        )
        gdf['type'] = 'mapillary'
        gdf.rename(columns = {'id':'node_id'}, inplace=True)
        gdf.rename(columns = {'mapillary_id':'id'}, inplace=True)
        return gdf
    
    def convert_to_geodataframe(csv_file_path, output_file_path=None):
        """
        Convert yfcc4k_newyork.csv to a GeoPandas DataFrame with essential columns
        matching the output of convert_mapillary_jsonl_to_geopandas.
        
        Args:
            csv_file_path (str): Path to the input CSV file
            output_file_path (str, optional): Path to save the output GeoPackage file
        
        Returns:
            gpd.GeoDataFrame: The converted GeoPandas DataFrame with essential columns
        """
        
        # Read CSV file
        df = pd.read_csv(csv_file_path, encoding='utf-8')
        
        print(f"Loaded {len(df)} records from {csv_file_path}")
        
        # Validate required columns
        required_cols = ['Longitude', 'Latitude']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Filter out rows with invalid coordinates
        initial_count = len(df)
        df = df.dropna(subset=['Longitude', 'Latitude'])
        df = df[(df['Longitude'] >= -180) & (df['Longitude'] <= 180)]
        df = df[(df['Latitude'] >= -90) & (df['Latitude'] <= 90)]
        
        filtered_count = initial_count - len(df)
        if filtered_count > 0:
            print(f"Filtered out {filtered_count} rows with invalid coordinates")
        
        if len(df) == 0:
            raise ValueError("No valid records with coordinates found in the CSV file")
        
        # Create geometry column from coordinates
        geometries = []
        skipped_count = 0
        
        for idx, row in df.iterrows():
            try:
                lon = float(row['Longitude'])
                lat = float(row['Latitude'])
                point = Point(lon, lat)
                geometries.append(point)
            except (ValueError, TypeError) as e:
                print(f"Warning: Could not create geometry for row {idx}: {e}")
                geometries.append(Point(0, 0))
                skipped_count += 1
        
        if skipped_count > 0:
            print(f"Warning: {skipped_count} rows had geometry creation issues")
        
        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(
            df,
            geometry=geometries,
            crs='EPSG:4326'  # WGS84 coordinate system
        )
        
        # Add type column
        gdf['type'] = 'yfcc4k'
        
        # Create id column from RecordNumber if it exists, otherwise use index
        if 'RecordNumber' in gdf.columns:
            gdf['id'] = gdf['RecordNumber'].astype(str)
        else:
            gdf['id'] = gdf.index.astype(str)
        
        # Create node_id column (same as id for consistency with mapillary format)
        gdf['node_id'] = gdf['id']
        
        # Map useful columns: Title -> name (if Title exists)
        if 'Title' in gdf.columns:
            gdf['name'] = gdf['Title']
        elif 'name' not in gdf.columns:
            # Create empty name column if neither exists
            gdf['name'] = None
        
        # Keep essential columns similar to mapillary output structure
        # The function will return all columns, but ensure essential ones exist
        print(f"Created GeoDataFrame with {len(gdf)} records")
        print(f"Columns: {list(gdf.columns)}")
        
        # Save to file if output path is provided
        if output_file_path:
            gdf.to_file(output_file_path, driver='GPKG', layer='yfcc4k_points')
            print(f"Saved GeoDataFrame to {output_file_path}")
        
        return gdf
    
    def create_subway_edges_gdf(subway_trip_df, subway_stations_gdf):
        """
        Creates a GeoDataFrame of subway trip edges aggregated by origin-destination pairs.

        Parameters:
            subway_trip_df (pd.DataFrame): DataFrame containing trip data with origin/destination station names and POINT geometries.
            subway_stations_gdf (gpd.GeoDataFrame): GeoDataFrame of stations with 'station_name' and 'id'.

        Returns:
            gpd.GeoDataFrame: GeoDataFrame of subway trip edges with id1, id2, count, and geometry.
        """

        # Ensure that station names are strings (to avoid match issues)
        subway_trip_df['Origin Station Complex Name'] = subway_trip_df['Origin Station Complex Name'].astype(str)
        subway_trip_df['Destination Station Complex Name'] = subway_trip_df['Destination Station Complex Name'].astype(str)

        # Map station names to ids for origin and destination
        # Fixed: changed from 'name' to 'station_name' to match the corrected column name
        name_to_id = dict(zip(subway_stations_gdf['name'], subway_stations_gdf['id']))

        subway_trip_df['id1'] = subway_trip_df['Origin Station Complex Name'].map(name_to_id)
        subway_trip_df['id2'] = subway_trip_df['Destination Station Complex Name'].map(name_to_id)

        # Convert WKT strings to Shapely Point objects if they are strings and not already converted
        if subway_trip_df['Origin Point'].dtype == 'object':
            # Check if the first non-null value is a string (WKT format)
            first_valid = subway_trip_df['Origin Point'].dropna().iloc[0]
            if isinstance(first_valid, str):
                subway_trip_df['Origin Point'] = subway_trip_df['Origin Point'].apply(wkt.loads)

        # Drop rows with missing mappings (if any station name is not found in subway_stations)
        subway_trip_df = subway_trip_df.dropna(subset=['id1', 'id2']).copy()
        subway_trip_df['id1'] = subway_trip_df['id1'].astype(int)
        subway_trip_df['id2'] = subway_trip_df['id2'].astype(int)

        # Group by id1 and id2, count number of trips, keep one example geometry per group
        edges = (
            subway_trip_df.groupby(['id1', 'id2'])
                .agg(
                count=('id1', 'size'),
                geometry=('Origin Point', 'first')  # or use LineString later if needed
            )
                .reset_index()
        )

        # Convert to GeoDataFrame
        edges_gdf = gpd.GeoDataFrame(edges, geometry='geometry', crs='EPSG:4326')

        return edges_gdf

    # bike_trip_stations = gpd.read_file('./data/geo/bike_trip_stations.geojson')
    # bike_trip_stations.to_file('./data/geo/bike_trip_stations.geojson', driver = 'GeoJSON')
    # edges_with_id1_id2 = edges_with_id1_id2[['id1', 'id2','geometry']]
    # edges_with_id1_id2['type'] = 'bike trip'
    # bike_trip_edges= gpd.GeoDataFrame(edges_with_id1_id2, geometry='geometry', crs='EPSG:4326')
    # bike_trip_edges.to_file('./data/geo/bike_trip_edges.geojson', driver = 'GeoJSON')
    # bike_trip_edges = gpd.read_file('./data/geo/bike_trip_edges.geojson')
    # taxi_zone = gpd.read_file("./data/geo/taxi_zones.geojson")
    # print('taxi_zone columns', taxi_zone.columns)
    # taxi_zone = gpd.read_parquet("./data/geo/taxi_zones_4326.parquet") #, engine='pyarrow')
    # taxi_zone['id'] = range(max(bike_trip_stations['id'])+1, max(bike_trip_stations['id'])+1+len(taxi_zone))
    # print(taxi_zone['id'].iloc[-1])
    # taxi_zone['type'] = 'taxi zone'
    # taxi_zone = taxi_zone[['id',  'zone', 'LocationID','geometry', 'type']]
    # taxi_zone.rename(columns = {'zone':'name'}, inplace=True)
    # taxi_zone.to_file("./data/geo/taxi_zones.geojson", driver = "GeoJSON")
    # subway_stations = gpd.read_file("./data/geo/subway_stations.geojson")
    # subway_stations = create_subway_stations_gdf(subway_trip)
    # print(bike_trip_stations.iloc[-1])
    # print(subway_stations.iloc[-1])
    # subway_stations['id'] = range(max(taxi_zone['id'])+1, max(taxi_zone['id'])+1+len(subway_stations))
    # print('subway_trip columns', subway_trip.columns)
    # subway_edges = create_subway_edges_gdf(subway_trip, subway_stations)
    # subway_edges.to_file("./data/geo/subway_edges.geojson", driver = "GeoJSON")
    # subway_edges = gpd.read_file("./data/geo/subway_edges.geojson")

    # subway_stations.to_file("./data/geo/subway_stations.geojson", driver = "GeoJSON")

    # taxi_zone = taxi_zone.to_crs("EPSG:32118")
    # taxi_zone = taxi_zone[['zone', 'LocationID', 'borough','geometry']].rename(columns = {'borough':'Borough', 'zone':'name'})
    # #
    # taxi_edges = pd.read_parquet('./data/geo/taxi_edges.parquet')
    # location_to_id = taxi_zone[['LocationID', 'id']].copy()
    #
    # # 步骤 2: 将 PULocationID 映射为 id1
    # edges_with_id1 = pd.merge(
    #     taxi_edges,
    #     location_to_id.rename(columns={'id': 'id1'}),
    #     left_on='PULocationID',
    #     right_on='LocationID',
    #     how='left'
    # ).drop(columns=['LocationID'])
    #
    # # 步骤 3: 将 DOLocationID 映射为 id2
    # edges_with_id1_id2 = pd.merge(
    #     edges_with_id1,
    #     location_to_id.rename(columns={'id': 'id2'}),
    #     left_on='DOLocationID',
    #     right_on='LocationID',
    #     how='left'
    # ).drop(columns=['LocationID'])
    #
    #
    # edges_with_id1_id2 = pd.merge(
    #     edges_with_id1_id2,
    #     bike_trip_stations[[ 'id','geometry']],
    #     left_on='id1',
    #     right_on='id',
    #     how='left'
    # )
    # edges_with_id1_id2 = edges_with_id1_id2[['id1', 'id2','geometry']]
    # edges_with_id1_id2['type'] = 'taxi trip'
    # taxi_trip_edges= gpd.GeoDataFrame(edges_with_id1_id2, geometry='geometry', crs='EPSG:4326')
    # taxi_trip_edges.to_file('./data/geo/taxi_trip_edges.geojson', driver = 'GeoJSON')
    # taxi_trip_edges = gpd.read_file('./data/geo/taxi_trip_edges.geojson')

    # print(taxi_edges.columns)

    # trip_points = pd.concat([bike_trip_stations,subway_stations])
    """Nearest streets to points"""
    # old_nodes = gpd.read_file(f'./data/geo/SR/osm_data/{args.place}/networkx_graph_nodes.gpkg')
    # old_nodes = old_nodes[~old_nodes.geometry.geom_type.isin(['MultiLineString', 'LineString'])]
    # new_nodes = old_nodes[~old_nodes['name'].isin(nodes['name'])].reset_index(drop=True)
    # new_nodes = new_nodes.drop(['id'], axis=1)
    # new_nodes['id'] = range(max(nodes['id']) + 1, max(nodes['id']) + 1 + len(new_nodes))
    # nodes.to_file(f'./data/geo/SR/osm_data/{args.place}/nodes_citygpt.geojson', driver='GeoJSON')
    #
    # nodes = pd.concat([nodes, new_nodes])
    # nodes.to_file(f'./data/geo/SR/osm_data/{args.place}/nodes_new.geojson', driver='GeoJSON')
    # mapillary_gdf1 = convert_mapillary_jsonl_to_geopandas(
    #     f'./data/geo/SR/osm_data/{args.place}/mapillary_results_cleaned1.jsonl')
    # mapillary_gdf2 = convert_mapillary_jsonl_to_geopandas(
    #     f'./data/geo/SR/osm_data/{args.place}/mapillary_results_cleaned.jsonl')
    # # mapillary_gdf2 = convert_mapillary_jsonl_to_geopandas(f'./data/geo/SR/osm_data/{args.place}/mapillary_results_merged_filtered.jsonl')
    # mapillary_gdf = pd.concat([mapillary_gdf1, mapillary_gdf2])
    # mapillary_gdf = mapillary_gdf.drop_duplicates(['id'])
    # mapillary_gdf = convert_mapillary_jsonl_to_geopandas("/home/xingtong/UrbanKG/data/geo/SR/osm_data/paris/filtered_perception_by_coordinates_paris_with_urls.jsonl", place=args.place)
    mapillary_gdf = convert_mapillary_jsonl_to_geopandas(f'./data/geo/SR/osm_data/{args.place}/mapillary_results_cleaned_urbanllava.jsonl')

    mapillary_gdf.to_file(f'./data/geo/SR/osm_data/{args.place}/nodes_mapillary_urbanllava.geojson', driver = 'GeoJSON')
    # mapillary_gdf = gpd.read_file(f'./data/geo/SR/osm_data/{args.place}/nodes_mapillary_urbanllava.geojson')
    # urbanllava_nodes = gpd.read_file(f'./data/geo/SR/osm_data/{args.place}/aoi2coor_geodataframe.geojson')
    # polygons = new_nodes[new_nodes.geometry.geom_type.isin(['MultiPolygon', 'Polygon'])]
    # points = new_nodes[new_nodes.geometry.geom_type.isin(['Point', 'MultiPoint'])]
    # points = mapillary_nodes
    import ast
    def csv_to_geopandas(csv_path, crs='EPSG:4326'):
        """
        Convert CSV file to GeoPandas DataFrame.

        Args:
            csv_path (str): Path to the CSV file
            crs (str): Coordinate reference system (default: EPSG:4326 for WGS84)

        Returns:
            geopandas.GeoDataFrame: GeoPandas DataFrame with geometry and attributes
        """
        # Read the CSV file
        df = pd.read_csv(csv_path)

        # Function to parse coordinates from string format
        def parse_coords(coord_str):
            try:
                # Remove brackets and split by comma
                # Handle the format: "[(lon, lat)]"
                coord_str = coord_str.strip()
                if coord_str.startswith('[') and coord_str.endswith(']'):
                    coord_str = coord_str[1:-1]  # Remove outer brackets
                    # Parse the tuple string
                    coords = ast.literal_eval(coord_str)
                    if isinstance(coords, tuple) and len(coords) == 2:
                        return Point(coords[0], coords[1])  # (lon, lat) -> Point(lon, lat)
                return None
            except (ValueError, SyntaxError, TypeError):
                return None

        # Create geometry column from coords
        df['geometry'] = df['coords'].apply(parse_coords)

        # Filter out rows where geometry creation failed
        df = df.dropna(subset=['geometry'])

        # Select only the columns we need and rename English_Address to address
        if 'aoi_name' in df.columns:
            result_df = df[['aoi_name','geometry']].copy()
            result_df = result_df.rename(columns={'aoi_name': 'name'})
        else:
            result_df = df[['name', 'English_Address', 'geometry']].copy()
            result_df = result_df.rename(columns={'English_Address': 'address'})

        # Create GeoPandas DataFrame
        gdf = gpd.GeoDataFrame(result_df, geometry='geometry', crs=crs)

        return gdf

    # citypgt_train_points = gpd.read_file("/root/lanyun-fs/UrbanKG/data/geo/SR/osm_data/newyork/filtered_entities_osm_attributes.geojson")
    # citypgt_train_points1 = gpd.read_file("/root/lanyun-fs/UrbanKG/data/geo/SR/osm_data/newyork/missing_streets_osm_results.geojson")
    # yfcc4k_gdf = convert_to_geodataframe("/home/xingtong/UrbanKG/yfcc4k_newyork_with_s2tokens.csv")
    # yfcc4k_gdf = yfcc4k_gdf.drop(['id'],axis=1)
    # yfcc4k_gdf.rename(columns= {'Secret2':'id'},inplace=True)
    # yfcc4k_gdf.to_file(f'./data/geo/SR/osm_data/{args.place}/nodes_mapillary_perception.geojson', driver='GeoJSON')
    # # citypgt_train_points = csv_to_geopandas(f"./data/geo/SR/osm_data/{args.place}/NewYork_aois.csv")
    # # citypgt_train_points1 = csv_to_geopandas(f"./data/geo/SR/osm_data/{args.place}/NewYork_aois_visual.csv")
    # existing_names = set(nodes.name.str.lower().dropna().unique())
    # citypgt_train_points = citypgt_train_points[~citypgt_train_points.name.str.lower().isin(existing_names)]
    # citypgt_train_points1 = citypgt_train_points1[~citypgt_train_points1.name.str.lower().isin(existing_names)]
    # #
    # citypgt_train_points = pd.concat([citypgt_train_points,citypgt_train_points1])
    # citypgt_train_points['id'] = range(max(nodes['id']) + 1, max(nodes['id']) + 1 + len(citypgt_train_points))
    # citypgt_train_points.to_file("/root/lanyun-fs/UrbanKG/data/geo/SR/osm_data/newyork/add_nodes.geojson", driver="GeoJSON")

    # Enrich citypgt_train_points with OSM data using Overpass API
    # print("Enriching citypgt_train_points with OSM data using Overpass API...")
    # try:
    #     from osm_overpass_query import enrich_citypgt_points_with_osm_overpass
    #
    #
    #     # Enrich points with OSM data
    #     citypgt_train_points = enrich_citypgt_points_with_osm_overpass(
    #         citypgt_train_points
    #     )
    #     citypgt_train_points['id'] = range(max(nodes['id']) + 1, max(nodes['id']) + 1 + len(citypgt_train_points))
    #
    #     # print(f"OSM enrichment completed. Enriched points shape: {citypgt_train_points.shape}")
    #     #
    #     # # Show OSM enrichment statistics
    #     # osm_columns = [col for col in citypgt_train_points.columns if col.startswith('osm_')]
    #     # if osm_columns:
    #     #     print("OSM enrichment statistics:")
    #     #     for col in osm_columns:
    #     #         non_null_count = citypgt_train_points[col].notna().sum()
    #     #         print(f"  {col}: {non_null_count}/{len(citypgt_train_points)} points have data")
    #
    #     # Save enriched dataset
    #     enriched_output_path = f'./data/geo/SR/osm_data/{args.place}/citypgt_train_points_with_osm.geojson'
    #     citypgt_train_points.to_file(enriched_output_path, driver="GeoJSON")
    #     print(f"Enriched dataset saved to: {enriched_output_path}")
    #
    # except ImportError as e:
    #     print(f"Warning: Could not import OSM Overpass query module: {e}")
    #     print("Continuing without OSM enrichment...")
    # except Exception as e:
    #     print(f"Warning: OSM enrichment failed: {e}")
    #     print("Continuing without OSM enrichment...")
    # nearest_streets = find_nearest_lines_4326(trip_points,streets_gdf)

    """Nearest line by calcaulation and by street names"""
    points = mapillary_gdf
    nearest_streets = find_nearest_lines_4326(points,streets_gdf_)

    analysis_results = complete_street_comparison_analysis(
        nearest_streets, points, streets_gdf_
    )
    # Convert the nearest_line_point column from Shapely Point to WKT string
    analysis_results['nearest_line_point'] = analysis_results['nearest_line_point'].apply(
        lambda x: x.wkt if hasattr(x, 'wkt') and x is not None else None
    )
    #
    # # Now save to GeoJSON - this should work without errors
    analysis_results = analysis_results[['point_id', 'line_id', 'distance', 'bearing', 'point_geometry',
                          'nearest_line_point', 'point_x', 'point_y', 'nearest_x', 'nearest_y',
                          'nearest_street_name', 'original_street_name', 'poi_name',
                          'name_similarity']]
    analysis_results['type'] ='nearest'
    analysis_results.rename(columns ={'point_id': 'id1', 'line_id': 'id2'}, inplace=True)
    # analysis_results.to_file(f'./data/geo/SR/osm_data/{args.place}/nearest_edges_add1.geojson', driver="GeoJSON")
    # all_nodes= pd.concat([nodes, citypgt_train_points])
    # all_nodes.to_file(f'./data/geo/SR/osm_data/{args.place}/all_nodes_add.geojson', driver="GeoJSON")

    # edges_mapillary = gpd.read_file(f'./data/geo/SR/osm_data/{args.place}/edges_mapillary.geojson')
    # old_nodes = gpd.read_file(f'./data/geo/SR/osm_data/{args.place}/edges_mapillary.geojson')



    # analysis_results.to_file(f'./data/geo/SR/osm_data/{args.place}/edges_mapillary_yfcc4k.geojson', driver="GeoJSON")
    analysis_results.to_file(f'./data/geo/SR/osm_data/{args.place}/edges_mapillary_urbanllava.geojson', driver="GeoJSON")
    # analysis_results = complete_street_comparison_analysis(
    #     nearest_streets, trip_points, streets_gdf
    # )
    # # Convert the nearest_line_point column from Shapely Point to WKT string
    # analysis_results['nearest_line_point'] = analysis_results['nearest_line_point'].apply(
    #     lambda x: x.wkt if hasattr(x, 'wkt') and x is not None else None
    # )

    # # Now save to GeoJSON - this should work without errors
    # print('analysis_results columns', analysis_results.columns)
    # analysis_results[['point_id', 'line_id', 'distance', 'bearing', 'point_geometry',
    #                       'nearest_line_point', 'point_x', 'point_y', 'nearest_x', 'nearest_y',
    #                       'nearest_street_name', 'original_street_name', 'poi_name',
    #                       'name_similarity']].to_file(f'./data/geo/SR/osm_data/{args.place}/nearest_line2points_edges_trip.geojson', driver="GeoJSON")
    """AOI boundary roads"""
    # Find boundary roads
    # polygons = citypgt_train_points[citypgt_train_points.geometry.geom_type.isin(['Polygon', 'MultiPolygon'])]
    # boundary_roads = find_aoi_boundary_roads(
    #     streets_gdf_,
    #     polygons,
    #     buffer_distance=0.0001,  # ~10 meters
    #     overlap_threshold=0.3  # 30% overlap minimum
    # )
    # boundary_roads[['street_id', 'aoi_id', 'aoi_name', 'street_name', 'overlap_ratio',
    #        'overlap_length_meters', 'total_street_length_meters', 'boundary_type',
    #        'geometry']].to_file(f'./data/geo/SR/osm_data/{args.place}/aoi_boundary_roads_add.geojson')



    # subway_edges=gpd.read_file("./data/geo/subway_edges.geojson")
    #
    # aoi_boundary_roads_taxi = gpd.read_file('./data/geo/SR/spatial/aoi_boundary_roads_taxi.geojson')
    # nearest_line2points_edges_trip = gpd.read_file('./data/geo/SR/spatial/nearest_line2points_edges_trip.geojson')
    # print('bike_trip_stations columns', bike_trip_stations.columns)
    # print('subway_stations columns', subway_stations.columns)
    # print('taxi_zone columns', taxi_zone.columns)
    # bike_trip_stations = bike_trip_stations.to_crs('EPSG:4326')
    # subway_stations = subway_stations.to_crs('EPSG:4326')
    # taxi_zone = taxi_zone.to_crs('EPSG:4326')
    # nodes1 = pd.concat([nodes, bike_trip_stations,subway_stations,taxi_zone])

    # taxi_trip_edges['type'] = 'taxi trip'
    # bike_trip_edges['type'] = 'bike trip'
    # subway_edges['type'] = 'subway trip'
    # print('taxi_trip_edges columns', taxi_trip_edges.columns)
    # print('bike_trip_edges columns', bike_trip_edges.columns)
    # print('subway_edges columns', subway_edges.columns)
    # taxi_trip_edges = taxi_trip_edges.to_crs('EPSG:4326')
    # bike_trip_edges = bike_trip_edges.to_crs('EPSG:4326')
    # subway_edges = subway_edges.to_crs('EPSG:4326')
    # edges1 = pd.concat([edges,taxi_trip_edges, bike_trip_edges,subway_edges])

    # with open("./data/geo/NewYorkWhole/spatial/nodes1.pkl", 'wb') as f:
    #     pickle.dump(nodes1, f)
    # # print("Saved nodes as pickle file for future use")
    # with open("./data/geo/NewYorkWhole/spatial/edges1.pkl", 'wb') as f:
    #     pickle.dump(edges1, f)
    # print("Saved nodes as pickle file for future use")
    """"POI inside AOI"""
    # pois_inside = find_pois_inside_aois(
    #     points,
    #     polygons,
    #     buffer_distance=0.00001,  # 小缓冲区（约1米）
    #     include_boundary=False     # 包含边界上的点
    # )
    # pois_inside.to_file('./data/geo/SR/spatial/pois_inside_aois.geojson')




    # 方法1b: 使用标准pickle

    #
    # from networkx_graph import create_networkx_from_gdfs
    #
    # """Reading nodes and edges and create networkx Graph"""
    # #
    # edges =gpd.read_file(f'./data/geo/SR/osm_data/{args.place}/edges_all_add1.geojson')
    # edges_path= '/root/lanyun-fs/UrbanKG/data/geo/SR/osm_data/newyork/edges_all.parquet'
    # edges = pd.read_parquet(edges_path)
    # poi_aois_crossings=  gpd.read_file(f'./data/geo/SR/osm_data/{args.place}/poi_aois_crossings_add.geojson')
    # aoi_boundary_roads = gpd.read_file(f'./data/geo/SR/osm_data/{args.place}/aoi_boundary_roads_add.geojson')
    # # # #
    # # nearest_line2points_edges = pd.concat([nearest_line2points_edges1, analysis_results])
    # street_crossing_edges = gpd.read_file(f'./data/geo/SR/osm_data/{args.place}/street_crossing_edges_add.geojson')
    # street_crossing_edges.rename(columns = {'line1_id':'id1','line2_id':'id2'},inplace=True)
    # street_crossing_edges=street_crossing_edges[['id1','id2', 'x', 'y', 'type','crossing_id', 'geometry']]
    # street_crossing_edges=street_crossing_edges[['id1','id2', 'x', 'y', 'type','crossing_id', 'geometry']]
    # # # #
    # crossing_pairs = gpd.read_file(f'./data/geo/SR/osm_data/{args.place}/crossing_pairs_add.geojson')
    # crossing_pairs['type'] = 'on_same_street'
    # crossing_pairs=crossing_pairs[['id1','id2','common_street_id','geometry','type','crossing_distance_meters', 'crossing1_x', 'crossing1_y', 'crossing2_x', 'crossing2_y',]]
    # # aoi_boundary_roads = pd.concat([aoi_boundary_roads1,boundary_roads])
    # aoi_boundary_roads.rename(columns = {'street_id':'id1','aoi_id':'id2', 'boundary_type':'type'},inplace=True)
    # aoi_boundary_roads = aoi_boundary_roads[['id1', 'id2', 'type','geometry']]
    # aoi_boundary_roads.loc[aoi_boundary_roads['type']=='internal', 'type'] = 'intersects'
    #
    # nearest_line2points_edges = gpd.read_file(f'./data/geo/SR/osm_data/{args.place}/nearest_edges_add1.geojson')
    # nearest_line2points_edges.rename(columns = {'point_id':'id1','line_id':'id2'},inplace=True)
    # nearest_line2points_edges=nearest_line2points_edges[['id1','id2', 'distance', 'bearing',  'point_x', 'point_y', 'nearest_x', 'nearest_y',
    #        'original_street_name','nearest_street_name', 'name_similarity', 'geometry']]
    # nearest_line2points_edges['type'] = 'nearest'
    # #
    # # #
    # # # #
    # edges = gpd.read_file(f'./data/geo/SR/osm_data/{args.place}/edges_all_add.geojson')
    # # # edges = pd.concat([edges,aoi_boundary_roads,nearest_line2points_edges ])
    # # # edges.to_parquet(f'./data/geo/SR/osm_data/{args.place}/edges_all_add.parquet')
    # # edges = pd.concat([crossing_pairs,aoi_boundary_roads,nearest_line2points_edges,street_crossing_edges])
    # edges.to_parquet(f'./data/geo/SR/osm_data/{args.place}/edges_all_add1.parquet')
    # edges.to_file(f'./data/geo/SR/osm_data/{args.place}/edges_all_add1.geojson', driver="GeoJSON")
    # # # # # # streets_gdf = gpd.read_file(f'./data/geo/SR/osm_data/{args.place}/lines_with_directions.geojson')
    # nodes = pd.concat([nodes,new_nodes])
    # nodes.to_file(f'./data/geo/SR/osm_data/{args.place}/nodes_all_add2.geojson', driver="GeoJSON")



# Usage Examples:
if __name__ == "__main__":
    main()

