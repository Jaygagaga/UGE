# Copyright (c) Alibaba, Inc. and its affiliates.
import base64
import collections
import math
import os
import random
import re
from io import BytesIO
from typing import Any, Callable, List, TypeVar, Union

import numpy as np
import requests
import torch
from PIL import Image
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from swift.utils import get_env_args

# >>> internvl
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _build_transform(input_size):
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def _find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def _dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set((i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1)
                        if min_num <= i * j <= max_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = _find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = ((i % (target_width // image_size)) * image_size, (i // (target_width // image_size)) * image_size,
               ((i % (target_width // image_size)) + 1) * image_size, ((i //
                                                                        (target_width // image_size)) + 1) * image_size)
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


# <<< internvl


def rescale_image(img: Image.Image, max_pixels: int) -> Image.Image:
    import torchvision.transforms as T
    width = img.width
    height = img.height
    if max_pixels is None or max_pixels <= 0 or width * height <= max_pixels:
        return img

    ratio = width / height
    height_scaled = math.sqrt(max_pixels / ratio)
    width_scaled = height_scaled * ratio
    return T.Resize((int(height_scaled), int(width_scaled)))(img)


_T = TypeVar('_T')


def load_file(path: Union[str, bytes, _T]) -> Union[BytesIO, _T]:
    res = path
    if isinstance(path, str):
        path = path.strip()
        if path.startswith('http'):
            retries = Retry(total=3, backoff_factor=1, allowed_methods=['GET'])
            with requests.Session() as session:
                session.mount('http://', HTTPAdapter(max_retries=retries))
                session.mount('https://', HTTPAdapter(max_retries=retries))

                timeout = float(os.getenv('SWIFT_TIMEOUT', '20'))
                request_kwargs = {'timeout': timeout} if timeout > 0 else {}

                response = session.get(path, **request_kwargs)
                response.raise_for_status()
                content = response.content
                res = BytesIO(content)

        elif os.path.exists(path) or (not path.startswith('data:') and len(path) <= 200):
            ROOT_IMAGE_DIR = get_env_args('ROOT_IMAGE_DIR', str, None)
            if ROOT_IMAGE_DIR is not None and not os.path.exists(path):
                path = os.path.join(ROOT_IMAGE_DIR, path)
            path = os.path.abspath(os.path.expanduser(path))
            with open(path, 'rb') as f:
                res = BytesIO(f.read())
        else:  # base64_str
            data = path
            if data.startswith('data:'):
                match_ = re.match(r'data:(.+?);base64,(.+)', data)
                assert match_ is not None
                data = match_.group(2)
            data = base64.b64decode(data)
            res = BytesIO(data)
    elif isinstance(path, bytes):
        res = BytesIO(path)
    return res


def load_image(image: Union[str, bytes, Image.Image]) -> Image.Image:
    image = load_file(image)
    if isinstance(image, BytesIO):
        image = Image.open(image)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return image


def load_batch(path_list: List[Union[str, None, Any, BytesIO]],
               load_func: Callable[[Any], _T] = load_image) -> List[_T]:
    res = []
    assert isinstance(path_list, (list, tuple)), f'path_list: {path_list}'
    for path in path_list:
        if path is None:  # ignore None
            continue
        res.append(load_func(path))
    return res


def load_video_hf(videos: List[str]):
    from transformers.video_utils import load_video
    res = []
    video_metadata = []
    for video in videos:
        if isinstance(video, (list, tuple)) and isinstance(video[0], str):
            # Case a: Video is provided as a list of image file names
            video = [np.array(load_image(image_fname)) for image_fname in video]
            video = np.stack(video)
            metadata = None
        else:
            # Case b: Video is provided as a single file path or URL or decoded frames in a np.ndarray or torch.tensor
            video_load_backend = get_env_args('video_load_backend', str, 'pyav')
            video, metadata = load_video(
                video,
                backend=video_load_backend,
            )
        res.append(video)
        video_metadata.append(metadata)
    return res, video_metadata


def _get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array(
        [int(start_idx + (seg_size / 2) + np.round(seg_size * idx)) for idx in range(num_segments)])
    return frame_indices


def transform_image(image, input_size=448, max_num=12):
    transform = _build_transform(input_size=input_size)
    images = _dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def load_video_internvl(video: Union[str, bytes], bound=None, num_segments=32):
    from decord import VideoReader, cpu
    video_io = load_file(video)
    vr = VideoReader(video_io, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    images = []
    frame_indices = _get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    for frame_index in frame_indices:
        images.append(Image.fromarray(vr[frame_index].asnumpy()).convert('RGB'))
    return images


def load_video_cogvlm2(video: Union[str, bytes]) -> np.ndarray:
    from decord import cpu, VideoReader, bridge
    video_io = load_file(video)
    bridge.set_bridge('torch')
    clip_end_sec = 60
    clip_start_sec = 0
    num_frames = get_env_args('num_frames', int, 24)
    decord_vr = VideoReader(video_io, ctx=cpu(0))
    duration = len(decord_vr)  # duration in terms of frames
    start_frame = int(clip_start_sec * decord_vr.get_avg_fps())
    end_frame = min(duration, int(clip_end_sec * decord_vr.get_avg_fps())) if \
        clip_end_sec is not None else duration
    frame_id_list = np.linspace(start_frame, end_frame - 1, num_frames, dtype=int)
    video_data = decord_vr.get_batch(frame_id_list)
    video_data = video_data.permute(3, 0, 1, 2)
    return video_data


def load_video_llava(video: Union[str, bytes]) -> np.ndarray:
    import av
    video_io = load_file(video)
    container = av.open(video_io)
    total_frames = container.streams.video[0].frames
    num_frames = get_env_args('num_frames', int, 16)
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format='rgb24') for x in frames])


def load_video_minicpmv_mplug_owl3(video: Union[str, bytes], max_num_frames):
    from decord import VideoReader, cpu  # pip install decord

    def uniform_sample(_l, _n):
        gap = len(_l) / _n
        idxs = [int(i * gap + gap / 2) for i in range(_n)]
        return [_l[i] for i in idxs]

    video_io = load_file(video)
    vr = VideoReader(video_io, ctx=cpu(0))
    sample_fps = round(vr.get_avg_fps() / 1)  # FPS
    frame_idx = [i for i in range(0, len(vr), sample_fps)]

    if len(frame_idx) > max_num_frames:
        frame_idx = uniform_sample(frame_idx, max_num_frames)
    frames = vr.get_batch(frame_idx).asnumpy()
    frames = [Image.fromarray(v.astype('uint8')) for v in frames]
    return frames


def load_audio(audio: Union[str, bytes], sampling_rate: int, return_sr: bool = False):
    import librosa
    audio_io = load_file(audio)
    res = librosa.load(audio_io, sr=sampling_rate)
    return res if return_sr else res[0]


def load_video_valley(video: Union[str, bytes]):
    import decord
    from torchvision import transforms
    video_io = load_file(video)
    video_reader = decord.VideoReader(video_io)
    decord.bridge.set_bridge('torch')
    video = video_reader.get_batch(np.linspace(0, len(video_reader) - 1, 8).astype(np.int_)).byte()
    images = [transforms.ToPILImage()(image.permute(2, 0, 1)).convert('RGB') for image in video]
    return images


def load_video_ovis2(video_path, num_frames):
    from moviepy.editor import VideoFileClip
    with VideoFileClip(video_path) as clip:
        total_frames = int(clip.fps * clip.duration)
        if total_frames <= num_frames:
            sampled_indices = range(total_frames)
        else:
            stride = total_frames / num_frames
            sampled_indices = [
                min(total_frames - 1, int((stride * i + stride * (i + 1)) / 2)) for i in range(num_frames)
            ]
        frames = [clip.get_frame(index / clip.fps) for index in sampled_indices]
        frames = [Image.fromarray(frame, mode='RGB') for frame in frames]
    return frames


def load_video_ovis2_5(video_path, num_frames):
    from moviepy.editor import VideoFileClip
    with VideoFileClip(video_path) as clip:
        total_frames = int(clip.fps * clip.duration)
        indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
        frames = [Image.fromarray(clip.get_frame(t)) for t in (idx / clip.fps for idx in indices)]
    return frames


def validate_graph_file(pkl_file: str) -> bool:
    """
    Validate that a graph pickle file exists and can be loaded successfully.
    This is used to filter out invalid samples during dataset preprocessing.

    Simplified version: assumes graphs are cleaned and standardized.
    Graph paths should be direct paths to cleaned graph files.

    Args:
        pkl_file: Path to pickle file

    Returns:
        True if file is valid and can be loaded, False otherwise
    """
    import pickle
    from swift.utils import get_logger
    logger = get_logger()

    if pkl_file is None or pkl_file == "":
        return False

    # Check if file exists
    if not os.path.exists(pkl_file):
        return False

    # Check file size
    try:
        file_size = os.path.getsize(pkl_file)
        if file_size == 0:
            return False
        if file_size < 100:  # Very small files are likely corrupted
            return False
    except OSError:
        return False

    # Try to load the file
    try:
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)

        # Check if it's a valid graph format
        import networkx as nx
        if isinstance(data, nx.Graph):
            return True
        elif isinstance(data, dict):
            if 'subgraph_data' in data and isinstance(data['subgraph_data'], dict):
                if 'subgraph' in data['subgraph_data']:
                    return isinstance(data['subgraph_data']['subgraph'], nx.Graph)
            elif 'subgraph' in data:
                return isinstance(data['subgraph'], nx.Graph)

        return False
    except (EOFError, pickle.UnpicklingError, Exception):
        return False


def load_graph(pkl_file, max_nodes=None):
    """
    Load a spatial graph from pickle file for multi-view training.

    Simplified version: assumes graphs are cleaned and standardized.
    Graph paths should be direct paths to cleaned graph files.

    Args:
        pkl_file: Path to pickle file containing NetworkX graph
        max_nodes: Maximum number of nodes to keep. If graph has more nodes,
                   samples or truncates to max_nodes. None means no limit.

    Returns:
        PyG Data object or None if loading fails
    """
    import pickle
    from swift.utils import get_logger
    logger = get_logger()

    if pkl_file is None or pkl_file == "":
        return None

    # Check if file exists
    if not os.path.exists(pkl_file):
        logger.warning(f"Graph file not found: {pkl_file}")
        return None

    # Check file size first (empty or very small files are likely corrupted)
    try:
        file_size = os.path.getsize(pkl_file)
        if file_size == 0:
            logger.warning(f"Graph file is empty (0 bytes): {pkl_file}")
            return None
        if file_size < 100:  # Very small files are likely corrupted
            logger.warning(f"Graph file is suspiciously small ({file_size} bytes): {pkl_file}")
            return None
    except OSError as e:
        logger.warning(f"Failed to get file size for {pkl_file}: {e}")
        return None

    with open(pkl_file, 'rb') as f:
        try:
            data = pickle.load(f)
        except (EOFError, pickle.UnpicklingError) as e:
            logger.warning(f"Failed to load pickle file (corrupted or incomplete): {pkl_file}. Error: {e}")
            return None

    # Handle different pickle formats
    import networkx as nx
    nx_graph = None

    if isinstance(data, nx.Graph):
        # Format 1: Direct NetworkX graph
        nx_graph = data
    elif isinstance(data, dict):
        if 'subgraph_data' in data and isinstance(data['subgraph_data'], dict):
            # Format 2a: Nested dict with subgraph_data
            if 'subgraph' in data['subgraph_data']:
                nx_graph = data['subgraph_data']['subgraph']
        elif 'subgraph' in data:
            # Format 2b: Simple dict with subgraph
            nx_graph = data['subgraph']

    if nx_graph is None or not isinstance(nx_graph, nx.Graph):
        logger.warning(f"Invalid graph format in {pkl_file}")
        return None

    # Reduce graph size if max_nodes is specified
    original_num_nodes = nx_graph.number_of_nodes()
    if max_nodes is not None and original_num_nodes > max_nodes:
        # Strategy: Sample nodes using BFS from a central node (if available)
        # or random sampling, then create induced subgraph
        node_list = list(nx_graph.nodes())

        # Try to find central node (often stored as attribute)
        central_node = None
        if hasattr(nx_graph, 'graph') and isinstance(nx_graph.graph, dict):
            central_node = nx_graph.graph.get('central_node_id') or nx_graph.graph.get('central_node')

        if central_node is None:
            # Try to find node with highest degree (likely central)
            degrees = dict(nx_graph.degree())
            if degrees:
                central_node = max(degrees.items(), key=lambda x: x[1])[0]

        if central_node is not None and central_node in nx_graph:
            # BFS sampling: start from central node, expand to neighbors
            visited = set([central_node])
            queue = collections.deque([central_node])
            sampled_nodes = [central_node]

            while queue and len(sampled_nodes) < max_nodes:
                current = queue.popleft()
                neighbors = list(nx_graph.neighbors(current))
                # Shuffle neighbors for diversity
                random.shuffle(neighbors)

                for neighbor in neighbors:
                    if neighbor not in visited and len(sampled_nodes) < max_nodes:
                        visited.add(neighbor)
                        queue.append(neighbor)
                        sampled_nodes.append(neighbor)

            # If we still need more nodes, add random ones
            remaining = [n for n in node_list if n not in visited]
            if len(sampled_nodes) < max_nodes and remaining:
                random.shuffle(remaining)
                sampled_nodes.extend(remaining[:max_nodes - len(sampled_nodes)])
        else:
            # Random sampling if no central node found
            sampled_nodes = random.sample(node_list, max_nodes)

        # Create induced subgraph
        nx_graph = nx_graph.subgraph(sampled_nodes).copy()
        logger.debug(f"Reduced graph from {original_num_nodes} to {nx_graph.number_of_nodes()} nodes")

    # Use structural-only conversion (skip PyG attribute conversion to avoid schema issues)
    # This is simpler and more robust for graphs with heterogeneous node attributes
    try:
        from torch_geometric.data import Data
        node_list = list(nx_graph.nodes())
        id2idx = {nid: i for i, nid in enumerate(node_list)}
        edges = list(nx_graph.edges())
        if len(edges) > 0:
            src = [id2idx[u] for (u, v) in edges]
            dst = [id2idx[v] for (u, v) in edges]
            edge_index = torch.tensor([src, dst], dtype=torch.long)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        pyg_data = Data(edge_index=edge_index, num_nodes=len(node_list))
        # Keep reference to original NetworkX graph for attribute access
        pyg_data.nx_graph = nx_graph
        # Store original node ID mapping
        pyg_data.original_id_to_idx = id2idx
        return pyg_data
    except Exception as e:
        logger.warning(f"PyG structural conversion failed for {pkl_file}: {e}, using NetworkX wrapper")

        # Fallback: NetworkX wrapper (minimal interface)
        class GraphWrapper:
            def __init__(self, graph):
                self.nx_graph = graph
                self.num_nodes = graph.number_of_nodes()
                self.num_edges = graph.number_of_edges()
                self.nodes = list(graph.nodes())
                self.edges = list(graph.edges())

        return GraphWrapper(nx_graph)

    # except (EOFError, pickle.UnpicklingError) as e:
    #     logger.warning(f"Pickle file corrupted or incomplete: {pkl_file}. Error: {e}")
    #     return None
    # except Exception as e:
    #     logger.error(f"Error loading graph from {pkl_file}: {e}")
    #     import traceback
    #     logger.debug(traceback.format_exc())
    #     return None
