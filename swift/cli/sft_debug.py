# Copyright (c) Alibaba, Inc. and its affiliates.
"""
Debug-friendly SFT script for PyCharm debugging with graph-related arguments.

This script allows you to:
1. Set breakpoints and debug in PyCharm
2. Configure all graph encoder hyperparameters directly
3. Test spatial auxiliary loss (distance prediction)
4. Run training with full control over arguments

Usage in PyCharm:
    1. Right-click this file → "Run 'sft_debug'"
    2. Set breakpoints in swift/llm/train/sft.py or swift/trainers/trainers.py
    3. Debug!
"""

import sys
import os
import subprocess
import logging
import json

# ═══════════════════════════════════════════════════════════════
# CRITICAL: Set DeepSpeed environment variables BEFORE any imports
# This prevents DeepSpeed from trying to find nvcc during import
# ═══════════════════════════════════════════════════════════════

# Configure logging early so logger.info statements are visible immediately
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("swift.sft_debug")

# Try to find actual CUDA_HOME if not set
if 'CUDA_HOME' not in os.environ:
    try:
        # Try to find nvcc in common locations
        result = subprocess.run(['which', 'nvcc'], capture_output=True, text=True, timeout=2)
        if result.returncode == 0:
            nvcc_path = result.stdout.strip()
            cuda_home = os.path.dirname(os.path.dirname(nvcc_path))
            os.environ['CUDA_HOME'] = cuda_home
            print(f"✅ Found CUDA_HOME: {cuda_home}")
        else:
            # Fallback: try common CUDA installation paths
            for cuda_path in ['/usr/local/cuda', '/usr/local/cuda-12.0', '/usr/local/cuda-11.8', '/usr']:
                if os.path.exists(os.path.join(cuda_path, 'bin', 'nvcc')):
                    os.environ['CUDA_HOME'] = cuda_path
                    print(f"✅ Found CUDA_HOME at: {cuda_path}")
                    break
            else:
                # Last resort: set to /usr (DeepSpeed will skip check with DS_SKIP_CUDA_CHECK)
                os.environ['CUDA_HOME'] = '/usr'
                print(f"⚠️ CUDA_HOME not found, setting to /usr (will skip CUDA check)")
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        # If which command fails or times out, use fallback
        os.environ['CUDA_HOME'] = '/usr'
        print(f"⚠️ Could not detect CUDA_HOME, setting to /usr (will skip CUDA check)")

# Set DeepSpeed environment variables to skip CUDA checks
# os.environ['DS_BUILD_OPS'] = "0"  # Don't build DeepSpeed ops (skip JIT compilation)
os.environ['DS_SKIP_CUDA_CHECK'] = "1"  # Skip CUDA version check entirely
os.environ['USE_FAST_INFERENCE'] = "False"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# ═══════════════════════════════════════════════════════════════
# NCCL Debugging Configuration
# ═══════════════════════════════════════════════════════════════
# Enable NCCL debugging to diagnose distributed communication issues
# This helps identify if barriers hang due to NCCL communication problems
# os.environ['NCCL_DEBUG'] = 'INFO'  # or 'WARN' for less verbose
# os.environ['NCCL_DEBUG_SUBSYS'] = 'ALL'  # Debug all subsystems
os.environ['NCCL_TIMEOUT'] = '100'  # 30 minutes timeout (for slow checkpoint saves)

# ═══════════════════════════════════════════════════════════════
# Fix cuDNN and NCCL library paths for PyTorch 2.6.0+cu124
# Pip-installed libraries need to be in LD_LIBRARY_PATH
# ═══════════════════════════════════════════════════════════════
def find_nvidia_library_path(library_name):
    """Find NVIDIA library path from pip-installed packages (cuDNN, NCCL, etc.)"""
    import site
    import glob
    
    # Get site-packages directories
    site_packages = site.getsitepackages()
    try:
        user_site = site.getusersitepackages()
        if user_site:
            site_packages.append(user_site)
    except:
        pass
    
    # Also check conda environment
    conda_env = os.environ.get('CONDA_PREFIX', '')
    if conda_env:
        conda_lib = os.path.join(conda_env, 'lib')
        if os.path.exists(conda_lib):
            site_packages.append(conda_lib)
    
    # Search for the library in common locations
    search_paths = []
    for sp in site_packages:
        # Check nvidia package directories (nvidia/cudnn/lib, nvidia/nccl/lib, etc.)
        nvidia_lib_path = os.path.join(sp, 'nvidia', library_name.lower(), 'lib')
        if os.path.exists(nvidia_lib_path):
            search_paths.append(nvidia_lib_path)
        
        # Check parent lib directory
        parent_lib = os.path.join(os.path.dirname(sp), 'lib')
        if os.path.exists(parent_lib):
            search_paths.append(parent_lib)
    
    # Also check common system locations
    common_paths = [
        '/usr/local/cuda/lib64',
        '/usr/local/cuda/lib',
        '/usr/lib/x86_64-linux-gnu',
    ]
    search_paths.extend(common_paths)
    
    # Search for the library file
    for search_path in search_paths:
        if os.path.exists(search_path):
            # Look for library files
            lib_patterns = [
                os.path.join(search_path, f'lib{library_name.lower()}*.so*'),
                os.path.join(search_path, f'lib{library_name}*.so*'),
            ]
            for pattern in lib_patterns:
                matches = glob.glob(pattern)
                if matches:
                    return search_path
    
    return None

# Find and set cuDNN library path
cudnn_lib_path = find_nvidia_library_path('cudnn')
if cudnn_lib_path:
    current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
    if cudnn_lib_path not in current_ld_path:
        os.environ['LD_LIBRARY_PATH'] = f"{cudnn_lib_path}:{current_ld_path}" if current_ld_path else cudnn_lib_path
        print(f"✅ Added cuDNN library path to LD_LIBRARY_PATH: {cudnn_lib_path}")
    else:
        print(f"✅ cuDNN library path already in LD_LIBRARY_PATH: {cudnn_lib_path}")
else:
    print(f"⚠️ cuDNN library path not found automatically")

# Find and set NCCL library path
nccl_lib_path = find_nvidia_library_path('nccl')
if nccl_lib_path:
    current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
    if nccl_lib_path not in current_ld_path:
        os.environ['LD_LIBRARY_PATH'] = f"{nccl_lib_path}:{current_ld_path}" if current_ld_path else nccl_lib_path
        print(f"✅ Added NCCL library path to LD_LIBRARY_PATH: {nccl_lib_path}")
    else:
        print(f"✅ NCCL library path already in LD_LIBRARY_PATH: {nccl_lib_path}")
else:
    # Try conda lib as fallback
    conda_env = os.environ.get('CONDA_PREFIX', '')
    if conda_env:
        conda_lib = os.path.join(conda_env, 'lib')
        if os.path.exists(conda_lib):
            current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
            if conda_lib not in current_ld_path:
                os.environ['LD_LIBRARY_PATH'] = f"{conda_lib}:{current_ld_path}" if current_ld_path else conda_lib
                print(f"⚠️ NCCL not found in pip packages, added conda lib to LD_LIBRARY_PATH: {conda_lib}")
    else:
        print(f"⚠️ NCCL library path not found. You may need to install nvidia-nccl-cu12")

print(f"🔧 DeepSpeed configuration:")
print(f"   CUDA_HOME: {os.environ.get('CUDA_HOME', 'NOT SET')}")
print(f"   LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'NOT SET')}")
print(f"   DS_BUILD_OPS: {os.environ.get('DS_BUILD_OPS')}")
print(f"   DS_SKIP_CUDA_CHECK: {os.environ.get('DS_SKIP_CUDA_CHECK')}")

# ═══════════════════════════════════════════════════════════════
# MONKEY-PATCH: Skip DeepSpeed CUDA check if nvcc not found
# This prevents DeepSpeed from failing when nvcc is not available
# ═══════════════════════════════════════════════════════════════
# Patch subprocess.check_output to handle missing nvcc gracefully
_original_check_output = subprocess.check_output

def _patched_check_output(*args, **kwargs):
    """Patched check_output that handles missing nvcc for DeepSpeed"""
    try:
        # Check if this is the nvcc -V call that DeepSpeed makes
        if len(args) > 0 and isinstance(args[0], list):
            cmd = args[0]
            if len(cmd) > 0 and 'nvcc' in str(cmd[0]).lower():
                # This is likely DeepSpeed checking CUDA version
                # Return a dummy version string to skip the check
                return "nvcc: NVIDIA (R) Cuda compiler driver\nCopyright (c) 2005-2023 NVIDIA Corporation\nBuilt on Wed_Nov_22_10:17:15_PST_2023\nCuda compilation tools, release 11.8, V11.8.89\nBuild cuda_11.8.r11.8/compiler.33967100_0\n"
        # For all other calls, use the original function
        return _original_check_output(*args, **kwargs)
    except FileNotFoundError as e:
        # If nvcc is not found, return dummy version instead of raising
        if len(args) > 0 and isinstance(args[0], list):
            cmd = args[0]
            if len(cmd) > 0 and 'nvcc' in str(cmd[0]).lower():
                return "nvcc: NVIDIA (R) Cuda compiler driver\nCopyright (c) 2005-2023 NVIDIA Corporation\nBuilt on Wed_Nov_22_10:17:15_PST_2023\nCuda compilation tools, release 11.8, V11.8.89\nBuild cuda_11.8.r11.8/compiler.33967100_0\n"
        # Re-raise for other FileNotFoundErrors
        raise

# Apply the patch
subprocess.check_output = _patched_check_output
print("✅ Patched subprocess.check_output to handle missing nvcc")

# ═══════════════════════════════════════════════════════════════
# PATH SETUP - Ensure we can import swift modules
# ═══════════════════════════════════════════════════════════════
# Get project root from __file__
current_file = os.path.abspath(__file__)  # .../swift/cli/sft_debug.py
cli_dir = os.path.dirname(current_file)  # .../swift/cli/
swift_dir = os.path.dirname(cli_dir)  # .../swift/
project_root = os.path.dirname(swift_dir)  # .../ms-swift-main/

# Add project root to path so swift can be imported as a package
# This prevents duplicate module imports that cause template registration errors
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ═══════════════════════════════════════════════════════════════
# ENVIRONMENT SETUP
# ═══════════════════════════════════════════════════════════════
# Set CUDA environment variables for debugging
os.environ.setdefault('CUDA_LAUNCH_BLOCKING', '1')  # Synchronous CUDA for debugging
# Set MAX_PIXELS for Qwen2-VL models (adjust based on GPU memory)
os.environ.setdefault('MAX_PIXELS', '400000')  # Reduced from 400k to 300k to save memory (OOM during backward pass)
# Enable graph encoder debug logging
os.environ.setdefault('SWIFT_DEBUG_GRAPHS', '0')  # Enable debug logs in graph_encoder_spatial.py

# ═══════════════════════════════════════════════════════════════
# CRITICAL: Enable expandable segments to reduce memory fragmentation
# This helps prevent OOM during backward pass with gradient checkpointing
# ═══════════════════════════════════════════════════════════════
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

# ═══════════════════════════════════════════════════════════════
# Clear GPU memory before training (helps with non-deterministic OOM)
# ═══════════════════════════════════════════════════════════════
import torch
import gc

# Clear CUDA cache before starting
if torch.cuda.is_available():
    gc.collect()
    torch.cuda.empty_cache()
    # Clear cache on all GPUs
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    print("✅ Cleared GPU memory cache before training")

# Optional: Initialize unsloth if needed
if int(os.environ.get('UNSLOTH_PATCH_TRL', '0')) != 0:
    import unsloth

# ═══════════════════════════════════════════════════════════════
# IMPORTS
# ═══════════════════════════════════════════════════════════════
from swift.llm.argument.train_args import TrainArguments
from swift.llm import sft_main
from swift.utils.experiment_tracking import setup_experiment_tracking

# ═══════════════════════════════════════════════════════════════
# CRITICAL: Patch get_model_tokenizer to handle device_map correctly with DeepSpeed
# ═══════════════════════════════════════════════════════════════
# The issue is that get_model_tokenizer calls get_default_device_map() when device_map=None,
# which returns 'auto' with multiple GPUs. We need to patch it to check for DeepSpeed first.
from swift.llm.model.register import get_model_tokenizer as _original_get_model_tokenizer
from swift.llm.model.register import get_default_device_map as _original_get_default_device_map

# Global flag to track if DeepSpeed is enabled (set before model loading)
_deepspeed_enabled_global = False

def _patched_get_default_device_map():
    """Patched get_default_device_map that returns None when DeepSpeed is enabled"""
    global _deepspeed_enabled_global
    if _deepspeed_enabled_global:
        return None
    # Otherwise, use original behavior
    return _original_get_default_device_map()

def _patched_get_model_tokenizer(
    model_id_or_path: str,
    torch_dtype=None,
    device_map=None,
    **kwargs
):
    """Patched get_model_tokenizer that ensures device_map=None when DeepSpeed is enabled"""
    global _deepspeed_enabled_global
    # If DeepSpeed is enabled and device_map is None, keep it None (don't call get_default_device_map)
    # The original function has: if device_map is None: device_map = get_default_device_map()
    # We need to prevent this by passing a sentinel value or patching the internal logic
    # Actually, we can just ensure device_map stays None by not calling get_default_device_map
    if _deepspeed_enabled_global and device_map is None:
        # Keep device_map as None - the patched get_default_device_map will return None anyway
        pass
    # Call original function - it will call our patched get_default_device_map which returns None
    return _original_get_model_tokenizer(
        model_id_or_path=model_id_or_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        **kwargs
    )

# Apply patches
from swift.llm.model import register
register.get_default_device_map = _patched_get_default_device_map
register.get_model_tokenizer = _patched_get_model_tokenizer
logger.info("✅ Patched get_default_device_map and get_model_tokenizer for DeepSpeed compatibility")
print("✅ Patched get_default_device_map and get_model_tokenizer for DeepSpeed compatibility", flush=True)

# ═══════════════════════════════════════════════════════════════
# CRITICAL: Patch TrainArguments._init_deepspeed BEFORE creating instance
# ═══════════════════════════════════════════════════════════════
# The framework's _init_deepspeed() raises an error when is_mp() is True,
# but it should only raise if device_map is actually set (not None).
# We patch it on the class before creating any instances.
_original_init_deepspeed = TrainArguments._init_deepspeed

def _patched_init_deepspeed(self):
    """Patched _init_deepspeed that checks device_map before raising error"""
    global _deepspeed_enabled_global
    from swift.utils import is_mp, get_device_count
    if self.deepspeed:
        # Set global flag for get_default_device_map patch
        _deepspeed_enabled_global = True
        from transformers.utils.versions import require_version
        require_version('deepspeed')
        # Only raise error if device_map is set (not None)
        # If device_map is None, DeepSpeed should work fine even with is_mp()=True
        if is_mp() and not self.use_ray and self.device_map is not None:
            raise ValueError('DeepSpeed is not compatible with `device_map`. '
                             f'n_gpu: {get_device_count()}, '
                             f'local_world_size: {self.local_world_size}, '
                             f'device_map: {self.device_map}.')
        # Continue with original logic - call original method but skip the check
        # We temporarily set use_ray to True to bypass the check, then restore it
        original_use_ray = getattr(self, 'use_ray', False)
        try:
            # Temporarily set use_ray to bypass the check in original method
            self.use_ray = True
            # Call original method which will handle the rest
            _original_init_deepspeed(self)
        finally:
            # Restore original use_ray value
            self.use_ray = original_use_ray
    else:
        # Clear flag if DeepSpeed is not enabled
        _deepspeed_enabled_global = False

# Apply patch to the class
TrainArguments._init_deepspeed = _patched_init_deepspeed
logger.info("✅ Patched TrainArguments._init_deepspeed to allow DeepSpeed when device_map=None")
print("✅ Patched TrainArguments._init_deepspeed to allow DeepSpeed when device_map=None", flush=True)

# ═══════════════════════════════════════════════════════════════
# 🎯 EXPERIMENT NAME CONFIGURATION
# ═══════════════════════════════════════════════════════════════
# You can specify experiment name in three ways (priority order):
# 1. Command-line argument: python swift/cli/sft_debug.py --experiment_name "my_exp"
# 2. Environment variable: export EXPERIMENT_NAME="my_exp"
# 3. Set here: EXPERIMENT_NAME = "my_exp" (or None for auto-generation)
EXPERIMENT_NAME = None  # Set to None for auto-generation, or specify a name like "qwen2vl_graph_encoder_lr5e6"

# ═══════════════════════════════════════════════════════════════
# 🎯 TRAINING CONFIGURATION
# ═══════════════════════════════════════════════════════════════

# Dataset paths - MODIFY THESE FOR YOUR DATA
# Expand glob patterns to actual file paths (Swift doesn't support glob patterns directly)
import glob

# List of dataset paths (can include glob patterns)
_raw_dataset_paths = [
    # Match all files ending with "stage2.jsonl" in the directory
    "mydata/multiview_three_pairs/*stage2.jsonl",
    
    # Alternative specific paths (commented out):
    # "mydata/multiview_three_pairs/both_images_and_graphs_cleaned.jsonl",
    # "mydata/multiview_three_pairs/mapillary_results_newyork_captions_swift_format_cleaned_add_stage2.jsonl",
    # "mydata/multiview_three_pairs/mapillary_results_singapore_captions_swift_format_cleaned_add_stage2.jsonl",
    # "mydata/shuffled_multiview_three_pairs/graph_only_pairs.jsonl",
    # "mydata/multiview_three_pairs/reasoning_path_mapillary_swift_no_intersection_nodes0_reversed_singapore_multiview_three_pairs.jsonl",
]

# Expand glob patterns to actual file paths
TRAIN_DATASET_PATHS = []
for path in _raw_dataset_paths:
    if '*' in path or '?' in path:
        # Expand glob pattern
        expanded = glob.glob(path)
        if expanded:
            TRAIN_DATASET_PATHS.extend(expanded)
            print(f"✅ Expanded glob pattern '{path}' → {len(expanded)} files")
        else:
            print(f"⚠️ No files found matching pattern: {path}")
    else:
        # Use path as-is (must be absolute)
        if os.path.exists(path):
            TRAIN_DATASET_PATHS.append(path)
        else:
            print(f"⚠️ File does not exist: {path}")

if not TRAIN_DATASET_PATHS:
    raise ValueError(
        "No valid dataset files found! Please check your TRAIN_DATASET_PATHS. "
        f"Tried: {_raw_dataset_paths}"
    )

print(f"📁 Using {len(TRAIN_DATASET_PATHS)} dataset file(s):")
for p in TRAIN_DATASET_PATHS:
    print(f"   - {p}")

VAL_DATASET_PATHS = [
]

SPLIT_DATASET_RATIO = 0.1 # 10% for validation if no val_dataset

# ═══════════════════════════════════════════════════════════════
# CRITICAL: Set DeepSpeed flag early if we detect DeepSpeed will be used
# ═══════════════════════════════════════════════════════════════
# We need to set this flag BEFORE creating TrainArguments because
# the model might be loaded during initialization. We'll detect DeepSpeed
# from command line args or environment, or set it when we create TrainArguments.
# For now, we'll set it when we detect deepspeed='zero2' in the config below.
# This is a temporary flag that will be properly set in _init_deepspeed.

# ═══════════════════════════════════════════════════════════════
# 🚀 CREATE TRAINING ARGUMENTS WITH GRAPH ENCODER CONFIG
# ═══════════════════════════════════════════════════════════════
# BASE_MODEL =  'microsoft/Phi-3-vision-128k-instruct'
# BASE_MODEL="iic/gme-Qwen2-VL-2B-Instruct"
# BASE_MODEL = 'llava-hf/llava-v1.6-mistral-7b-hf'
# BASE_MODEL = 'Qwen/Qwen2-VL-2B-Instruct'
BASE_MODEL = 'Qwen/Qwen2.5-VL-7B-Instruct' #Qwen/Qwen2-VL-2B-Instruct
# BASE_MODEL = 'OpenGVLab/InternVL2_5-2B'  # InternVL3-1B model
# "output/stage1_vlm2vec2b/v1-20251217-200945/checkpoint-3120/"

training_arguments = TrainArguments(
    # ═══════════════════════════════════════════════════════════
    # MODEL CONFIGURATION
    # ═══════════════════════════════════════════════════════════
    model=BASE_MODEL,
    task_type='embedding',
    # template='internvl2_graph',  # Graph-enabled template for InternVL3-1B
    template='qwen2_5_vl_graph',
    # template='qwen2_vl_graph',
    # template='phi3_vision_graph',
    # template='llava1_6_mistral_graph_hf',
    # ═══════════════════════════════════════════════════════════
    # DATASET CONFIGURATION
    # ═══════════════════════════════════════════════════════════
    dataset=TRAIN_DATASET_PATHS,
    val_dataset=VAL_DATASET_PATHS if VAL_DATASET_PATHS else [],  # Use provided val dataset or split from train
    split_dataset_ratio=SPLIT_DATASET_RATIO if not VAL_DATASET_PATHS else 0.0,  # Split only if no val dataset provided
    dataset_shuffle=True,
    
    # ═══════════════════════════════════════════════════════════
    # GRAPH ENCODER CONFIGURATION
    # ═══════════════════════════════════════════════════════════
    # Enable graph encoder for multi-view training
    use_graph_encoder=True,  # ✅ Enable graph encoder
    
    # Graph architecture
    graph_num_layers=2,  # Number of GNN layers
    edge_dim=64,  # Edge embedding dimension
    graph_max_nodes=1000,  # ✅ Maximum nodes per graph (reduces memory and sequence length)
    
    # Spatial encoding (PE-GNN style)
    use_spatial_encoding=True,  # ✅ Enable spatial positional encoding
    spatial_embed_dim=64,  # Spatial embedding dimension
    spatial_frequency_num=16,  # Number of frequency bands
    
    # Edge features (GeoGNN style)
    use_edge_features=True,  # ✅ Enable geodesic edge features (distance, bearing, displacement)
    edge_use_distance=True,  # Enable distance feature (haversine distance)
    edge_use_direction=True,  # Enable direction feature (bearing angle)
    edge_use_displacement=True,  # Enable displacement feature (Δlat, Δlon) - set to False for ablation
    
    # GNN type
    use_gat=True,  # ✅ Use GATv2 (Graph Attention) instead of GCN
    gat_heads=4,  # Number of attention heads
    
    # ═══════════════════════════════════════════════════════════
    # TRAINING HYPERPARAMETERS
    # ═══════════════════════════════════════════════════════════
    num_train_epochs=5,

    per_device_train_batch_size=3,  # 4Minimum 2 for InfoNCE (negatives = 2 * 4 GPUs = 8)
    per_device_eval_batch_size=3,  # 4Minimum 2 for InfoNCE
    gradient_accumulation_steps=2,  # Effective batch size: 2 * 8 * 4 = 64 (same as before)

    save_strategy='steps',
    save_steps=800,  # Save checkpoint every N steps
    eval_strategy='steps',  # ✅ ENABLED - evaluate during training to select best model
    eval_steps=800,  # Evaluate every N steps (same as save_steps)
    save_total_limit=2,  # Keep only the final checkpoint
    logging_steps=300, # Match Stage 1
    disable_tqdm=False,
    remove_unused_columns=True,  # Match Stage 1
    load_from_cache_file=True,  # Match Stage 1
    
    # ═══════════════════════════════════════════════════════════
    # OPTIMIZATION
    # ═══════════════════════════════════════════════════════════
    optimizer='multiview_graph',
    # optimizer='ogd_multiview_graph',  # ✅ Use OGD-enabled multiview optimizer to preserve image-text alignment
                                 # This prevents scheduler loading errors when adding graph encoder
                                 # Note: Stage 1 used default optimizer (null), but we use multiview_graph
                                 # to ensure scheduler compatibility when adding graph encoder
    learning_rate=5e-5,  # Learning rate for training (same as Stage 1)
    warmup_ratio=0.05,  # Match Stage 1
    weight_decay=0.1,  # Match Stage 1
    max_grad_norm=1.0,  # Match Stage 1
    # Fine-tune learning rates for different components
    image_text_lr_scale=0.1,  # Scale down LR for image/text encoders to preserve alignment (10% of base LR)
                              # With learning_rate=1e-6, image/text will use 1e-7 (1e-6 * 0.1)
                              # This is very conservative and helps preserve the pre-trained alignment
    graph_lr=5e-5,  # Learning rate for graph encoder (can be higher since it's new, but using same for consistency)
    
    # ═══════════════════════════════════════════════════════════
    # LOSS CONFIGURATION
    # ═══════════════════════════════════════════════════════════
    loss_type='infonce',  # InfoNCE loss for embedding training
    
    # ═══════════════════════════════════════════════════════════
    # OUTPUT CONFIGURATION
    # ═══════════════════════════════════════════════════════════
    # Base output directory - experiment tracking will create a subdirectory
    # with timestamp and experiment name inside this directory.
    # Final structure: output_dir/YYYYMMDD-HHMMSS_experiment_name/
    # output_dir='output/stage2_qwen25vl7b_edge_embed_32',
    output_dir='output/stage2_qwen25vl7b_test',
    
    # ═══════════════════════════════════════════════════════════
    # IMAGE/VISION PARAMETERS
    # ═══════════════════════════════════════════════════════════
    # max_pixels=None,  # Let Qwen2-VL processor handle smart resize
    # max_image_tokens=4096,
    max_length=32768,  # Match Stage 1
    truncation_strategy='delete',  # Match Stage 1

    # ═══════════════════════════════════════════════════════════
    # DISTRIBUTED TRAINING
    # ═══════════════════════════════════════════════════════════
    # Match working version (14:40) that had no OOM:
    # - Use ZeRO-2 (same as working version)
    # - Training mode uses MORE memory than evaluation mode
    # - If OOM persists, reduce batch size or use zero3_offload
    deepspeed='zero2',  # Match working version - ZeRO-2 (stage 2, no offload)
    # device_map=None,  # Commented out like original - let framework handle it
    dataloader_num_workers=0,  # Match Stage 1
    dataloader_drop_last=True,  # Match Stage 1
    
    # ═══════════════════════════════════════════════════════════
    # MEMORY AND EFFICIENCY
    # ═══════════════════════════════════════════════════════════
    gradient_checkpointing=True,  # Match working version (14:40)
    vit_gradient_checkpointing=True,  # Match working version (14:40) - critical for vision models
    dataloader_pin_memory=True,  # Match working version (14:40)
    # Limit graph node processing to reduce memory
    # (This will be handled in graph encoder if needed)
    
    # ═══════════════════════════════════════════════════════════
    # PRECISION
    # ═══════════════════════════════════════════════════════════
    fp16=False,  # Disable for debugging (more stable)
    bf16=True,
    
    # ═══════════════════════════════════════════════════════════
    # LoRA CONFIGURATION
    # ═══════════════════════════════════════════════════════════
    train_type='lora',  # Match Stage 1
    lora_rank=16,  # Match Stage 1
    lora_alpha=32,  # Match Stage 1
    lora_dropout=0.1,  # Match Stage 1 (was 0.05, not 0.1)
    target_modules=["q_proj", "v_proj"],
    
    # ═══════════════════════════════════════════════════════════
    # ATTENTION
    # ═══════════════════════════════════════════════════════════
    #  attn_impl="flash_attn",
    attn_impl="sdpa",
    # attn_impl="eager", #internvl1b
    
    # ═══════════════════════════════════════════════════════════
    # CHECKPOINTING
    # ═══════════════════════════════════════════════════════════
    load_best_model_at_end=True,  # ✅ ENABLED - load best model based on evaluation metric
    metric_for_best_model='margin',  # Use margin metric for embedding training
    greater_is_better=True,  # Higher margin is better
    # ═══════════════════════════════════════════════════════════
    # CHECKPOINT RESUME CONFIGURATION
    # ═══════════════════════════════════════════════════════════
    # resume_from_checkpoint="output/stage1_ph3/v1-20251214-121357/checkpoint-1605/",
    # resume_from_checkpoint="output/stage1_llava/checkpoint-4990/",
    # resume_from_checkpoint="output/stage1_qwen253b/v3-20260103-104045/checkpoint-935/",
    resume_from_checkpoint="output/stage1_qwen25vl7b/v0-20251212-113829/checkpoint-2085/",
    # resume_from_checkpoint="output/stage1_vlm2vec2b/checkpoint-2495/",
    # resume_from_checkpoint="output/stage1_qwen27b_more/v1-20251215-115023/checkpoint-2495/",
    # resume_from_checkpoint="output/stage1_qwen25vl7b/v0-20251212-113829/checkpoint-2085/",
    # resume_from_checkpoint="output/stage1_internvl31b/v5-20251214-120648/checkpoint-1895/",
    # resume_from_checkpoint="output/stage1_internvl252b/v10-20260124-123834/checkpoint-4160/",
    resume_only_model=True,  # ✅ Only load model weights, NOT training state (global_step, epoch, etc.)
                             # This is correct for continual learning with a NEW task (different dataset + new modality)
                             # The model weights will be loaded, but training will start from step 0
                             # Set to False if you want to continue from exact training state
    ignore_data_skip=True,  # ✅ Don't restore training state (global_step, epoch) - start fresh from step 0
                            # This ensures the scheduler and training progress start from 0 for the new task
                            # while still loading the pre-trained model weights
)

# ═══════════════════════════════════════════════════════════════
# CRITICAL: Set DeepSpeed flag IMMEDIATELY after TrainArguments creation
# ═══════════════════════════════════════════════════════════════
# This must be done BEFORE any model loading happens. The flag is used by
# get_default_device_map() to return None when DeepSpeed is enabled.
if training_arguments.deepspeed:
    _deepspeed_enabled_global = True
    logger.info("✅ Set DeepSpeed flag early to prevent device_map='auto'")
    print("✅ Set DeepSpeed flag early to prevent device_map='auto'", flush=True)

# ═══════════════════════════════════════════════════════════════
# NOTE: split_dataset_ratio handling
# ═══════════════════════════════════════════════════════════════
# With eval_strategy='steps', the validation dataset will be automatically split
# and used for evaluation during training. The split_dataset_ratio is already set
# in TrainArguments, so no override is needed.
# The validation dataset will be saved to val_dataset.jsonl for reference.

# ═══════════════════════════════════════════════════════════════
# CRITICAL: Ensure device_map is None when DeepSpeed is enabled
# ═══════════════════════════════════════════════════════════════
# DeepSpeed is not compatible with device_map, so we must ensure it's None
# Also set the global flag for get_default_device_map patch BEFORE model loading
# NOTE: This is needed for Qwen2.5-VL models but may not be needed for InternVL models
if training_arguments.deepspeed:
    # Set global flag early so get_default_device_map returns None
    _deepspeed_enabled_global = True
    if training_arguments.device_map is not None:
        logger.warning(f"⚠️ device_map={training_arguments.device_map} is set but DeepSpeed is enabled. Setting to None.")
        print(f"⚠️ device_map={training_arguments.device_map} is set but DeepSpeed is enabled. Setting to None.", flush=True)
    training_arguments.device_map = None
    if hasattr(training_arguments, 'training_args'):
        # Also ensure it's None in nested training_args if it exists
        if hasattr(training_arguments.training_args, 'device_map'):
            training_arguments.training_args.device_map = None
    logger.info("✅ Ensured device_map=None for DeepSpeed compatibility")
    print("✅ Ensured device_map=None for DeepSpeed compatibility", flush=True)
else:
    # Clear flag if DeepSpeed is not enabled
    _deepspeed_enabled_global = False

# ═══════════════════════════════════════════════════════════
# OPTIMIZER STATE SAVING CONFIGURATION
# ═══════════════════════════════════════════════════════════
# For final stage training, you may not need to save optimizer state
# This can significantly speed up checkpoint saving and avoid hangs
# Set to True to skip optimizer/scheduler state saving (only save model weights)
# NOTE: This will be set on training_args after it's created in __post_init__
SKIP_OPTIMIZER_STATE_SAVE = True #kip optimizer state saving for final stage


def log_ogd_settings(args: TrainArguments):
    """Log OGD-related settings so we can confirm behaviour at runtime."""
    if not getattr(args, "use_ogd", False):
        logger.info(
            "OGD is disabled. Training will proceed without orthogonal gradient constraints."
        )
        return

    logger.info("OGD is enabled for continual contrastive learning.")
    logger.info(
        "OGD memory size: %s | update batches: %s | OGD+: %s | use_gradients: %s",
        getattr(args, "ogd_memory_size", "N/A"),
        getattr(args, "ogd_update_memory_batches", "N/A"),
        getattr(args, "ogd_use_ogd_plus", False),
        getattr(args, "ogd_use_gradients", False),
    )
    logger.info(
        "Protected layers: %s",
        getattr(args, "ogd_protected_layers", "auto-detect"),
    )
    logger.info(
        "Resume-only-model: %s | Ignore data skip: %s",
        getattr(args, "resume_only_model", False),
        getattr(args, "ignore_data_skip", False),
    )


# ═══════════════════════════════════════════════════════════════
# 🎯 PRINT CONFIGURATION SUMMARY
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    # ═══════════════════════════════════════════════════════════
    # 🎯 PARSE COMMAND-LINE ARGUMENTS FOR EXPERIMENT NAME
    # ═══════════════════════════════════════════════════════════
    import argparse
    parser = argparse.ArgumentParser(add_help=False)  # Don't show help, we'll handle it
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Name for the experiment (overrides auto-generation)')
    known_args, _ = parser.parse_known_args()
    
    # Determine experiment name (priority: CLI arg > env var > script variable > auto-generate)
    experiment_name = None
    if known_args.experiment_name:
        experiment_name = known_args.experiment_name
        print(f"✅ Using experiment name from command-line: {experiment_name}")
    elif os.environ.get('EXPERIMENT_NAME'):
        experiment_name = os.environ.get('EXPERIMENT_NAME')
        print(f"✅ Using experiment name from environment: {experiment_name}")
    elif EXPERIMENT_NAME is not None:
        experiment_name = EXPERIMENT_NAME
        print(f"✅ Using experiment name from script: {experiment_name}")
    else:
        # Auto-generate experiment name from key parameters
        exp_name_parts = []
        if training_arguments.use_graph_encoder:
            exp_name_parts.append("graph")
            if training_arguments.use_spatial_encoding:
                exp_name_parts.append("spatial")
            if training_arguments.use_edge_features:
                exp_name_parts.append("edgefeat")
            if training_arguments.use_gat:
                exp_name_parts.append(f"gat{training_arguments.gat_heads}")
        if training_arguments.use_ogd:
            exp_name_parts.append("ogd")
        if training_arguments.use_spatial_auxiliary:
            exp_name_parts.append("s2aux")
        
        # Add model info
        model_name = training_arguments.model.split('/')[-1].replace('-', '_')
        exp_name_parts.insert(0, model_name)
        
        # Add loss type
        exp_name_parts.append(training_arguments.loss_type)
        
        experiment_name = "_".join(exp_name_parts) if exp_name_parts else "default"
        print(f"✅ Auto-generated experiment name: {experiment_name}")
    
    log_ogd_settings(training_arguments)
    
    # ═══════════════════════════════════════════════════════════
    # 🎯 EXPERIMENT TRACKING SETUP
    # ═══════════════════════════════════════════════════════════
    
    # Get seed from training arguments if available
    seed = getattr(training_arguments, 'seed', None)
    if seed is None:
        # Try to get from training_args
        if hasattr(training_arguments, 'training_args'):
            seed = getattr(training_arguments.training_args, 'seed', None)
    
    # Get full command
    full_command = " ".join(sys.argv)
    
    # Convert training arguments to dict for config saving
    # Get all attributes that are not callable or private
    config_dict = {}
    for key in dir(training_arguments):
        if not key.startswith('_') and not callable(getattr(training_arguments, key, None)):
            try:
                value = getattr(training_arguments, key)
                # Skip very large objects or functions
                if not callable(value):
                    config_dict[key] = value
            except:
                pass
    
    # Also include nested training_args if available
    if hasattr(training_arguments, 'training_args'):
        for key in dir(training_arguments.training_args):
            if not key.startswith('_') and not callable(getattr(training_arguments.training_args, key, None)):
                try:
                    value = getattr(training_arguments.training_args, key)
                    if not callable(value):
                        config_dict[f'training_args.{key}'] = value
                except:
                    pass
    
    # Add dataset paths to config
    config_dict['train_dataset_paths'] = TRAIN_DATASET_PATHS
    config_dict['val_dataset_paths'] = VAL_DATASET_PATHS
    config_dict['split_dataset_ratio'] = SPLIT_DATASET_RATIO
    
    # Set up experiment tracking
    # Note: We'll update output_dir after tracking is set up
    # The base_output_dir (from line 600) is used as the parent directory.
    # Experiment tracking creates: base_output_dir/YYYYMMDD-HHMMSS_experiment_name/
    base_output_dir = training_arguments.output_dir
    exp_info = setup_experiment_tracking(
        output_dir=base_output_dir,  # Base directory from TrainArguments (line 600)
        config_dict=config_dict,
        command=full_command,
        experiment_name=experiment_name,
        seed=seed,
        dataset_paths=TRAIN_DATASET_PATHS,
        save_code_copy=True,
        main_script_path=__file__,
        repo_path=project_root
    )
    
    # Update training arguments output_dir to use the final tracked directory
    # This replaces the base directory with the timestamped subdirectory
    # Example: 'output/graph_encoder_no_edge_feature' 
    #       -> 'output/graph_encoder_no_edge_feature/20251201-221530_experiment_name'
    training_arguments.output_dir = exp_info['output_dir']
    if hasattr(training_arguments, 'training_args'):
        training_arguments.training_args.output_dir = exp_info['output_dir']
    
    print("\n" + "=" * 70)
    print("🚀 DEBUG MODE: Starting Training with Graph Encoder")
    print(f"📁 Experiment Output Directory: {exp_info['output_dir']}")
    print(f"📝 Git Commit: {exp_info['git_info']['commit_hash'][:8]} ({exp_info['git_info']['branch']})")
    print("=" * 70)
    print("\n📋 Configuration Summary:")
    print(f"  Model: {training_arguments.model}")
    print(f"  Template: {training_arguments.template}")
    print(f"  Task Type: {training_arguments.task_type}")
    print(f"  Loss Type: {training_arguments.loss_type}")
    print(f"\n🔧 Graph Encoder:")
    print(f"  use_graph_encoder: {training_arguments.use_graph_encoder}")
    print(f"  graph_num_layers: {training_arguments.graph_num_layers}")
    print(f"  edge_dim: {training_arguments.edge_dim}")
    print(f"  use_spatial_encoding: {training_arguments.use_spatial_encoding}")
    print(f"  spatial_embed_dim: {training_arguments.spatial_embed_dim}")
    print(f"  spatial_frequency_num: {training_arguments.spatial_frequency_num}")
    print(f"  use_edge_features: {training_arguments.use_edge_features}")
    print(f"\n💾 Checkpoint Saving:")
    print(f"  skip_optimizer_state_save: {getattr(training_arguments, 'skip_optimizer_state_save', False)}")
    if training_arguments.use_edge_features:
        edge_use_distance = getattr(training_arguments, 'edge_use_distance', True)
        edge_use_direction = getattr(training_arguments, 'edge_use_direction', True)
        edge_use_displacement = getattr(training_arguments, 'edge_use_displacement', True)
        enabled_components = []
        if edge_use_distance:
            enabled_components.append("distance")
        if edge_use_direction:
            enabled_components.append("direction/bearing")
        if edge_use_displacement:
            enabled_components.append("displacement")
        print(f"    Components: {', '.join(enabled_components) if enabled_components else 'none'}")
    print(f"  use_gat: {training_arguments.use_gat}")
    print(f"  gat_heads: {training_arguments.gat_heads}")
    print(f"\n📊 Training:")
    print(f"  Effective Batch Size: {training_arguments.per_device_train_batch_size * training_arguments.gradient_accumulation_steps}")
    print(f"  Learning Rate: {training_arguments.learning_rate}")
    print(f"  Epochs: {training_arguments.num_train_epochs}")
    print(f"  Output Dir: {training_arguments.output_dir}")
    print(f"\n📁 Datasets:")
    print(f"  Train: {len(TRAIN_DATASET_PATHS)} file(s)")
    for i, path in enumerate(TRAIN_DATASET_PATHS, 1):
        print(f"    {i}. {path}")
    if VAL_DATASET_PATHS:
        print(f"  Val: {len(VAL_DATASET_PATHS)} file(s)")
    else:
        print(f"  Val: Split from train ({SPLIT_DATASET_RATIO*100:.0f}%)")
    print("=" * 70)
    print("\n💡 Debugging Tips:")
    print("  1. Set breakpoints in:")
    print("     - swift/trainers/trainers.py (EmbeddingTrainer.compute_loss)")
    print("     - swift/llm/model/model/graph_encoder_spatial.py")
    print("     - swift/llm/model/model/spatial_encoders.py")
    print("  2. Monitor logs for:")
    print("     - train/main_loss (InfoNCE)")
    print("     - train/spatial_loss (Distance prediction)")
    print("     - train/distance_mae (Mean absolute error in meters)")
    print("     - train/distance_r2 (R² correlation)")
    print("=" * 70)
    print("\n🚀 Starting training...\n")
    
    # Start training
    # Set skip_optimizer_state_save on training_args after it's created
    # This happens in TrainArguments.__post_init__, so we set it here before training starts
    if SKIP_OPTIMIZER_STATE_SAVE:
        if hasattr(training_arguments, 'training_args'):
            training_arguments.training_args.skip_optimizer_state_save = True
            print(f"✅ Set skip_optimizer_state_save=True on training_args.training_args (will skip optimizer state saving)", flush=True)
            logger.info(f"✅ Set skip_optimizer_state_save=True on training_args.training_args (will skip optimizer state saving)")
        # Also set it on the TrainArguments object itself in case it's checked there
        if hasattr(training_arguments, 'skip_optimizer_state_save'):
            training_arguments.skip_optimizer_state_save = True
            print(f"✅ Set skip_optimizer_state_save=True on training_arguments (will skip optimizer state saving)", flush=True)
            logger.info(f"✅ Set skip_optimizer_state_save=True on training_arguments (will skip optimizer state saving)")
    
    sft_main(training_arguments)

