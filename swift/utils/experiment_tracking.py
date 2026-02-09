# Copyright (c) Alibaba, Inc. and its affiliates.
"""
Experiment Tracking Utility for Reproducible Research

This module provides utilities for tracking experiments with:
1. Git commit hash recording
2. Configuration saving (YAML/JSON)
3. Environment information logging
4. Standardized directory structure
5. Command recording

Usage:
    from swift.utils.experiment_tracking import setup_experiment_tracking
    
    exp_info = setup_experiment_tracking(
        output_dir="output/my_experiment",
        config_dict=training_args.__dict__,
        command="python train.py --lr 0.01",
        experiment_name="resnet50_add_se_block_lr0.01"
    )
"""

import os
import sys
import json
import subprocess
import platform
import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union
import torch
import logging

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

from .logger import get_logger

logger = get_logger()


def get_git_commit_hash(repo_path: Optional[str] = None) -> Dict[str, str]:
    """
    Get Git commit hash and branch information.
    
    Args:
        repo_path: Path to git repository. If None, uses current working directory.
    
    Returns:
        Dictionary with 'commit_hash', 'branch', 'is_dirty', 'remote_url'
    """
    if repo_path is None:
        repo_path = os.getcwd()
    
    result = {
        'commit_hash': 'unknown',
        'branch': 'unknown',
        'is_dirty': False,
        'remote_url': 'unknown',
        'commit_message': 'unknown'
    }
    
    try:
        # Check if we're in a git repository
        git_dir = os.path.join(repo_path, '.git')
        if not os.path.exists(git_dir):
            # Try to find git root
            try:
                proc = subprocess.run(
                    ['git', 'rev-parse', '--show-toplevel'],
                    cwd=repo_path,
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if proc.returncode == 0:
                    repo_path = proc.stdout.strip()
            except (subprocess.TimeoutExpired, FileNotFoundError):
                logger.warning("Git not found or not in a git repository")
                return result
        
        # Get commit hash
        try:
            proc = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=5
            )
            if proc.returncode == 0:
                result['commit_hash'] = proc.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        # Get branch name
        try:
            proc = subprocess.run(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=5
            )
            if proc.returncode == 0:
                result['branch'] = proc.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        # Check if working directory is dirty
        try:
            proc = subprocess.run(
                ['git', 'diff', '--quiet'],
                cwd=repo_path,
                capture_output=True,
                timeout=5
            )
            result['is_dirty'] = (proc.returncode != 0)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        # Get remote URL
        try:
            proc = subprocess.run(
                ['git', 'config', '--get', 'remote.origin.url'],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=5
            )
            if proc.returncode == 0:
                result['remote_url'] = proc.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        # Get commit message
        try:
            proc = subprocess.run(
                ['git', 'log', '-1', '--pretty=%B'],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=5
            )
            if proc.returncode == 0:
                result['commit_message'] = proc.stdout.strip().split('\n')[0]  # First line only
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
            
    except Exception as e:
        logger.warning(f"Error getting git information: {e}")
    
    return result


def get_environment_info() -> Dict[str, Any]:
    """
    Get environment information including PyTorch, CUDA, GPU, etc.
    
    Returns:
        Dictionary with environment information
    """
    env_info = {
        'python_version': sys.version,
        'platform': platform.platform(),
        'pytorch_version': torch.__version__ if hasattr(torch, '__version__') else 'unknown',
        'cuda_available': torch.cuda.is_available() if hasattr(torch, 'cuda') else False,
        'cuda_version': 'unknown',
        'gpu_count': 0,
        'gpu_models': [],
    }
    
    # Get CUDA version
    if env_info['cuda_available']:
        try:
            env_info['cuda_version'] = torch.version.cuda
        except:
            pass
        
        # Get GPU count and models
        try:
            env_info['gpu_count'] = torch.cuda.device_count()
            env_info['gpu_models'] = [
                torch.cuda.get_device_name(i) for i in range(env_info['gpu_count'])
            ]
        except:
            pass
    
    # Get CPU info
    try:
        env_info['cpu_count'] = os.cpu_count()
    except:
        pass
    
    return env_info


def save_config_to_yaml(config_dict: Dict[str, Any], output_path: str) -> None:
    """
    Save configuration dictionary to YAML file.
    
    Args:
        config_dict: Configuration dictionary
        output_path: Path to save YAML file
    """
    if not YAML_AVAILABLE:
        logger.warning("PyYAML not available, skipping YAML config save. Install with: pip install PyYAML")
        return
    
    # Convert non-serializable objects to strings
    def convert_to_serializable(obj):
        if isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        else:
            return str(obj)
    
    serializable_config = convert_to_serializable(config_dict)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(serializable_config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def save_config_to_json(config_dict: Dict[str, Any], output_path: str) -> None:
    """
    Save configuration dictionary to JSON file.
    
    Args:
        config_dict: Configuration dictionary
        output_path: Path to save JSON file
    """
    # Convert non-serializable objects to strings
    def convert_to_serializable(obj):
        if isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        else:
            return str(obj)
    
    serializable_config = convert_to_serializable(config_dict)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_config, f, indent=2, ensure_ascii=False)


def create_experiment_log_header(
    command: str,
    git_info: Dict[str, str],
    env_info: Dict[str, Any],
    config_dict: Dict[str, Any],
    seed: Optional[int] = None,
    dataset_paths: Optional[Union[str, list]] = None
) -> str:
    """
    Create a formatted log header with all experiment information.
    
    Args:
        command: Full command used to run the experiment
        git_info: Git information dictionary
        env_info: Environment information dictionary
        config_dict: Configuration dictionary
        seed: Random seed used
        dataset_paths: Dataset paths used for training
    
    Returns:
        Formatted log header string
    """
    lines = []
    lines.append("=" * 80)
    lines.append("EXPERIMENT TRACKING INFORMATION")
    lines.append("=" * 80)
    lines.append("")
    
    # Command
    lines.append("1. COMMAND:")
    lines.append(f"   {command}")
    lines.append("")
    
    # Environment
    lines.append("2. ENVIRONMENT:")
    lines.append(f"   Python Version: {env_info.get('python_version', 'unknown').split()[0]}")
    lines.append(f"   Platform: {env_info.get('platform', 'unknown')}")
    lines.append(f"   PyTorch Version: {env_info.get('pytorch_version', 'unknown')}")
    lines.append(f"   CUDA Available: {env_info.get('cuda_available', False)}")
    if env_info.get('cuda_available'):
        lines.append(f"   CUDA Version: {env_info.get('cuda_version', 'unknown')}")
        lines.append(f"   GPU Count: {env_info.get('gpu_count', 0)}")
        for i, gpu_model in enumerate(env_info.get('gpu_models', [])):
            lines.append(f"   GPU {i}: {gpu_model}")
    lines.append(f"   CPU Count: {env_info.get('cpu_count', 'unknown')}")
    lines.append("")
    
    # Seed
    lines.append("3. SEED:")
    lines.append(f"   Random Seed: {seed if seed is not None else 'Not set'}")
    lines.append("")
    
    # Git Hash
    lines.append("4. GIT HASH (Code Version):")
    lines.append(f"   Commit Hash: {git_info.get('commit_hash', 'unknown')}")
    lines.append(f"   Branch: {git_info.get('branch', 'unknown')}")
    lines.append(f"   Is Dirty: {git_info.get('is_dirty', False)}")
    if git_info.get('remote_url') != 'unknown':
        lines.append(f"   Remote URL: {git_info.get('remote_url', 'unknown')}")
    if git_info.get('commit_message') != 'unknown':
        lines.append(f"   Commit Message: {git_info.get('commit_message', 'unknown')}")
    lines.append("")
    
    # Dataset Paths
    if dataset_paths:
        lines.append("5. DATASET PATHS:")
        if isinstance(dataset_paths, str):
            dataset_paths = [dataset_paths]
        for i, path in enumerate(dataset_paths, 1):
            lines.append(f"   Dataset {i}: {path}")
        lines.append("")
    
    # Config
    lines.append("6. CONFIG (All Hyperparameters):")
    for key, value in sorted(config_dict.items()):
        # Truncate very long values
        value_str = str(value)
        if len(value_str) > 200:
            value_str = value_str[:200] + "... (truncated)"
        lines.append(f"   {key}: {value_str}")
    lines.append("")
    
    lines.append("=" * 80)
    lines.append("")
    
    return "\n".join(lines)


def setup_experiment_tracking(
    output_dir: str,
    config_dict: Dict[str, Any],
    command: Optional[str] = None,
    experiment_name: Optional[str] = None,
    seed: Optional[int] = None,
    dataset_paths: Optional[Union[str, list]] = None,
    save_code_copy: bool = True,
    main_script_path: Optional[str] = None,
    repo_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Set up experiment tracking with all required information.
    
    This function:
    1. Creates standardized output directory structure
    2. Records Git commit hash
    3. Saves configuration to YAML
    4. Creates experiment log header
    5. Optionally saves code copy
    
    Args:
        output_dir: Base output directory
        config_dict: Configuration dictionary (all hyperparameters)
        command: Full command used to run the experiment
        experiment_name: Name for the experiment (used in directory naming)
        seed: Random seed used
        dataset_paths: Dataset paths used for training
        save_code_copy: Whether to save a copy of the main script
        main_script_path: Path to main training script (for code copy)
        repo_path: Path to git repository (defaults to current directory)
    
    Returns:
        Dictionary with experiment information including:
        - output_dir: Final output directory path
        - git_info: Git information
        - env_info: Environment information
        - log_header: Formatted log header string
        - config_path: Path to saved config YAML
    """
    # Get git information
    git_info = get_git_commit_hash(repo_path)
    
    # Get environment information
    env_info = get_environment_info()
    
    # Create standardized output directory
    timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    
    if experiment_name:
        # Format: timestamp_experiment_name
        exp_dir_name = f"{timestamp}_{experiment_name}"
    else:
        # Format: timestamp_exp
        exp_dir_name = f"{timestamp}_exp"
    
    final_output_dir = os.path.join(output_dir, exp_dir_name)
    os.makedirs(final_output_dir, exist_ok=True)
    
    # Save configuration to YAML
    config_yaml_path = os.path.join(final_output_dir, 'config.yaml')
    save_config_to_yaml(config_dict, config_yaml_path)
    logger.info(f"✅ Saved configuration to: {config_yaml_path}")
    
    # Also save as JSON for compatibility
    config_json_path = os.path.join(final_output_dir, 'config.json')
    save_config_to_json(config_dict, config_json_path)
    
    # Save git information
    git_info_path = os.path.join(final_output_dir, 'git_info.json')
    with open(git_info_path, 'w', encoding='utf-8') as f:
        json.dump(git_info, f, indent=2)
    
    # Save environment information
    env_info_path = os.path.join(final_output_dir, 'environment.json')
    with open(env_info_path, 'w', encoding='utf-8') as f:
        json.dump(env_info, f, indent=2)
    
    # Create log header
    if command is None:
        command = " ".join(sys.argv)
    
    log_header = create_experiment_log_header(
        command=command,
        git_info=git_info,
        env_info=env_info,
        config_dict=config_dict,
        seed=seed,
        dataset_paths=dataset_paths
    )
    
    # Save log header to file
    log_header_path = os.path.join(final_output_dir, 'experiment_info.txt')
    with open(log_header_path, 'w', encoding='utf-8') as f:
        f.write(log_header)
    
    # Print log header to console
    print("\n" + log_header)
    logger.info(f"✅ Saved experiment info to: {log_header_path}")
    
    # Save code copy if requested
    if save_code_copy and main_script_path and os.path.exists(main_script_path):
        code_copy_path = os.path.join(final_output_dir, os.path.basename(main_script_path))
        try:
            import shutil
            shutil.copy2(main_script_path, code_copy_path)
            logger.info(f"✅ Saved code copy to: {code_copy_path}")
        except Exception as e:
            logger.warning(f"⚠️ Could not save code copy: {e}")
    
    return {
        'output_dir': final_output_dir,
        'git_info': git_info,
        'env_info': env_info,
        'log_header': log_header,
        'config_path': config_yaml_path,
        'log_header_path': log_header_path,
    }

