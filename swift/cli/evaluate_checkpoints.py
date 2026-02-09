#!/usr/bin/env python3
"""
evaluate_checkpoints.py

Evaluate multiple checkpoints from a training run and select the best one.

This script:
1. Finds all checkpoints in a training output directory
2. Loads each checkpoint and evaluates on the saved validation dataset
3. Selects the best checkpoint based on evaluation metrics
4. Saves results to a summary file

Usage:
    python swift/cli/evaluate_checkpoints.py \
        --training_output_dir /home/xingtong/ms_swift/output/stage2_qwen25vl7b_edge_feature/v0-20251219-205258/20251219-205258_Qwen2.5_VL_7B_Instruct_graph_spatial_edgefeat_gat4_infonce \
        --val_dataset_path output/stage2_qwen25vl7b_test/v0-20260201-100213/20260201-100214_Qwen2.5_VL_7B_Instruct_graph_spatial_edgefeat_gat4_infonce/val_dataset.jsonl \
        --output_dir output/checkpoint_selection \
        --batch_size 4 \
        --metric_for_best margin
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from swift.llm import TrainArguments
from swift.utils import get_logger, seed_everything

logger = get_logger()


def find_checkpoints(training_output_dir: str) -> List[str]:
    """
    Find all checkpoint directories in the training output directory.
    
    Args:
        training_output_dir: Path to training output directory
        
    Returns:
        List of checkpoint directory paths, sorted by checkpoint number
    """
    checkpoints = []
    training_dir = Path(training_output_dir)
    
    if not training_dir.exists():
        raise ValueError(f"Training output directory does not exist: {training_output_dir}")
    
    # Look for checkpoint-* directories
    for checkpoint_dir in training_dir.glob("checkpoint-*"):
        if checkpoint_dir.is_dir():
            # Extract checkpoint number
            match = re.search(r"checkpoint-(\d+)", checkpoint_dir.name)
            if match:
                checkpoint_num = int(match.group(1))
                checkpoints.append((checkpoint_num, str(checkpoint_dir)))
    
    # Sort by checkpoint number
    checkpoints.sort(key=lambda x: x[0])
    return [cp[1] for cp in checkpoints]




def evaluate_checkpoint(
    checkpoint_path: str,
    val_dataset_path: str,
    model_name: str,
    template: str,
    batch_size: int = 4,
    device_map: str = "auto",
    torch_dtype: str = "bfloat16",
    max_pixels: Optional[int] = None,
    limit: Optional[int] = None,
    training_args_path: Optional[str] = None,
) -> Dict[str, float]:
    """
    Evaluate a single checkpoint on the validation dataset using SwiftSft framework.
    
    Args:
        checkpoint_path: Path to checkpoint directory
        val_dataset_path: Path to validation dataset JSONL file
        model_name: Base model name
        template: Template name
        batch_size: Batch size for evaluation
        device_map: Device map for model loading
        torch_dtype: Torch dtype
        max_pixels: Maximum pixels for images
        limit: Limit number of samples (for testing)
        training_args_path: Path to training_args.json (for loading config)
        
    Returns:
        Dictionary of evaluation metrics
    """
    logger.info(f"Evaluating checkpoint: {checkpoint_path}")
    
    try:
        from swift.llm import sft_main, TrainArguments
        from swift.llm.dataset import load_dataset
        
        # Load training args if available to get full config
        train_args_kwargs = {}
        if training_args_path and os.path.exists(training_args_path):
            try:
                with open(training_args_path, 'r') as f:
                    saved_args = json.load(f)
                    # Extract relevant args - include all graph encoder and model config args
                    graph_encoder_keys = [
                        'task_type', 'template', 'loss_type', 
                        'use_graph_encoder', 'graph_num_layers', 'edge_dim', 'graph_max_nodes',
                        'use_spatial_encoding', 'spatial_embed_dim', 'spatial_frequency_num',
                        'use_edge_features', 'edge_use_distance', 'edge_use_direction', 'edge_use_displacement',
                        'use_gat', 'gat_heads',
                        'attn_impl',  # Attention implementation
                        'optimizer',  # Optimizer type (for compatibility)
                    ]
                    for key in graph_encoder_keys:
                        if key in saved_args:
                            train_args_kwargs[key] = saved_args[key]
            except Exception as e:
                logger.warning(f"Could not load training args: {e}")
        
        # Use provided template or from saved args
        template_name = template or train_args_kwargs.get('template', 'qwen2_5_vl_graph')
        
        # Create evaluation arguments
        # Extract graph encoder args with defaults matching sft_debug.py
        # NOTE: TrainArguments requires a dataset, so we use the validation dataset
        # and set split_dataset_ratio=0.0 to prevent splitting
        eval_args = TrainArguments(
            model=model_name,
            task_type=train_args_kwargs.get('task_type', 'embedding'),
            template=template_name,
            dataset=[val_dataset_path],  # Use val dataset as dataset (required by TrainArguments)
            val_dataset=[val_dataset_path],  # Also set as val_dataset for evaluation
            split_dataset_ratio=0.0,  # Don't split since we're only evaluating
            per_device_eval_batch_size=batch_size,
            eval_limit=limit,
            resume_from_checkpoint=checkpoint_path,
            resume_only_model=True,
            device_map=device_map,
            torch_dtype=torch_dtype,
            max_pixels=max_pixels,
            eval_strategy='no',  # We'll run evaluation manually
            output_dir=os.path.join(checkpoint_path, 'eval_temp'),
            # Graph encoder args - match sft_debug.py defaults
            use_graph_encoder=train_args_kwargs.get('use_graph_encoder', True),
            graph_num_layers=train_args_kwargs.get('graph_num_layers', 2),
            edge_dim=train_args_kwargs.get('edge_dim', 64),
            graph_max_nodes=train_args_kwargs.get('graph_max_nodes', 1000),
            # Spatial encoding args
            use_spatial_encoding=train_args_kwargs.get('use_spatial_encoding', True),
            spatial_embed_dim=train_args_kwargs.get('spatial_embed_dim', 64),
            spatial_frequency_num=train_args_kwargs.get('spatial_frequency_num', 16),
            # Edge features args
            use_edge_features=train_args_kwargs.get('use_edge_features', True),
            edge_use_distance=train_args_kwargs.get('edge_use_distance', True),
            edge_use_direction=train_args_kwargs.get('edge_use_direction', True),
            edge_use_displacement=train_args_kwargs.get('edge_use_displacement', True),
            # GNN type args
            use_gat=train_args_kwargs.get('use_gat', True),
            gat_heads=train_args_kwargs.get('gat_heads', 4),
            # Other model config args
            attn_impl=train_args_kwargs.get('attn_impl', 'sdpa'),
            # Pass through any other args that might be needed
            **{k: v for k, v in train_args_kwargs.items() if k not in [
                'task_type', 'template', 'use_graph_encoder', 'graph_num_layers', 'edge_dim', 'graph_max_nodes',
                'use_spatial_encoding', 'spatial_embed_dim', 'spatial_frequency_num',
                'use_edge_features', 'edge_use_distance', 'edge_use_direction', 'edge_use_displacement',
                'use_gat', 'gat_heads', 'attn_impl', 'optimizer'
            ]}
        )
        
        # Run evaluation using SwiftSft
        # Following the pattern from eval_urban_ranking_multiview.py:
        # 1. Load model with checkpoint (graph encoder should be loaded automatically)
        # 2. If graph encoder exists, reload its state dict to ensure it's properly loaded
        # 3. Move to device AFTER loading state dict (avoids meta tensor issue)
        from swift.llm.train.sft import SwiftSft
        import torch
        
        pipeline = SwiftSft(eval_args)
        pipeline._prepare_template()
        
        # Load validation dataset
        val_dataset, _ = load_dataset([val_dataset_path], split_dataset_ratio=0.0)
        val_dataset, _ = pipeline._encode_dataset(val_dataset, None, pre_process=True)
        
        # Prepare model - this will load the checkpoint including graph encoder if it exists
        pipeline.model = pipeline.prepare_model(
            pipeline.args, pipeline.model, template=pipeline.template, train_dataset=None
        )
        
        # Check if graph encoder exists and needs state dict reloading
        # This follows the pattern from eval_urban_ranking_multiview.py:load_graph_encoder
        checkpoint_path_obj = Path(checkpoint_path)
        graph_encoder_bin = checkpoint_path_obj / "graph_encoder.bin"
        
        if graph_encoder_bin.exists() and hasattr(pipeline.model, 'graph_encoder'):
            # Graph encoder exists - reload state dict to ensure it's properly loaded
            # This avoids meta tensor issues by loading real data before moving to device
            try:
                graph_encoder = pipeline.model.graph_encoder
                model_device = next(pipeline.model.parameters()).device
                model_dtype = next(pipeline.model.parameters()).dtype
                
                # Load state dict from checkpoint (this populates tensors with real data)
                state_dict = torch.load(graph_encoder_bin, map_location="cpu")
                
                # Load state dict (replaces any meta tensors with real data)
                missing, unexpected = graph_encoder.load_state_dict(state_dict, strict=False)
                if missing:
                    logger.warning(f"Missing parameters when loading graph encoder: {missing[:5]}")
                if unexpected:
                    logger.warning(f"Unexpected parameters when loading graph encoder: {unexpected[:5]}")
                
                # Now move to device AFTER loading state dict (avoids meta tensor issue)
                # This is the key: load data first, then move to device
                graph_encoder.to(device=model_device, dtype=model_dtype)
                graph_encoder.eval()
                logger.info(f"✅ Graph encoder reloaded from checkpoint and moved to device {model_device}")
            except Exception as e:
                logger.warning(f"Could not reload graph encoder state dict: {e}. Continuing with existing graph encoder.")
        elif graph_encoder_bin.exists() and not hasattr(pipeline.model, 'graph_encoder'):
            # Graph encoder checkpoint exists but wasn't attached - this shouldn't happen
            logger.warning("Graph encoder checkpoint exists but graph_encoder not found on model. "
                         "This may indicate an issue with checkpoint loading.")
        
        from swift.trainers import TrainerFactory
        trainer_cls = TrainerFactory.get_trainer_cls(pipeline.args)
        trainer = trainer_cls(
            model=pipeline.model,
            args=pipeline.args.training_args,
            data_collator=pipeline._get_data_collator(),
            train_dataset=None,
            eval_dataset=val_dataset,
            template=pipeline.template,
        )
        
        # Run evaluation
        eval_results = trainer.evaluate()
        
        # Extract metrics
        metrics = {}
        for key, value in eval_results.items():
            if isinstance(value, (int, float)):
                metrics[key] = float(value)
            elif isinstance(value, np.number):
                metrics[key] = float(value)
        
        logger.info(f"Checkpoint {Path(checkpoint_path).name} metrics: {metrics}")
        return metrics
        
    except Exception as e:
        logger.error(f"Error evaluating checkpoint {checkpoint_path}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {}
    finally:
        # Clean up
        import torch
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()






def select_best_checkpoint(
    checkpoint_results: Dict[str, Dict[str, float]],
    metric_for_best: str = "margin",
) -> Tuple[str, Dict[str, float]]:
    """
    Select the best checkpoint based on a metric.
    
    Args:
        checkpoint_results: Dictionary mapping checkpoint paths to metrics
        metric_for_best: Metric name to use for selection (e.g., 'margin', 'eval_margin')
        
    Returns:
        Tuple of (best_checkpoint_path, best_metrics)
    """
    if not checkpoint_results:
        return None, {}
    
    best_checkpoint = None
    best_score = -1.0
    
    for checkpoint_path, metrics in checkpoint_results.items():
        # Try both the metric name as-is and with 'eval_' prefix
        metric_key = metric_for_best
        if metric_key not in metrics:
            # Try with eval_ prefix
            eval_metric_key = f"eval_{metric_for_best}"
            if eval_metric_key in metrics:
                metric_key = eval_metric_key
            else:
                continue
        
        score = metrics[metric_key]
        if score > best_score:
            best_score = score
            best_checkpoint = checkpoint_path
    
    if best_checkpoint:
        return best_checkpoint, checkpoint_results[best_checkpoint]
    
    return None, {}


def main():
    parser = argparse.ArgumentParser(description="Evaluate checkpoints and select the best one")
    parser.add_argument(
        "--training_output_dir",
        type=str,
        required=True,
        help="Path to training output directory containing checkpoints",
    )
    parser.add_argument(
        "--val_dataset_path",
        type=str,
        required=True,
        help="Path to validation dataset JSONL file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for evaluation results",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Base model name (if not in checkpoint config)",
    )
    parser.add_argument(
        "--template",
        type=str,
        default=None,
        help="Template name (if not in checkpoint config)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default="auto",
        help="Device map for model loading",
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Torch dtype",
    )
    parser.add_argument(
        "--max_pixels",
        type=int,
        default=None,
        help="Maximum pixels for images",
    )
    parser.add_argument(
        "--metric_for_best",
        type=str,
        default="margin",
        help="Metric to use for selecting best checkpoint (default: 'margin' to match training script)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of validation samples (for testing)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    
    args = parser.parse_args()
    
    # Set seed
    seed_everything(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find checkpoints
    logger.info(f"Finding checkpoints in {args.training_output_dir}")
    checkpoints = find_checkpoints(args.training_output_dir)
    logger.info(f"Found {len(checkpoints)} checkpoints")
    
    if not checkpoints:
        logger.error("No checkpoints found!")
        return
    
    # Verify validation dataset exists
    if not os.path.exists(args.val_dataset_path):
        logger.error(f"Validation dataset file does not exist: {args.val_dataset_path}")
        return
    
    # Try to get model name from training args if not provided
    model_name = args.model
    if not model_name:
        # Try to load from training args
        training_args_path = os.path.join(args.training_output_dir, "args.json")
        if os.path.exists(training_args_path):
            try:
                with open(training_args_path, 'r') as f:
                    training_args = json.load(f)
                    model_name = training_args.get("model")
            except Exception:
                pass
    
    if not model_name:
        logger.error("Model name not provided and could not be loaded from training args!")
        return
    
    logger.info(f"Using model: {model_name}")
    
    # Try to get template from training args if not provided
    template_name = args.template
    if not template_name:
        training_args_path = os.path.join(args.training_output_dir, "training_args.json")
        if os.path.exists(training_args_path):
            try:
                with open(training_args_path, 'r') as f:
                    training_args = json.load(f)
                    template_name = training_args.get("template")
            except Exception:
                pass
    
    if not template_name:
        logger.warning("Template not provided, using default 'qwen2_5_vl_graph'")
        template_name = "qwen2_5_vl_graph"
    
    logger.info(f"Using template: {template_name}")
    
    # Get training args path
    training_args_path = os.path.join(args.training_output_dir, "training_args.json")
    
    # Evaluate each checkpoint
    checkpoint_results = {}
    for checkpoint_path in tqdm(checkpoints, desc="Evaluating checkpoints"):
        metrics = evaluate_checkpoint(
            checkpoint_path=checkpoint_path,
            val_dataset_path=args.val_dataset_path,
            model_name=model_name,
            template=template_name,
            batch_size=args.batch_size,
            device_map=args.device_map,
            torch_dtype=args.torch_dtype,
            max_pixels=args.max_pixels,
            limit=args.limit,
            training_args_path=training_args_path,
        )
        
        if metrics:
            checkpoint_results[checkpoint_path] = metrics
    
    if not checkpoint_results:
        logger.error("No checkpoints were successfully evaluated!")
        return
    
    # Select best checkpoint
    best_checkpoint, best_metrics = select_best_checkpoint(
        checkpoint_results,
        metric_for_best=args.metric_for_best,
    )
    
    # Save results
    results = {
        "best_checkpoint": best_checkpoint,
        "best_metrics": best_metrics,
        "metric_for_best": args.metric_for_best,
        "all_results": {
            Path(cp).name: metrics for cp, metrics in checkpoint_results.items()
        },
    }
    
    results_path = os.path.join(args.output_dir, "checkpoint_evaluation_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 70)
    print("CHECKPOINT EVALUATION RESULTS")
    print("=" * 70)
    print(f"\nBest checkpoint (by {args.metric_for_best}):")
    print(f"  {best_checkpoint}")
    print(f"\nBest metrics:")
    for metric, value in best_metrics.items():
        print(f"  {metric}: {value:.4f}")
    print(f"\nFull results saved to: {results_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()

