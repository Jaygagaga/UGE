# Copyright (c) Alibaba, Inc. and its affiliates.
"""
Ablation study configurations for graph encoder.

This module provides pre-defined configurations for systematic ablation studies.
Each function returns a dictionary of graph encoder parameters that can be
merged into TrainArguments.

Usage:
    from swift.cli.ablation_configs import get_ablation_config
    
    # Get configuration for a specific ablation study
    graph_config = get_ablation_config('spatial_off')
    
    # Merge into training arguments
    training_arguments = TrainArguments(
        model='Qwen/Qwen2-VL-7B-Instruct',
        # ... other args ...
        **graph_config,  # Merge ablation config
        output_dir='output/ablation_spatial_off',
    )
"""

from typing import Dict, Any

# Baseline configuration (current settings)
BASELINE_CONFIG = {
    'use_graph_encoder': True,
    'graph_num_layers': 2,
    'edge_dim': 64,
    'graph_max_nodes': 800,
    'use_spatial_encoding': True,
    'spatial_embed_dim': 64,
    'spatial_frequency_num': 16,
    'use_edge_features': True,
    'edge_use_distance': True,
    'edge_use_direction': True,
    'edge_use_displacement': True,
    'use_gat': True,
    'gat_heads': 4,
    'use_spatial_auxiliary': False,
    'spatial_loss_weight': 0.1,
}


def get_ablation_config(study_id: str) -> Dict[str, Any]:
    """
    Get graph encoder configuration for a specific ablation study.
    
    Args:
        study_id: Identifier for the ablation study (e.g., 'spatial_off', 'gnn_layers_1')
    
    Returns:
        Dictionary of graph encoder parameters to merge into TrainArguments
    
    Raises:
        ValueError: If study_id is not recognized
    """
    configs = {
        # ═══════════════════════════════════════════════════════════
        # 1. BASELINE COMPARISONS
        # ═══════════════════════════════════════════════════════════
        'baseline_no_graph': {
            'use_graph_encoder': False,
        },
        'baseline_with_graph': BASELINE_CONFIG.copy(),
        
        # ═══════════════════════════════════════════════════════════
        # 2. GNN LAYERS
        # ═══════════════════════════════════════════════════════════
        'gnn_layers_1': {
            **BASELINE_CONFIG,
            'graph_num_layers': 1,
        },
        'gnn_layers_2': {
            **BASELINE_CONFIG,
            'graph_num_layers': 2,
        },
        'gnn_layers_3': {
            **BASELINE_CONFIG,
            'graph_num_layers': 3,
        },
        'gnn_layers_4': {
            **BASELINE_CONFIG,
            'graph_num_layers': 4,
        },
        
        # ═══════════════════════════════════════════════════════════
        # 3. GNN TYPE (GAT vs GCN)
        # ═══════════════════════════════════════════════════════════
        'gnn_gcn': {
            **BASELINE_CONFIG,
            'use_gat': False,
        },
        'gnn_gat_2heads': {
            **BASELINE_CONFIG,
            'use_gat': True,
            'gat_heads': 2,
        },
        'gnn_gat_4heads': {
            **BASELINE_CONFIG,
            'use_gat': True,
            'gat_heads': 4,
        },
        'gnn_gat_8heads': {
            **BASELINE_CONFIG,
            'use_gat': True,
            'gat_heads': 8,
        },
        
        # ═══════════════════════════════════════════════════════════
        # 4. SPATIAL ENCODING
        # ═══════════════════════════════════════════════════════════
        'spatial_off': {
            **BASELINE_CONFIG,
            'use_spatial_encoding': False,
        },
        'spatial_small': {
            **BASELINE_CONFIG,
            'use_spatial_encoding': True,
            'spatial_embed_dim': 32,
            'spatial_frequency_num': 8,
        },
        'spatial_medium': {
            **BASELINE_CONFIG,
            'use_spatial_encoding': True,
            'spatial_embed_dim': 64,
            'spatial_frequency_num': 16,
        },
        'spatial_large': {
            **BASELINE_CONFIG,
            'use_spatial_encoding': True,
            'spatial_embed_dim': 128,
            'spatial_frequency_num': 32,
        },
        
        # ═══════════════════════════════════════════════════════════
        # 5. EDGE FEATURES
        # ═══════════════════════════════════════════════════════════
        'edge_features_off': {
            **BASELINE_CONFIG,
            'use_edge_features': False,
        },
        'edge_distance_bearing_only': {
            **BASELINE_CONFIG,
            'edge_use_distance': True,
            'edge_use_direction': True,
            'edge_use_displacement': False,
        },
        'edge_dim_32': {
            **BASELINE_CONFIG,
            'edge_dim': 32,
        },
        'edge_dim_64': {
            **BASELINE_CONFIG,
            'edge_dim': 64,
        },
        'edge_dim_128': {
            **BASELINE_CONFIG,
            'edge_dim': 128,
        },
        
        # ═══════════════════════════════════════════════════════════
        # 6. SPATIAL AUXILIARY LOSS
        # ═══════════════════════════════════════════════════════════
        'auxiliary_off': {
            **BASELINE_CONFIG,
            'use_spatial_auxiliary': False,
        },
        'auxiliary_on_light': {
            **BASELINE_CONFIG,
            'use_spatial_auxiliary': True,
            'spatial_loss_weight': 0.05,
        },
        'auxiliary_on_medium': {
            **BASELINE_CONFIG,
            'use_spatial_auxiliary': True,
            'spatial_loss_weight': 0.1,
        },
        'auxiliary_on_heavy': {
            **BASELINE_CONFIG,
            'use_spatial_auxiliary': True,
            'spatial_loss_weight': 0.2,
        },
        
        # ═══════════════════════════════════════════════════════════
        # 7. GRAPH COMPLEXITY (MAX NODES)
        # ═══════════════════════════════════════════════════════════
        'max_nodes_400': {
            **BASELINE_CONFIG,
            'graph_max_nodes': 400,
        },
        'max_nodes_800': {
            **BASELINE_CONFIG,
            'graph_max_nodes': 800,
        },
        'max_nodes_1200': {
            **BASELINE_CONFIG,
            'graph_max_nodes': 1200,
        },
        'max_nodes_1600': {
            **BASELINE_CONFIG,
            'graph_max_nodes': 1600,
        },
        
        # ═══════════════════════════════════════════════════════════
        # 8. COMPONENT COMBINATIONS
        # ═══════════════════════════════════════════════════════════
        'minimal_graph': {
            'use_graph_encoder': True,
            'graph_num_layers': 2,
            'edge_dim': 64,
            'graph_max_nodes': 800,
            'use_spatial_encoding': False,
            'spatial_embed_dim': 64,
            'spatial_frequency_num': 16,
            'use_edge_features': False,
            'use_gat': False,  # Use GCN instead
            'gat_heads': 4,
            'use_spatial_auxiliary': False,
            'spatial_loss_weight': 0.1,
        },
        'spatial_only': {
            **BASELINE_CONFIG,
            'use_edge_features': False,
        },
        'edge_only': {
            **BASELINE_CONFIG,
            'use_spatial_encoding': False,
        },
        'spatial_edge': BASELINE_CONFIG.copy(),
        'full_with_auxiliary': {
            **BASELINE_CONFIG,
            'use_spatial_auxiliary': True,
            'spatial_loss_weight': 0.1,
        },
    }
    
    if study_id not in configs:
        available = ', '.join(sorted(configs.keys()))
        raise ValueError(
            f"Unknown ablation study ID: '{study_id}'. "
            f"Available studies: {available}"
        )
    
    return configs[study_id].copy()


def list_ablation_studies() -> Dict[str, str]:
    """
    List all available ablation studies with descriptions.
    
    Returns:
        Dictionary mapping study_id to description
    """
    return {
        # Baseline
        'baseline_no_graph': 'No graph encoder (image-text only baseline)',
        'baseline_with_graph': 'Full graph encoder (current baseline)',
        
        # GNN Layers
        'gnn_layers_1': '1 GNN layer',
        'gnn_layers_2': '2 GNN layers (baseline)',
        'gnn_layers_3': '3 GNN layers',
        'gnn_layers_4': '4 GNN layers',
        
        # GNN Type
        'gnn_gcn': 'GCN instead of GAT',
        'gnn_gat_2heads': 'GAT with 2 attention heads',
        'gnn_gat_4heads': 'GAT with 4 attention heads (baseline)',
        'gnn_gat_8heads': 'GAT with 8 attention heads',
        
        # Spatial Encoding
        'spatial_off': 'Spatial encoding disabled',
        'spatial_small': 'Spatial encoding: 32 dim, 8 frequencies',
        'spatial_medium': 'Spatial encoding: 64 dim, 16 frequencies (baseline)',
        'spatial_large': 'Spatial encoding: 128 dim, 32 frequencies',
        
        # Edge Features
        'edge_features_off': 'Edge features disabled',
        'edge_distance_bearing_only': 'Edge features = distance + bearing (no displacement)',
        'edge_dim_32': 'Edge dimension: 32',
        'edge_dim_64': 'Edge dimension: 64 (baseline)',
        'edge_dim_128': 'Edge dimension: 128',
        
        # Auxiliary Loss
        'auxiliary_off': 'Spatial auxiliary loss disabled (baseline)',
        'auxiliary_on_light': 'Spatial auxiliary loss: weight 0.05',
        'auxiliary_on_medium': 'Spatial auxiliary loss: weight 0.1',
        'auxiliary_on_heavy': 'Spatial auxiliary loss: weight 0.2',
        
        # Max Nodes
        'max_nodes_400': 'Max nodes: 400',
        'max_nodes_800': 'Max nodes: 800 (baseline)',
        'max_nodes_1200': 'Max nodes: 1200',
        'max_nodes_1600': 'Max nodes: 1600',
        
        # Combinations
        'minimal_graph': 'Minimal graph encoder (no spatial, no edge features, GCN)',
        'spatial_only': 'Graph encoder with spatial encoding only',
        'edge_only': 'Graph encoder with edge features only',
        'spatial_edge': 'Graph encoder with both spatial and edge features (baseline)',
        'full_with_auxiliary': 'Full configuration with auxiliary loss',
    }


if __name__ == '__main__':
    """Print all available ablation studies."""
    print("Available Ablation Studies:")
    print("=" * 70)
    studies = list_ablation_studies()
    for study_id, description in sorted(studies.items()):
        print(f"  {study_id:30s} - {description}")
    print("\nUsage:")
    print("  from swift.cli.ablation_configs import get_ablation_config")
    print("  config = get_ablation_config('spatial_off')")





