# UrbanGraphEmbedding
![Framework overview](overall.png)

This repository implements a **two-stage training framework** for learning spatially-aware vision-language model (VLM) embeddings that capture urban spatial relationships beyond visual appearance alone.

## Overview

### Motivation

Urban understanding is not simply visual and articulated, but **spatial and relational**. The semantics of a location is not solely determined by what is visible in a single street-view image or expressible in a local caption, but also by how the location is positioned within the city's topology—such as its connectivity, proximity, and broader neighborhood structure.

Current urban VLMs rely on instruction tuning and QA-style supervision, encoding spatial information implicitly through language and confining it to isolated question contexts, resulting in fragmented, task-bound representations that lack explicit urban grounding.

### Our Approach

We propose a **two-stage training framework** that progressively injects spatial knowledge into VLM embeddings while preserving their visual–language alignment:

1. **Stage 1**: Establishes robust image–text alignment via instruction-conditioned contrastive learning with spatial reasoning cues
2. **Stage 2**: Injects structured spatial knowledge into image representations through a graph encoder that propagates topological and relational information from localized spatial subgraphs

This progressive approach prevents unstable optimization that can occur when naively combining images, text, and spatial graphs during training, as pretrained visual–language representations are sensitive to abrupt or unbalanced updates introduced by an additional modality.


## Download data

Please download the training data and bechmarks from https://pan.quark.cn/s/c4b059eebab5 and put them into mydata and benchmark folder respectively.

## Prerequisites

- Python 3.10+
- CUDA-capable GPUs (4 GPUs recommended, 30GB+ VRAM each)
- DeepSpeed (for distributed training)
- PyTorch with CUDA support
- Swift framework dependencies

### Environment Setup

```bash
# Set CUDA_HOME if not auto-detected
export CUDA_HOME=/usr/local/cuda-12.0  # Adjust to your CUDA installation

# Set GPU visibility (adjust based on your setup)
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

## Stage 1: Image-Text Alignment with Spatial Reasoning

Stage 1 establishes robust image–text alignment through instruction-conditioned contrastive learning. Spatial awareness is first evoked through textual spatial reasoning cues embedded in the training instructions.

### Dataset Format

Stage 1 uses JSONL files with image–text pairs. Each line should contain:

```json
{
  "messages": [
    {"role": "user", "content": "<image> [instruction]"}
  ],
  "positive_messages": [
    [
      {"role": "user", "content": "[detailed spatial description]"}
    ]
  ],
  "images": ["/path/to/image.jpg"],
  "graphs": [],
  "pair_type": "stage1_image_only",
  "label": 1
}
```

**Key fields:**
- `messages`: Query instruction with `<image>` token
- `positive_messages`: Positive text descriptions with spatial context
- `images`: Array of image file paths (absolute paths)
- `graphs`: Empty array for stage 1


### Running Stage 1

```bash examples/train/embedding/run_stage1.sh
```

### Configuration

Edit `run_stage1.sh` to customize:

- **Dataset path**: Modify `DATASET_BASE` (line 78)
  ```bash
  DATASET_BASE="/path/to/your/stage1_image_text_pairs_complete"
  ```

- **Model**: Change the model in the `swift sft` command (line 98)
  ```bash
  --model iic/gme-Qwen2-VL-2B-Instruct
  ```

- **GPU settings**: Adjust `CUDA_VISIBLE_DEVICES` and `nproc_per_node` (lines 74, 94)
  ```bash
  nproc_per_node=4  # Number of GPUs
  CUDA_VISIBLE_DEVICES=0,1,2,3
  ```

- **Training hyperparameters**:
  - `--per_device_train_batch_size 8`: Batch size per GPU
  - `--gradient_accumulation_steps 2`: Effective batch size multiplier
  - `--learning_rate 5e-5`: Learning rate
  - `--num_train_epochs 5`: Number of training epochs
  - `--output_dir output/stage1_qwen25vl7b`: Output directory

### Stage 1 Output

The training will save checkpoints to `output/stage1_qwen25vl7b/` (or your specified output directory). The final checkpoint path will be needed for Stage 2:

```
output/stage1_qwen25vl7b/checkpoint-XXXX/
```

## Stage 2: Graph-Enhanced Spatial Embedding

Stage 2 adds a graph encoder to inject structured spatial knowledge into the vision-language model trained in Stage 1. The graph encoder propagates topological and relational information from localized spatial subgraphs.

### Dataset Format

Stage 2 uses JSONL files with image–text–graph triplets. Each line should contain:

```json
{
  "messages": [
    {"role": "user", "content": "<graph><image> [instruction]"}
  ],
  "positive_messages": [
    [
      {"role": "user", "content": "[spatial description with graph context]"}
    ]
  ],
  "images": ["/path/to/image.jpg"],
  "graphs": ["/path/to/graph.json"],
  "pair_type": "stage2_image_graph",
  "label": 1
}
```

**Key fields:**
- `messages`: Query instruction with `<image>` token
- `positive_messages`: Positive text descriptions
- `images`: Array of image file paths (absolute paths)
- `graphs`: Array of graph file paths (JSON format with nodes, edges, spatial features)


**Graph format** (JSON file):
```json
{
  "nodes": [
    {
      "node_id": "node_0",
      "coords": "...",
      "node_text": "..."
    }
  ],
  "edges": [
    {
      "source": "node_0",
      "target": "node_1",
    }
  ],
  "center_node": "node_0"
}
```

### Running Stage 2

```bash examples/train/embedding/run_stage2.sh
```

### Configuration

Edit `swift/cli/sft_debug.py` to customize Stage 2 training:

#### Dataset Paths (lines 286-330)

```python
_raw_dataset_paths = [
    "/path/to/your/data/*stage2.jsonl",
    # Add more paths as needed
]
```

#### Model Configuration (lines 341-352)

```python
BASE_MODEL = 'Qwen/Qwen2.5-VL-7B-Instruct'
template='qwen2_5_vl_graph'  # Graph-enabled template
```

#### Graph Encoder Settings (lines 369-389)

```python
use_graph_encoder=True,
graph_num_layers=2,  # Number of GNN layers
edge_dim=16,  # Edge embedding dimension
graph_max_nodes=1000,  # Maximum nodes per graph

# Spatial encoding (PE-GNN style)
use_spatial_encoding=True,
spatial_embed_dim=64,
spatial_frequency_num=16,

# Edge features (GeoGNN style)
use_edge_features=True,
edge_use_distance=True,  # Haversine distance
edge_use_direction=True,  # Bearing angle
edge_use_displacement=True,  # Δlat, Δlon

# GNN type
use_gat=True,  # Use GATv2 (Graph Attention)
gat_heads=4,  # Number of attention heads
```

#### Training Hyperparameters (lines 407-438)

```python
num_train_epochs=5,
per_device_train_batch_size=4,
gradient_accumulation_steps=2,
learning_rate=5e-5,
warmup_ratio=0.05,
weight_decay=0.1,
max_grad_norm=1.0,

# Fine-tune learning rates for different components
image_text_lr_scale=0.1,  # Scale down LR for image/text encoders
graph_lr=5e-5,  # Learning rate for graph encoder
```

#### Checkpoint Resume (lines 519-527)

```python
resume_from_checkpoint="/path/to/stage1/checkpoint-XXXX/",
resume_only_model=True,  # Only load model weights, not training state
ignore_data_skip=True,  # Start training from step 0
```

#### Output Directory (line 451)

```python
output_dir='output/stage2_qwen25vl7b_edge_embed_16',
```

### Stage 2 Output

The training will save checkpoints to the output directory specified in `sft_debug.py`. Experiment tracking creates timestamped subdirectories:

```
output/stage2_qwen25vl7b_edge_embed_16/YYYYMMDD-HHMMSS_experiment_name/checkpoint-XXXX/
```

## Troubleshooting

### Out of Memory (OOM) Errors

- **Reduce batch size**: Decrease `per_device_train_batch_size`
- **Increase gradient accumulation**: Increase `gradient_accumulation_steps` to maintain effective batch size
- **Reduce image resolution**: Lower `MAX_PIXELS` (e.g., 600000 → 300000)
- **Limit graph size**: Reduce `graph_max_nodes` (e.g., 1000 → 500)


### CUDA/DeepSpeed Issues

The scripts automatically handle CUDA detection and DeepSpeed configuration. If you encounter issues:

```bash
# Manually set CUDA_HOME
export CUDA_HOME=/usr/local/cuda-12.0

# Verify CUDA is accessible
nvcc --version
```

## Evaluation

After training, evaluate the learned representations using the benchmark evaluation script `run_eval_urban_ranking_multiview.sh`. This script runs comprehensive evaluations across multiple benchmark datasets.

### Running Benchmark Evaluation

```bash run_eval_urban_ranking_multiview.sh
```

### Configuration

Edit `run_eval_urban_ranking_multiview.sh` to customize evaluation settings:

#### Model and Checkpoint Configuration (lines 44-72)

```bash
# Base model name
MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"

# Path to trained checkpoint (from Stage 2 output)
ADAPTERS_PATH="output/stage2_qwen25vl7b/v0-YYYYMMDD-HHMMSS/.../checkpoint-XXXX/"
```

#### Output Directory (line 31)

```bash
OUTPUT_DIR="eval_output/stage2_qwen25vl7b"
```

#### Inference Parameters (lines 73-83)

```bash
BATCH_SIZE=1                    # Batch size for inference
DEVICE_MAP="cuda:0"            # GPU device
MAX_PIXELS=600000              # Maximum image pixels
GRAPH_MAX_NODES=900            # Maximum nodes per graph
TORCH_DTYPE="bfloat16"         # Data type for inference
ATTN_IMPL="eager"              # Attention implementation ("eager" or "sdpa")
K_VALUES="1,3,5,10"           # K values for ranking metrics
EVAL_LIMIT=700                 # Limit number of samples per dataset
```

### Benchmark Datasets

The script evaluates on multiple benchmark folders:

1. **Geolocation**: Location-based ranking tasks
2. **Perception**: Visual perception tasks
3. **Retrieval**: Image retrieval tasks (uses `eval_image_retrieval.py`)
4. **Spatial Reasoning**: Spatial reasoning tasks by city:
   - `by_type_paris`
   - `by_type_beijing`
   - `by_type_newyork`
   - `by_type_singapore`

Each benchmark can be evaluated in two modes:
- **image_only**: Uses image-only queries
- **with_graph**: Uses graph-enhanced fusion queries

### Evaluation Modes

The script supports different query modes:
- `--query_mode "image"`: Image-only queries
- `--query_mode "fusion"`: Graph-image fusion queries
- `--query_mode "text"`: Text queries (for retrieval tasks)

### Output

Evaluation results are saved to the specified `OUTPUT_DIR` with metrics including:
- Recall@K (for K=1, 3, 5, 10)
- Ranking accuracy
- Per-dataset performance summaries

Results are organized by benchmark folder and query mode for easy analysis.



## References

- **VLM2Vec**: [Paper](https://arxiv.org/pdf/2507.04590)
- **PE-GNN**: [Paper](https://arxiv.org/abs/2111.10144) - Positional Encoder Graph Neural Networks for Geographic Data
- **Swift Framework**: [Documentation](https://github.com/modelscope/swift)




