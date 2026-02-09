# UGData Generation Pipeline

This directory contains scripts for generating urban graph data, including edge construction, spatial reasoning path (SRP) generation, and spatial context-aware image captioning (SCC).

## Overview

The UGData generation pipeline consists of three main components:

1. **Edge Construction** (`make_edges.py`) - Builds spatial graph edges from OpenStreetMap (OSM) data
2. **Spatial Reasoning Path (SRP) Generation** (`srp_generation.py`) - Generates reasoning paths for training spatial reasoning models
3. **Spatial Context Caption Generation** (`generate_captions_with_existing_subgraphs.py`)- Generates image captions enriched with spatial context from graph structures

## Prerequisites

### Required Python Packages

```bash
pip install geopandas pandas numpy networkx shapely geopy requests tqdm pillow openai
```

### Optional GPU Acceleration

For GPU-accelerated path finding (optional):
```bash
conda install -c rapidsai cugraph cudf
```

### Environment Variables

Set the following environment variables for data paths:

```bash
export URBANKG_DATA_ROOT=/path/to/UrbanKG/data
export URBANKG_OUTPUT_ROOT=/path/to/output/data
```

Or modify the `DATA_ROOT` and `OUTPUT_ROOT` variables in each script.

## 1. Edge Construction (`make_edges.py`)

### Purpose

Constructs spatial graph edges from OSM node data, creating connectivity relationships between geographic entities (POIs, crossings, streets, etc.).

### Key Features

- **Reverse Geocoding**: Uses Nominatim API to enrich nodes with address information
- **Edge Creation**: Builds edges between spatially connected nodes
- **Geometry Processing**: Handles various geometry types (Point, LineString, Polygon, MultiLineString, MultiPolygon)
- **Distance Calculation**: Computes geodesic distances between connected nodes

### Usage

```bash
python make_edges.py --place singapore --input nodes.geojson --output edges.geojson
```

### Parameters

- `--place`: Place name (e.g., 'singapore', 'newyork', 'beijing', 'paris')
- `--input`: Input GeoJSON file containing nodes
- `--output`: Output GeoJSON file for edges
- `--max-distance`: Maximum distance threshold for edge creation (default: varies by place)

### Output

Generates a GeoJSON file containing:
- Edge connections between nodes
- Distance attributes
- Geometry information for visualization

## 2. Spatial Reasoning Path (SRP) Generation (`srp_generation.py`)

### Key Features

- **Ultra-Optimized Architecture**: Uses cached adjacency dictionaries instead of NetworkX graphs for 100-1000x speedup
- **S2 Cell Partitioning**: Partitions space using S2 cells for efficient processing
- **Reverse BFS**: Finds reachable nodes using reverse breadth-first search
- **Path Diversity**: Generates diverse paths with different characteristics (distance, node types, etc.)
- **Resume Capability**: Supports resuming from last processed S2 cell
- **GPU Acceleration**: Optional GPU support for large-scale processing

### Architecture

```
1. Build/load adjacency dictionary from edges (cached to disk)
2. For each S2 cell:
   - Select up to N mapillary nodes (configurable)
   - For each mapillary node:
     - Use cached adjacency dict for reverse BFS
     - Find reachable nodes at different hop distances
     - Construct diverse paths
     - Format as training examples
```

### Usage

```bash
python srp_generation.py \
    --data-folder singapore \
    --filename reasoning_paths_singapore \
    --max-images-per-cell 3 \
    --max-hops 3 \
    --max-paths-per-node 10
```

### Parameters

- `--data-folder`: Data folder name (e.g., 'singapore', 'newyork')
- `--filename`: Base filename for output (will create `{filename}.jsonl`)
- `--max-images-per-cell`: Maximum mapillary images to process per S2 cell (default: 3)
- `--max-hops`: Maximum number of hops for path finding (default: 3)
- `--max-paths-per-node`: Maximum paths to generate per mapillary node (default: 10)
- `--s2-level`: S2 cell level for partitioning (default: 18)

### Output Format

Each line in the output JSONL file contains:

```json
{
  "mapillary_node": 123456789,
  "paths": [
    {
      "nodes": [123456789, 987654321, ...],
      "path_length": 150.5,
      "path_type": "shortest",
      "directions": ["north", "east", ...]
    }
  ],
  "s2_cell": "1234567890123456789",
  "timestamp": "2024-01-01T00:00:00"
}
```


## 3. Spatial Context Caption (SCC) Generation

### Overview

Generates image captions enriched with spatial context from graph structures. Two scripts are provided for different use cases:

### 3.1 Caption Generation from Mapillary Results (`generate_captions_from_mapillary_results.py`)

Generates captions for mapillary images by building subgraphs from coordinates.

#### Features

- **Subgraph Construction**: Builds spatial subgraphs around image coordinates
- **Mapillary Node Integration**: Adds mapillary nodes to subgraphs and expands with neighbors
- **Enhanced Prompts**: Creates prompts with comprehensive spatial context
- **Checkpoint Support**: Supports resuming from checkpoints
- **Multiple API Providers**: Supports Qwen VL and Yinli APIs

#### Usage

```bash
python generate_captions_from_mapillary_results.py \
    --place singapore \
    --mapillary_results_file mapillary_results_cleaned.jsonl \
    --training_data_file training_data.jsonl \
    --output_file captions.jsonl \
    --api_provider qwen \
    --subgraph_radius 700
```

#### Parameters

- `--place`: Place name (default: 'singapore')
- `--mapillary_results_file`: Path to mapillary results JSONL file
- `--training_data_file`: Path to training data JSONL (for filtering existing images)
- `--output_file`: Output file for captions
- `--api_key`: API key for caption generation
- `--api_provider`: API provider ('qwen' or 'yinli', default: 'qwen')
- `--delay`: Delay between API calls in seconds (default: 6)
- `--subgraph_radius`: Radius in meters for subgraph creation (default: 700)

#### Output Format

```json
{
  "image_paths": "/path/to/image.jpg",
  "prompt": "Enhanced prompt with spatial context...",
  "image_caption": "Detailed image description...",
  "summarization": "Spatial context summarization...",
  "swift_format": {
    "messages": [...],
    "images": ["/path/to/image.jpg"],
    "graphs": ["/path/to/subgraph.pkl"],
    "label": 1.0
  },
  "mapillary_id": "123456789",
  "coordinates": [103.8500, 1.3000],
  "timestamp": "2024-01-01T00:00:00"
}
```

### 3.2 Caption Generation with Existing Subgraphs (`generate_captions_with_existing_subgraphs.py`)

Generates captions for training data images using pre-existing subgraph files.

#### Features

- **Existing Subgraph Support**: Uses pre-computed subgraph pickle files
- **Subgraph Expansion**: Expands subgraphs to include mapillary nodes if missing
- **Node Text Generation**: Creates node_text and coords attributes for all nodes
- **Resume Capability**: Supports checkpoint-based resuming
- **Path Normalization**: Handles different path formats and normalizes them

#### Usage

```bash
python generate_captions_with_existing_subgraphs.py \
    --place singapore \
    --training_data_file_name reasoning_path_mapillary_swift_no_intersection_nodes_reversed_2km \
    --existing_captions_file existing_captions.jsonl \
    --output_file new_captions.jsonl \
    --api_provider qwen \
    --delay 6
```

#### Parameters

- `--place`: Place name (default: 'singapore')
- `--training_data_file_name`: Base name of training data file (without .jsonl extension)
- `--existing_captions_file`: Path to existing captions file (optional)
- `--output_file`: Output file for new captions
- `--api_key`: API key for caption generation
- `--api_provider`: API provider ('qwen' or 'yinli', default: 'qwen')
- `--delay`: Delay between API calls in seconds (default: 6)

#### Workflow

1. Load checkpoint (if resuming)
2. Load existing captions (to avoid duplicates)
3. Extract image-graph pairs from training data
4. For each image:
   - Load subgraph from pickle file
   - Expand subgraph if mapillary node missing
   - Generate node_text and coords attributes
   - Generate subgraph description
   - Create enhanced prompt
   - Call API to generate caption
   - Extract and save caption with summarization

## Data Directory Structure

```
data/
├── geo/
│   └── SR/
│       └── osm_data/
│           └── {place}/
│               ├── nodes.geojson
│               ├── nodes_with_districts.geojson
│               ├── nodes_mapillary.geojson
│               ├── edges.geojson
│               ├── edges_mapillary.geojson
│               ├── mapillary_results_cleaned.jsonl
│               ├── subgraphs/
│               │   └── *.pkl
│               └── images/
│                   └── *.jpg
└── output/
    └── enhanced_image_data_with_paths_and_captions_{place}/
        ├── captions.jsonl
        ├── subgraphs_with_paths/
        │   └── *.pkl
        └── *_checkpoint.jsonl
```

## Configuration

### Setting Data Paths

All scripts use environment variables or default paths. To customize:

1. **Set Environment Variables**:
   ```bash
   export URBANKG_DATA_ROOT=/path/to/data
   export URBANKG_OUTPUT_ROOT=/path/to/output
   ```

2. **Modify Script Defaults**: Edit the `DATA_ROOT` and `OUTPUT_ROOT` variables in each script.

### API Configuration

For caption generation, set API keys:

```bash
# For Qwen API
export DASHSCOPE_API_KEY=your_qwen_api_key

# For Yinli API
export NEWAPI_API_KEY=your_yinli_api_key
```

Or pass via command line:
```bash
--api_key your_api_key
```

## Common Workflows

### Complete Pipeline

1. **Build Edges**:
   ```bash
   python make_edges.py --place singapore
   ```

2. **Generate SRP Paths**:
   ```bash
   python srp_generation.py --data-folder singapore --filename reasoning_paths
   ```

3. **Create Training Data**:
   ```bash
   python run_training_creation.py --data-folder singapore
   ```

4. **Generate Captions**:
   ```bash
   # Option 1: From mapillary results
   python generate_captions_from_mapillary_results.py --place singapore
   
   # Option 2: With existing subgraphs
   python generate_captions_with_existing_subgraphs.py --place singapore
   ```

### Resuming Interrupted Jobs

Most scripts support resuming:

- **SRP Generation**: Automatically resumes from last processed S2 cell
- **Caption Generation**: Uses checkpoint files (`*_checkpoint.jsonl`)
- **Training Creation**: Auto-detects enhanced subgraphs and resumes

## Troubleshooting

### Memory Issues

If running out of memory:
- Reduce `--max-images-per-cell` in SRP generation
- Process smaller batches
- Use `--no-enhance` flag in training creation to skip graph extension

### API Rate Limits

If hitting API rate limits:
- Increase `--delay` between calls
- Use checkpoint files to resume after interruptions
- Process in smaller batches

### Missing Dependencies

Install missing packages:
```bash
pip install -r requirements.txt
```


## Citation

If you use this code in your research, please cite:

```bibtex
@article{your_paper,
  title={Your Paper Title},
  author={Your Name},
  journal={Journal Name},
  year={2024}
}
```

## License

[Specify your license here]

## Contact

[Your contact information]

