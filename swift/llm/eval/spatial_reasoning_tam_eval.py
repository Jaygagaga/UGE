import json
import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from matplotlib.font_manager import FontProperties
from matplotlib import rcParams, ft2font
import os
import contextily as cx
CANDIDATE_FONTS = [
    "fonts/NotoSansCJKsc-Regular.otf",

]

CJK_FP = None
for p in CANDIDATE_FONTS:
    if os.path.exists(p):
        try:
            ft2font.FT2Font(p)  # ✅ real validation
            CJK_FP = FontProperties(fname=p)
            print(f"[INFO] Loaded CJK font: {p}")
            break
        except Exception as e:
            print(f"[WARN] Failed loading font {p}: {e}")

if CJK_FP:
    rcParams["axes.unicode_minus"] = False


import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import networkx as nx
from geopy.distance import geodesic
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from swift.llm.argument import EvalArguments
from swift.llm.base import SwiftPipeline
from swift.llm.infer import prepare_model_template
from swift.utils import get_logger, append_to_jsonl

logger = get_logger()

try:
    import umap

    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    logger.warning("UMAP not available. Will use t-SNE for dimensionality reduction.")
import re
_COORDS_IN_TEXT_RE = re.compile(
    r"Coordinates\s*:\s*\(\s*([-+]?\d+\.?\d*)\s*,\s*([-+]?\d+\.?\d*)\s*\)",
    re.IGNORECASE
)

# Matches: ID: 12554
_NODE_ID_RE = re.compile(
    r"\bID\s*:\s*(\d+)\b",
    re.IGNORECASE
)

@dataclass
class SpatialReasoningTAMEvalArguments(EvalArguments):
    """Arguments for TAM-based spatial reasoning evaluation"""

    # Data arguments
    spatial_reasoning_data_path: str = field(
        default="benchmark/spatial_reasoning/by_type_newyork/specific_category_distance_with_graph.jsonl",
        metadata={
            "help": "Path to spatial reasoning evaluation dataset (JSONL format). Specify via --spatial-reasoning-data-path <path>"}
    )
    image_root: str = field(
        default="benchmark/benchmark_images/",
        metadata={"help": "Root directory for images"}
    )
    eval_limit: Optional[int] = field(
        default=None,
        metadata={"help": "Limit evaluation to N samples (useful for testing with just 1 sample)"}
    )

    # TAM visualization arguments
    tam_output_dir: str = field(
        default="eval_output/spatial_reasoning_tam_visualizations",
        metadata={"help": "Directory to save TAM visualizations"}
    )
    visualize_top_k: int = field(
        default=10,
        metadata={"help": "Number of top samples to visualize"}
    )
    save_attention_maps: bool = field(
        default=True,
        metadata={"help": "Save individual attention maps"}
    )

    # TAM computation arguments
    tau_gaussian: float = field(
        default=0.15,
        metadata={"help": "Gaussian filter parameter for TAM"}
    )
    use_causal_inference: bool = field(
        default=True,
        metadata={"help": "Use causal inference module (recommended)"}
    )

    # Processing arguments
    eval_batch_size: int = field(
        default=1,
        metadata={"help": "Batch size for TAM evaluation (must be 1 for proper tracking)"}
    )
    max_new_tokens: int = field(
        default=256,
        metadata={"help": "Maximum tokens to generate"}
    )

    # Spatial analysis arguments
    compute_spatial_decay: bool = field(
        default=True,
        metadata={"help": "Compute spatial decay (distance-based attention analysis)"}
    )
    distance_bins: List[float] = field(
        default_factory=lambda: [0, 200, 500, 1000, 2000, 5000],
        metadata={"help": "Distance bins (meters) for spatial decay analysis"}
    )

    # Comparison arguments
    compare_stage1_stage2: bool = field(
        default=False,
        metadata={"help": "Compare Stage 1 vs Stage 2 models (requires two adapter paths). Uses image-only samples."}
    )
    stage1_adapters: Optional[str] = field(
        default=None,
        metadata={"help": "Path to Stage 1 model adapters"}
    )
    stage2_adapters: Optional[str] = field(
        default="output/stage2_qwen25vl7b_no_edge_feature/v1-20251216-105341/20251216-105342_Qwen2.5_VL_7B_Instruct_graph_spatial_infonce/checkpoint-4025/",
        metadata={"help": "Path to Stage 2 model adapters"}
    )
    image_only_mode: bool = field(
        default=False,
        metadata={"help": "Only process samples without graphs (for Stage 1 vs Stage 2 comparison)"}
    )

    # CRITICAL: TAM requires hidden states tracking
    output_hidden_states: bool = field(
        default=True,
        metadata={"help": "MUST be True for TAM. Tracks hidden states during generation."}
    )
    visualize_embeddings: bool = field(
        default=False,
        metadata={"help": "Visualize embedding space for Stage 1 vs Stage 2 comparison"}
    )
    embedding_reduction_method: str = field(
        default="tsne",
        metadata={"help": "Dimensionality reduction method: 'tsne', 'umap', or 'pca'"}
    )

    def __post_init__(self):
        """Initialize TAM-specific settings"""
        from swift.llm.argument import DeployArguments
        DeployArguments.__post_init__(self)

        # Create output directories
        self.eval_output_dir = os.path.abspath(self.eval_output_dir) if self.eval_output_dir else 'tam_output'
        os.makedirs(self.eval_output_dir, exist_ok=True)
        os.makedirs(self.tam_output_dir, exist_ok=True)

        import datetime as dt
        self.time = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
        self.result_jsonl = os.path.join(self.eval_output_dir, 'spatial_reasoning_tam_eval_result.jsonl')


class TokenActivationMap:
    """
    Token Activation Map computation following the ICCV 2025 paper.

    Computes causal contribution of each token (vision/graph/text) to
    the generation of output tokens.
    """

    def __init__(self, tau: float = 0.15, use_causal_inference: bool = True):
        self.tau = tau
        self.use_causal_inference = use_causal_inference

    def compute_activation_scores_from_cosine(
            self,
            hidden_states: torch.Tensor,
            candidate_embedding: torch.Tensor,
            token_range: Tuple[int, int],
    ) -> np.ndarray:
        """
        Compute token-level activation scores based on cosine similarity
        between each token's hidden state and a candidate embedding.

        This is designed for embedding-based / ranking tasks where the
        downstream score is cos_sim(f(query), f(candidate)) rather than
        next-token logits.

        Args:
            hidden_states: Token hidden states from the last layer.
                Shape: (seq_len, hidden_dim) or (1, seq_len, hidden_dim).
            candidate_embedding: Candidate embedding vector. Shape: (hidden_dim,)
                or (1, hidden_dim).
            token_range: (start, end) indices of tokens to analyze.

        Returns:
            activation_scores: (1, num_tokens) numpy array suitable for
                further processing by TAM (gaussian filter + aggregation).
        """
        start_idx, end_idx = token_range

        # Normalize shapes - handle various input formats
        if hidden_states.dim() == 3:
            # (batch=1, seq_len, hidden_dim) -> (seq_len, hidden_dim)
            hidden_states = hidden_states[0]
        elif hidden_states.dim() == 1:
            # If 1D, this is wrong - we need 2D (seq_len, hidden_dim)
            raise ValueError(
                f"hidden_states must be 2D (seq_len, hidden_dim) or 3D (batch, seq_len, hidden_dim), "
                f"but got shape {hidden_states.shape}"
            )
        elif hidden_states.dim() != 2:
            raise ValueError(
                f"Unexpected hidden_states dimension: {hidden_states.dim()}, shape: {hidden_states.shape}"
            )

        # Now hidden_states should be 2D: (seq_len, hidden_dim)
        if hidden_states.shape[0] == 0:
            return np.zeros((1, 0), dtype=np.float32)

        seq_len, hidden_dim = hidden_states.shape
        start_idx = max(0, start_idx)
        end_idx = min(seq_len, end_idx)
        if start_idx >= end_idx:
            return np.zeros((1, 0), dtype=np.float32)

        # Slice token states for the requested range
        token_states = hidden_states[start_idx:end_idx]  # (num_tokens, hidden_dim)

        # Ensure candidate embedding is 1D
        if candidate_embedding.dim() == 2 and candidate_embedding.size(0) == 1:
            candidate_embedding = candidate_embedding[0]

        # Cosine similarity between each token state and candidate embedding
        token_states_norm = F.normalize(token_states, dim=-1)  # (num_tokens, dim)
        cand_norm = F.normalize(candidate_embedding.view(1, -1), dim=-1)  # (1, dim)
        scores = torch.matmul(token_states_norm, cand_norm.t()).squeeze(-1)  # (num_tokens,)

        # Check for NaN or Inf
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            logger.warning(f"NaN or Inf detected in scores, replacing with zeros")
            scores = torch.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

        # Ensure scores are valid (cosine similarity should be in [-1, 1])
        scores = scores.clamp(min=-1.0, max=1.0)

        # Convert to float32 before numpy (numpy doesn't support bfloat16)
        scores_fp32 = scores.detach().cpu().float()

        # Return with a fake "time" dimension so TAM can treat it like 1-step logits
        result = scores_fp32.numpy()[None, :]  # (1, num_tokens)

        return result

    def compute_activation_scores(
            self,
            logits: List[torch.Tensor],
            generated_ids: torch.Tensor,
            token_range: Tuple[int, int]
    ) -> np.ndarray:
        """
        Compute activation scores for tokens in token_range.

        Args:
            logits: List of logit tensors for each generation step
            generated_ids: Generated token IDs (1D tensor)
            token_range: (start, end) indices of tokens to analyze

        Returns:
            activation_scores: (num_steps, num_tokens) array
        """
        start_idx, end_idx = token_range
        num_tokens = end_idx - start_idx
        num_steps = len(logits)

        activation_scores = np.zeros((num_steps, num_tokens))

        for step_idx, step_logits in enumerate(logits):
            # step_logits: (batch=1, seq_len, vocab_size)
            if step_logits.dim() == 3:
                step_logits = step_logits.squeeze(0)  # (seq_len, vocab_size)

            # Get the generated token for this step
            if step_idx < len(generated_ids):
                generated_token_id = generated_ids[step_idx].item()

                # Extract logits for tokens in range
                if start_idx < step_logits.size(0):
                    relevant_logits = step_logits[start_idx:end_idx]  # (num_tokens, vocab_size)

                    # Get probability of generated token from each position
                    probs = F.softmax(relevant_logits, dim=-1)
                    token_probs = probs[:, generated_token_id]  # (num_tokens,)

                    activation_scores[step_idx] = token_probs.cpu().detach().numpy()

        return activation_scores

    def apply_rank_gaussian_filter(self, scores: np.ndarray) -> np.ndarray:
        """Apply rank-based Gaussian filter to reduce noise."""
        if scores.size == 0:
            return scores

        filtered = np.zeros_like(scores)
        for step_idx in range(scores.shape[0]):
            step_scores = scores[step_idx]
            if step_scores.max() > 0:
                ranks = np.argsort(np.argsort(step_scores))
                normalized_ranks = ranks / (len(ranks) - 1) if len(ranks) > 1 else np.zeros_like(ranks)
                gaussian_weights = np.exp(-(1 - normalized_ranks) ** 2 / (2 * self.tau ** 2))
                filtered[step_idx] = step_scores * gaussian_weights
            else:
                filtered[step_idx] = step_scores

        return filtered

    def aggregate_scores(self, activation_scores: np.ndarray, method: str = 'mean') -> np.ndarray:
        """Aggregate activation scores across generation steps."""
        if method == 'mean':
            return np.mean(activation_scores, axis=0)
        elif method == 'max':
            return np.max(activation_scores, axis=0)
        elif method == 'sum':
            return np.sum(activation_scores, axis=0)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")


class SpatialReasoningTAMEval(SwiftPipeline):
    """TAM-based spatial reasoning evaluator with spatial decay analysis"""

    args_class = SpatialReasoningTAMEvalArguments
    args: args_class

    def __init__(self, args: Union[List[str], SpatialReasoningTAMEvalArguments, None] = None):
        super().__init__(args)

        # Log CUDA device information for debugging
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "not set")
        logger.info("CUDA_VISIBLE_DEVICES: %s", cuda_visible)
        if torch.cuda.is_available():
            logger.info("PyTorch sees %d CUDA device(s):", torch.cuda.device_count())
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                free_mem = torch.cuda.memory_reserved(i) / 1e9 if torch.cuda.memory_reserved(i) > 0 else 0
                total_mem = props.total_memory / 1e9
                logger.info("  Device %d: %s (%.2f GB total)", i, props.name, total_mem)
        else:
            logger.warning("CUDA is not available!")

        # Set device - when CUDA_VISIBLE_DEVICES is set, cuda:0 refers to the first visible device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Using device: %s", self.device)

        # Cache for nodes.geojson lookups (keyed by city name). This is used as a
        # fallback when graph nodes do not store explicit lon/lat coordinates.
        self._nodes_geojson_cache: Dict[str, Any] = {}
        self._nodes_geojson_index_cache: Dict[str, Any] = {}

        # Initialize model and template
        logger.info(f"Loading model with backend: {self.args.infer_backend}")
        self.model, self.template = prepare_model_template(self.args)

        # Log which device the model is actually on
        if hasattr(self.model, 'parameters'):
            try:
                model_device = next(self.model.parameters()).device
                logger.info("Model loaded on device: %s", model_device)
            except Exception:
                pass

        # Set model reference on template
        self.template.model = self.model

        # Load tokenizer and graph encoder from checkpoint
        self._load_checkpoint_components()

        # Set template to inference mode
        self.template.set_mode('pt')

        # Initialize TAM computer
        self.tam_computer = TokenActivationMap(
            tau=self.args.tau_gaussian,
            use_causal_inference=self.args.use_causal_inference
        )
        city='singapore'
        self.gdf = self._load_nodes_geojson(city)

        # Configure model output settings
        if hasattr(self.model, 'config'):
            self.model.config.output_hidden_states = self.args.output_hidden_states
            if not self.args.output_hidden_states:
                logger.warning("⚠️  TAM requires output_hidden_states=True! Setting it now.")
                self.model.config.output_hidden_states = True

        logger.info(f"Spatial reasoning TAM evaluation initialized. Output dir: {self.args.tam_output_dir}")

    def _load_checkpoint_components(self):
        """Load tokenizer and graph encoder from checkpoint"""
        if not self.args.adapters:
            logger.warning("No adapters specified, skipping checkpoint loading")
            return

        adapter_path = self.args.adapters
        if isinstance(adapter_path, list):
            adapter_path = adapter_path[0]

        logger.info(f"✅ Adapters loaded from: {adapter_path}")

        # Load graph_encoder if needed
        if self.args.template in ('qwen2_vl_graph', 'qwen2_5_vl_graph'):
            if not hasattr(self.model, 'graph_encoder'):
                logger.info("🔧 Loading graph_encoder from checkpoint...")
                # try:
                from swift.llm.model.model.graph_encoder_spatial import TextAttributedGraphEncoderSpatial

                graph_encoder_path = os.path.join(adapter_path, 'graph_encoder.bin')
                graph_config_path = os.path.join(adapter_path, 'graph_encoder_config.json')

                if os.path.exists(graph_encoder_path):
                    # Load checkpoint early to inspect shapes for parameter inference
                    graph_encoder_state = torch.load(graph_encoder_path, map_location='cpu')

                    if os.path.exists(graph_config_path):
                        with open(graph_config_path, 'r') as f:
                            saved_config = json.load(f)
                    else:
                        saved_config = {}

                    # Defaults mirror training-time init (see init_graph_encoder)
                    defaults = {
                        'hidden_dim': self.model.config.hidden_size,
                        'output_dim': self.model.config.hidden_size,
                        'num_layers': 2,
                        'edge_dim': 64,
                        'use_spatial_encoding': True,
                        'spatial_embed_dim': 128,
                        'spatial_frequency_num': 8,
                        'use_edge_features': True,
                        'use_gat': True,
                        'gat_heads': 4,
                        'edge_use_distance': True,
                        'edge_use_direction': True,
                        'edge_use_displacement': True,
                    }

                    # Merge with defaults but ignore None values and deprecated parameters
                    graph_config = defaults.copy()
                    for k, v in saved_config.items():
                        if k != 'use_spatial_auxiliary' and v is not None:
                            graph_config[k] = v

                    # Infer parameters from checkpoint to ensure compatibility
                    try:
                        # Infer spatial_frequency_num from spatial encoder projection
                        spatial_proj_key = "spatial_encoder.projection.0.weight"
                        if spatial_proj_key in graph_encoder_state:
                            weight_shape = graph_encoder_state[spatial_proj_key].shape
                            if len(weight_shape) == 2:
                                input_dim = weight_shape[1]
                                inferred_freq_num = input_dim // 4
                                if inferred_freq_num > 0 and input_dim % 4 == 0:
                                    graph_config['spatial_frequency_num'] = inferred_freq_num
                                    logger.info(
                                        "Inferred spatial_frequency_num=%d from checkpoint "
                                        "(spatial_encoder.projection.0.weight shape: %s)",
                                        inferred_freq_num, weight_shape
                                    )

                        # Infer gat_heads from GNN layer attention or linear layer shapes
                        att_key = "convs.0.att"
                        if att_key in graph_encoder_state:
                            att_shape = graph_encoder_state[att_key].shape
                            if len(att_shape) >= 2:
                                inferred_gat_heads = att_shape[1]
                                graph_config['gat_heads'] = int(inferred_gat_heads)
                                logger.info(
                                    "Inferred gat_heads=%d from checkpoint (convs.0.att shape: %s)",
                                    inferred_gat_heads, att_shape
                                )
                        else:
                            # Fallback: infer from linear layer shape
                            lin_l_key = "convs.0.lin_l.weight"
                            if lin_l_key in graph_encoder_state:
                                lin_l_shape = graph_encoder_state[lin_l_key].shape
                                if len(lin_l_shape) == 2 and lin_l_shape[1] > 0:
                                    hidden_dim = lin_l_shape[1]
                                    if lin_l_shape[0] % hidden_dim == 0:
                                        inferred_gat_heads = lin_l_shape[0] // hidden_dim
                                        graph_config['gat_heads'] = int(inferred_gat_heads)
                                        logger.info(
                                            "Inferred gat_heads=%d from checkpoint "
                                            "(convs.0.lin_l.weight shape: %s, hidden_dim: %d)",
                                            inferred_gat_heads, lin_l_shape, hidden_dim
                                        )

                        # Infer edge_dim and enabled components from edge_combiner layer shape
                        edge_combiner_key = "edge_encoder.edge_combiner.0.weight"
                        if edge_combiner_key in graph_encoder_state:
                            combiner_shape = graph_encoder_state[edge_combiner_key].shape
                            logger.info("Found edge_combiner.0.weight in checkpoint with shape: %s", combiner_shape)
                            if len(combiner_shape) == 2:
                                output_dim = combiner_shape[0]  # edge_embed_dim * 2
                                input_dim = combiner_shape[1]  # edge_embed_dim * enabled_components

                                # Infer edge_dim from output_dim
                                candidate_edge_dim = output_dim // 2
                                if candidate_edge_dim > 0 and output_dim % 2 == 0:
                                    # Infer enabled_components from input_dim
                                    enabled_comp = input_dim // candidate_edge_dim
                                    if enabled_comp > 0 and input_dim % candidate_edge_dim == 0:
                                        graph_config['edge_dim'] = candidate_edge_dim
                                        logger.info(
                                            "Inferred edge_dim=%d from checkpoint "
                                            "(edge_encoder.edge_combiner.0.weight shape: %s)",
                                            candidate_edge_dim, combiner_shape
                                        )

                                        # Infer which edge components were enabled
                                        if enabled_comp == 1:
                                            graph_config['edge_use_distance'] = True
                                            graph_config['edge_use_direction'] = False
                                            graph_config['edge_use_displacement'] = False
                                            logger.info("Inferred enabled_components=1: only distance enabled")
                                        elif enabled_comp == 2:
                                            # Common case: distance + direction (displacement disabled)
                                            graph_config['edge_use_distance'] = True
                                            graph_config['edge_use_direction'] = True
                                            graph_config['edge_use_displacement'] = False
                                            logger.info(
                                                "Inferred enabled_components=2: distance + direction enabled "
                                                "(displacement disabled)"
                                            )
                                        elif enabled_comp == 3:
                                            # All enabled
                                            graph_config['edge_use_distance'] = True
                                            graph_config['edge_use_direction'] = True
                                            graph_config['edge_use_displacement'] = True
                                            logger.info("Inferred enabled_components=3: all edge features enabled")
                    except Exception as exc:
                        logger.warning("Could not infer some parameters from checkpoint: %s", exc)

                    language_model = self.model.model if hasattr(self.model, 'model') else self.model
                    graph_encoder = TextAttributedGraphEncoderSpatial(
                        qwen_model=language_model,
                        tokenizer=self.template.tokenizer,
                        **graph_config,
                        training_phase="frozen",
                    )

                    graph_encoder.load_state_dict(graph_encoder_state, strict=False)
                    graph_encoder.eval()
                    graph_encoder.to(self.device)
                    self.model.add_module('graph_encoder', graph_encoder)
                    logger.info("✅ Loaded graph_encoder")
                # except Exception as e:
                #     logger.error(f"❌ Failed to load graph_encoder: {e}")

    def load_spatial_reasoning_dataset(self) -> List[Dict[str, Any]]:
        """Load spatial reasoning dataset"""
        data_path = self.args.spatial_reasoning_data_path
        logger.info(f"Loading dataset from {data_path}")

        # Validate dataset path
        if not data_path:
            raise ValueError("spatial_reasoning_data_path must be provided. Use --spatial-reasoning-data-path <path>")

        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"Dataset file not found: {data_path}\n"
                f"Please provide a valid path using --spatial-reasoning-data-path <path>"
            )

        if not data_path.endswith('.jsonl'):
            logger.warning(f"Dataset file does not have .jsonl extension: {data_path}")

        samples = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                sample = json.loads(line)
                if 'messages' in sample:
                    # Filter for image-only samples if requested
                    if self.args.image_only_mode or self.args.compare_stage1_stage2:
                        # Check if sample has no graph
                        graphs = sample.get('graphs', [])
                        if graphs and any(g for g in (graphs if isinstance(graphs, list) else [graphs]) if g):
                            continue  # Skip samples with graphs

                    samples.append(sample)

                    # Limit to eval_limit samples if specified
                    if self.args.eval_limit and len(samples) >= self.args.eval_limit:
                        logger.info(f"Limited to {self.args.eval_limit} samples as requested")
                        break

        if self.args.eval_limit and len(samples) < self.args.eval_limit:
            logger.warning(f"Requested {self.args.eval_limit} samples but only {len(samples)} available")

        if self.args.image_only_mode or self.args.compare_stage1_stage2:
            logger.info(f"Filtered to {len(samples)} image-only samples (no graphs)")

        return samples

    def load_graph_from_pickle(self, pkl_file: str) -> Optional[nx.Graph]:
        """Load NetworkX graph from pickle file"""
        if not os.path.exists(pkl_file):
            return None

        try:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)

            if isinstance(data, nx.Graph):
                return data
            elif isinstance(data, dict):
                if 'subgraph' in data:
                    return data['subgraph']
                elif 'subgraph_data' in data and 'subgraph' in data['subgraph_data']:
                    return data['subgraph_data']['subgraph']
        except Exception as e:
            logger.warning(f"Failed to load graph from {pkl_file}: {e}")

        return None


    # ------------------------------------------------------------------
    # Geo helpers for map plotting
    # ------------------------------------------------------------------
    def _infer_city_from_path(self, path: Optional[str]) -> Optional[str]:
        """Infer city name from a file path (graph path / dataset path).

        We use this only for locating a matching nodes.geojson fallback.
        """
        if not path:
            return None
        p = str(path).lower()
        # Common cities used in this project
        for city in [
            'beijing', 'singapore', 'newyork', 'new_york', 'paris', 'london',
            'tokyo', 'shanghai', 'hongkong', 'sanfrancisco', 'san_francisco'
        ]:
            if city in p:
                return city.replace('new_york', 'newyork').replace('san_francisco', 'sanfrancisco')
        return None

    def _get_nodes_geojson_path(self, city: Optional[str]) -> Optional[str]:
        """Return nodes.geojson path for a city, if available.

        Priority:
        1) args.nodes_geojson (explicit file)
        2) args.nodes_geojson_dir/<city>/nodes.geojson
        3) common default: UrbanKG/data/geo/SR/osm_data/<city>/nodes.geojson
        """
        # Explicit single file path
        nodes_geojson = getattr(self.args, 'nodes_geojson', None)
        if nodes_geojson and os.path.exists(nodes_geojson):
            return nodes_geojson

        # Directory containing per-city folders
        nodes_geojson_dir = getattr(self.args, 'nodes_geojson_dir', None)
        if city and nodes_geojson_dir:
            cand = os.path.join(nodes_geojson_dir, city, 'nodes.geojson')
            if os.path.exists(cand):
                return cand

        # Default path used in your repo
        if city:
            default = os.path.join('UrbanKG/data/geo/SR/osm_data', city, 'nodes.geojson')
            if os.path.exists(default):
                return default
        return None

    def _load_nodes_geojson(self, city: Optional[str]) -> Optional['gpd.GeoDataFrame']:
        """Load nodes.geojson as a GeoDataFrame indexed by integer 'id'. Cached."""
        if not city:
            return None
        #
        # if city in self._nodes_geojson_cache:
        #     return self._nodes_geojson_cache[city]

        geojson_path = self._get_nodes_geojson_path(city)
        if not geojson_path:
            self._nodes_geojson_cache[city] = None
            return None

        try:
            import geopandas as gpd  # type: ignore
            gdf = gpd.read_file(geojson_path)
            # Expect an 'id' column (or something equivalent). Make robust.
            id_col = None
            for cand in ['id', 'osmid', 'osm_id', 'node_id']:
                if cand in gdf.columns:
                    id_col = cand
                    break
            if id_col is None:
                logger.warning(f"[MapPlot] nodes.geojson has no id column: {geojson_path} (cols={list(gdf.columns)})")
                self._nodes_geojson_cache[city] = None
                return None

            gdf = gdf.copy()
            gdf[id_col] = gdf[id_col].astype('int64', errors='ignore')
            gdf = gdf.set_index(id_col, drop=False)
            self._nodes_geojson_cache[city] = gdf
            logger.info(f"[MapPlot] Loaded nodes.geojson for city={city}: {geojson_path} (n={len(gdf)})")
            return gdf
        except Exception as e:
            logger.warning(f"[MapPlot] Failed to load nodes.geojson for city={city}: {e}")
            self._nodes_geojson_cache[city] = None
            return None

    def _extract_node_id_from_text(self, node_text: str) -> Optional[int]:
        """Parse integer node ID from node_text like '... ID: 12554, ...'."""
        if not node_text:
            return None
        m = _NODE_ID_RE.search(node_text)
        if not m:
            return None
        try:
            return int(m.group(1))
        except Exception:
            return None

    def _get_node_lonlat(self, graph: nx.Graph, node_id: Any, *, city: Optional[str] = None,
                         graph_path: Optional[str] = None) -> Optional[Tuple[float, float]]:
        """Return (lon, lat) for node_id.

        1) attrs['coords'] if present
        2) parse from attrs['node_text'] -> 'Coordinates: (lon, lat)'
        3) fallback: parse ID from node_text and lookup in nodes.geojson
        """
        # if graph is None or not graph.has_node(node_id):
        #     return None
        # print('node_id: ',node_id)
        attrs = graph.nodes[node_id] or {}

        # 1) Preferred: explicit coords attribute
        coords = attrs.get('coords', None)
        # print('coords: ', coords)
        if coords is not None and isinstance(coords, (tuple, list)) and len(coords) >= 2:
            return (float(coords[0]), float(coords[1]))


        # 2) Fallback: parse from node_text
        node_text = attrs.get('node_text', '') or ''
        m = _COORDS_IN_TEXT_RE.search(node_text)
        if m:
            # try:
            lon = float(m.group(1))
            lat = float(m.group(2))
            return (lon, lat)


        # 3) GeoJSON lookup via node ID
        if len(str(node_id)) < 12:
            # nid_int = self._extract_node_id_from_text(node_text)
            # print('node_text: ', node_text)
            # print('ID: ', nid_int)

            geom = self.gdf[self.gdf['id']==node_id].geometry.iloc[0]

            # Use centroid for all geometry types:
            # Point, Polygon, MultiPolygon, LineString, MultiLineString, etc.
            c = geom.centroid
            print('geom.centroid: ', c)
            if hasattr(c, "x") and hasattr(c, "y"):
                return (float(c.x), float(c.y))
        #     except Exception:
        #         return None
        #
        # except Exception:
        #     return None

        return None

    def compute_node_distances(
            self,
            graph: nx.Graph,
            image_coords: Tuple[float, float],
            node_ids: List[str]
    ) -> Dict[str, float]:
        """
        Compute geodesic distance from image location to each graph node.

        Args:
            graph: NetworkX graph
            image_coords: (lon, lat) of image location
            node_ids: List of node IDs in graph

        Returns:
            Dictionary mapping node_id to distance in meters
        """
        distances = {}
        image_lat, image_lon = image_coords[1], image_coords[0]  # (lat, lon) for geodesic

        for node_id in node_ids:
            if graph.has_node(node_id):
                node_attrs = graph.nodes[node_id]
                coords = node_attrs.get('coords', None)

                if coords is not None and isinstance(coords, (tuple, list)) and len(coords) >= 2:
                    try:
                        node_lon, node_lat = float(coords[0]), float(coords[1])
                        distance_m = geodesic((image_lat, image_lon), (node_lat, node_lon)).meters
                        distances[node_id] = distance_m
                    except (ValueError, TypeError):
                        distances[node_id] = float('inf')
                else:
                    distances[node_id] = float('inf')
            else:
                distances[node_id] = float('inf')

        return distances

    def analyze_spatial_decay(
            self,
            graph_attention_scores: np.ndarray,
            node_ids: List[str],
            node_distances: Dict[str, float],
            distance_bins: List[float]
    ) -> Dict[str, Any]:
        """
        Analyze how attention scores decay with distance from image location.

        Args:
            graph_attention_scores: Per-node attention scores (num_nodes,)
            node_ids: List of node IDs corresponding to attention scores
            node_distances: Dictionary mapping node_id to distance in meters
            distance_bins: List of distance bin boundaries in meters

        Returns:
            Dictionary with spatial decay statistics
        """
        if len(graph_attention_scores) == 0 or len(node_ids) == 0:
            return {}

        # Group nodes by distance bins
        bin_attention = {i: [] for i in range(len(distance_bins) - 1)}
        bin_counts = {i: 0 for i in range(len(distance_bins) - 1)}

        for idx, node_id in enumerate(node_ids):
            if idx >= len(graph_attention_scores):
                continue

            distance = node_distances.get(node_id, float('inf'))
            attention = float(graph_attention_scores[idx])

            # Find which bin this node belongs to
            for bin_idx in range(len(distance_bins) - 1):
                if distance_bins[bin_idx] <= distance < distance_bins[bin_idx + 1]:
                    bin_attention[bin_idx].append(attention)
                    bin_counts[bin_idx] += 1
                    break

        # Compute statistics per bin
        bin_stats = {}
        for bin_idx in range(len(distance_bins) - 1):
            attentions = bin_attention[bin_idx]
            if attentions:
                bin_stats[f'bin_{bin_idx}'] = {
                    'distance_range': (distance_bins[bin_idx], distance_bins[bin_idx + 1]),
                    'mean_attention': float(np.mean(attentions)),
                    'std_attention': float(np.std(attentions)),
                    'max_attention': float(np.max(attentions)),
                    'min_attention': float(np.min(attentions)),
                    'count': len(attentions),
                }
            else:
                bin_stats[f'bin_{bin_idx}'] = {
                    'distance_range': (distance_bins[bin_idx], distance_bins[bin_idx + 1]),
                    'mean_attention': 0.0,
                    'std_attention': 0.0,
                    'max_attention': 0.0,
                    'min_attention': 0.0,
                    'count': 0,
                }

        # Compute correlation between distance and attention
        distances_list = []
        attentions_list = []
        for idx, node_id in enumerate(node_ids):
            if idx < len(graph_attention_scores):
                distance = node_distances.get(node_id, float('inf'))
                if distance != float('inf'):
                    distances_list.append(distance)
                    attentions_list.append(float(graph_attention_scores[idx]))

        correlation = 0.0
        if len(distances_list) > 1:
            correlation = float(np.corrcoef(distances_list, attentions_list)[0, 1])
            if np.isnan(correlation):
                correlation = 0.0

        return {
            'bin_statistics': bin_stats,
            'distance_attention_correlation': correlation,
            'total_nodes': len(node_ids),
            'nodes_with_valid_distance': len(distances_list),
        }

    def _identify_token_boundaries(self, inputs: Dict, has_graph: bool) -> Dict[str, Optional[Tuple[int, int]]]:
        """
        Robustly identify where vision, graph, and text tokens are in the sequence.

        Key idea:
        - For Qwen2/2.5-VL, the "image region" is typically represented by many <|image_pad|> tokens,
          optionally surrounded by <|vision_start|> ... <|vision_end|>.
        - Do NOT assume vision ends at <|graph_start|>. Instead, locate the contiguous <|image_pad|> block.
        """
        tokenizer = self.template.tokenizer

        def _tok_id(tok: str) -> Optional[int]:
            try:
                _id = tokenizer.convert_tokens_to_ids(tok)
                # Some tokenizers return unk id for unknown; treat that as missing
                if _id is None:
                    return None
                return int(_id)
            except Exception:
                return None

        # Graph special token IDs
        GRAPH_START_ID = _tok_id("<|graph_start|>")
        GRAPH_END_ID = _tok_id("<|graph_end|>")
        GRAPH_PAD_ID = _tok_id("<|graph_pad|>")

        # Vision/image special token IDs (Qwen2/2.5-VL commonly uses these)
        VISION_START_ID = _tok_id("<|vision_start|>")
        VISION_END_ID = _tok_id("<|vision_end|>")
        IMAGE_PAD_ID = _tok_id("<|image_pad|>")

        input_ids = inputs["input_ids"]
        if input_ids.dim() > 1:
            input_ids = input_ids[0]
        input_ids_list = input_ids.detach().cpu().tolist()
        seq_len = len(input_ids_list)

        # -------------------------
        # 1) Vision range (robust)
        # -------------------------
        vision_range = None

        # (A) Preferred: use explicit <|vision_start|> ... <|vision_end|> if present
        if VISION_START_ID is not None and VISION_END_ID is not None:
            try:
                vs = input_ids_list.index(VISION_START_ID)
                ve = input_ids_list.index(VISION_END_ID, vs + 1)
                # Inside this region, the actual image "tokens" are typically <|image_pad|>.
                if IMAGE_PAD_ID is not None:
                    pads = [i for i in range(vs + 1, ve) if input_ids_list[i] == IMAGE_PAD_ID]
                    if pads:
                        vision_range = (pads[0], pads[-1] + 1)
                    else:
                        # fallback: take the whole region excluding delimiters
                        if vs + 1 < ve:
                            vision_range = (vs + 1, ve)
                else:
                    if vs + 1 < ve:
                        vision_range = (vs + 1, ve)
            except ValueError:
                pass

        # (B) Fallback: find the first contiguous <|image_pad|> block anywhere in the sequence
        if vision_range is None and IMAGE_PAD_ID is not None:
            pad_positions = [i for i, tid in enumerate(input_ids_list) if tid == IMAGE_PAD_ID]
            if pad_positions:
                # find the first contiguous block of image_pad tokens
                start = pad_positions[0]
                end = start + 1
                while end < seq_len and input_ids_list[end] == IMAGE_PAD_ID:
                    end += 1
                vision_range = (start, end)

                # If there are multiple pad blocks (multi-image or template quirks),
                # prefer the *largest* contiguous block (most likely the real image grid).
                # This helps when a small pad run exists elsewhere.
                blocks = []
                i = 0
                while i < len(pad_positions):
                    s = pad_positions[i]
                    e = s + 1
                    while e < seq_len and input_ids_list[e] == IMAGE_PAD_ID:
                        e += 1
                    blocks.append((s, e))
                    # jump to next non-pad
                    while i < len(pad_positions) and pad_positions[i] < e:
                        i += 1
                if blocks:
                    # pick the largest block by length
                    vision_range = max(blocks, key=lambda x: x[1] - x[0])

        # (C) Very last fallback: if pixel_values exists and we found nothing, keep your old coarse behavior
        if vision_range is None and "pixel_values" in inputs and seq_len > 1:
            vision_range = (1, seq_len)

        # -------------------------
        # 2) Graph range (as before)
        # -------------------------
        graph_range = None
        if has_graph and GRAPH_START_ID is not None and GRAPH_END_ID is not None and GRAPH_PAD_ID is not None:
            try:
                gs = input_ids_list.index(GRAPH_START_ID)
                ge = input_ids_list.index(GRAPH_END_ID, gs + 1)
                graph_pad_indices = [i for i in range(gs + 1, ge) if input_ids_list[i] == GRAPH_PAD_ID]
                if graph_pad_indices:
                    graph_range = (min(graph_pad_indices), max(graph_pad_indices) + 1)
                else:
                    # fallback: take region between start/end
                    if gs + 1 < ge:
                        graph_range = (gs + 1, ge)
            except ValueError:
                pass

        # -------------------------
        # 3) Prompt/text range
        # -------------------------
        prompt_start = 1

        # If we have explicit end delimiter for vision, prompt should start after it.
        if VISION_END_ID is not None and VISION_END_ID in input_ids_list:
            try:
                ve = input_ids_list.index(VISION_END_ID)
                prompt_start = max(prompt_start, ve + 1)
            except ValueError:
                pass

        # Otherwise, start after the vision pad block we detected
        if vision_range is not None:
            prompt_start = max(prompt_start, vision_range[1])

        # And if graph exists, start after graph end marker or graph_range
        if graph_range is not None:
            prompt_start = max(prompt_start, graph_range[1])
        elif has_graph and GRAPH_END_ID is not None and GRAPH_END_ID in input_ids_list:
            try:
                ge = input_ids_list.index(GRAPH_END_ID)
                prompt_start = max(prompt_start, ge + 1)
            except ValueError:
                pass

        prompt_start = min(prompt_start, seq_len)
        prompt_range = (prompt_start, seq_len)

        return {
            "vision_range": vision_range,
            "graph_range": graph_range,
            "prompt_range": prompt_range,
        }

    def analyze_sample_with_tam(
            self,
            sample: Dict[str, Any],
            sample_idx: int
    ) -> Dict[str, Any]:
        """
        Analyze a single sample using TAM with spatial decay analysis.

        Returns:
            Dictionary containing attention scores, spatial decay stats, etc.
        """
        # Get image path
        img_path = sample.get('images', '')
        if isinstance(img_path, list):
            img_path = img_path[0] if img_path else ''

        if not os.path.isabs(img_path):
            img_path = os.path.join(self.args.image_root, os.path.basename(img_path))

        if not os.path.exists(img_path):
            logger.warning(f"Image not found: {img_path}")
            return {}

        # Get graph path
        graph_path = sample.get('graphs', None)
        if isinstance(graph_path, list):
            graph_path = graph_path[0] if graph_path else None

        has_graph = graph_path is not None and os.path.exists(graph_path) if graph_path else False

        # Get image coordinates
        image_coords = sample.get('image_coords', None)
        if image_coords is None:
            # Try alternative field names
            image_coords = sample.get('coordinates', None)
        if image_coords and isinstance(image_coords, list) and len(image_coords) >= 2:
            # Ensure format is (lon, lat)
            image_coords = (float(image_coords[0]), float(image_coords[1]))

        # Prepare input
        encode_input = {
            'messages': sample['messages'],
            'images': [img_path]
        }

        if has_graph:
            encode_input['graphs'] = [graph_path]

        # Encode input
        inputs = self.template.encode(encode_input)

        # Convert to tensors
        tensor_keys = {'input_ids', 'attention_mask', 'labels', 'position_ids',
                       'image_grid_thw', 'video_grid_thw', 'pixel_values', 'image_sizes'}

        for key, value in inputs.items():
            if key in tensor_keys and isinstance(value, list):
                inputs[key] = torch.tensor(value, device=self.device)
            elif isinstance(value, torch.Tensor):
                inputs[key] = value.to(self.device)

        # Get token boundaries BEFORE generation
        token_boundaries = self._identify_token_boundaries(inputs, has_graph)
        # 🔍 DEBUG: verify extracted token spans
        if token_boundaries["vision_range"] is not None:
            vs, ve = token_boundaries["vision_range"]
            logger.info(
                f"[SpanCheck][sample={sample_idx}] "
                f"vision_range=({vs},{ve}) len={ve - vs}"
            )
        else:
            logger.warning(f"[SpanCheck][sample={sample_idx}] vision_range=None")

        if token_boundaries["graph_range"] is not None:
            gs, ge = token_boundaries["graph_range"]
            logger.info(
                f"[SpanCheck][sample={sample_idx}] "
                f"graph_range=({gs},{ge}) len={ge - gs}"
            )
        else:
            logger.info(f"[SpanCheck][sample={sample_idx}] graph_range=None")
        # Load graph for spatial analysis
        graph = None
        node_ids = []
        node_distances = {}
        if has_graph and graph_path:
            graph = self.load_graph_from_pickle(graph_path)
            if graph is not None:
                # Get node IDs in order (matching graph encoder output order)
                # The graph encoder processes nodes in the order they appear in the graph
                # We need to match this order for proper attention-to-node mapping
                node_ids = list(graph.nodes())
                if image_coords:
                    node_distances = self.compute_node_distances(graph, image_coords, node_ids)

        # Embedding-style forward pass with hidden states tracking
        with torch.no_grad():
            encoded_inputs = self.template._post_encode(self.model, inputs)

            # Ensure batch dimension exists for Qwen-style models
            for key in ('input_ids', 'attention_mask', 'position_ids'):
                if key in encoded_inputs and isinstance(encoded_inputs[key], torch.Tensor):
                    if encoded_inputs[key].dim() == 1:
                        encoded_inputs[key] = encoded_inputs[key].unsqueeze(0)

            # Ensure attention_mask exists (required by embedding hook in patcher.py)
            if 'attention_mask' not in encoded_inputs or encoded_inputs['attention_mask'] is None:
                if 'input_ids' in encoded_inputs:
                    input_ids = encoded_inputs['input_ids']
                    if input_ids.dim() == 1:
                        input_ids = input_ids.unsqueeze(0)
                    encoded_inputs['attention_mask'] = torch.ones_like(input_ids, dtype=torch.long)
                else:
                    raise ValueError("Cannot create attention_mask: input_ids not found in encoded_inputs")

            # Forward pass to get last-layer hidden states for the query
            # Use base model if available (for models with LM head wrapper)
            model_to_use = self.model
            using_base_model = False
            if hasattr(self.model, 'model'):
                # This is likely a model with LM head, use the base model
                model_to_use = self.model.model
                using_base_model = True
            elif hasattr(self.model, 'get_model'):
                # Some wrappers use get_model()
                model_to_use = self.model.get_model()
                using_base_model = True

            # IMPORTANT: For TAM, we need ALL token hidden states, not just the last token
            # The embedding hook in patcher.py intercepts forward() and returns only last token
            # We need to temporarily disable hooks to get full sequence hidden states

            # Store and temporarily remove forward hooks
            original_hooks = {}
            hooks_removed = False

            # Check both model_to_use and its submodules for hooks
            models_to_check = [model_to_use]
            if hasattr(model_to_use, 'model'):
                models_to_check.append(model_to_use.model)

            for model in models_to_check:
                if hasattr(model, '_forward_hooks') and len(model._forward_hooks) > 0:
                    original_hooks[id(model)] = model._forward_hooks.copy()
                    model._forward_hooks.clear()
                    hooks_removed = True
                    logger.debug(
                        f"Temporarily removed {len(original_hooks[id(model)])} forward hooks from model for TAM")

            try:
                # Filter out image-related parameters when using base model
                # The base model (Qwen2_5_VLModel) doesn't accept pixel_values, image_grid_thw, etc.
                # These are only accepted by the full model (Qwen2_5_VLForConditionalGeneration)
                if using_base_model:
                    base_model_params = {
                        'input_ids', 'attention_mask', 'position_ids', 'past_key_values',
                        'inputs_embeds', 'use_cache', 'output_attentions', 'output_hidden_states',
                        'return_dict', 'cache_position'
                    }
                    filtered_inputs = {k: v for k, v in encoded_inputs.items() if k in base_model_params}
                else:
                    filtered_inputs = encoded_inputs

                outputs = model_to_use(
                    **filtered_inputs,
                    output_hidden_states=True,
                    return_dict=True,
                )
            finally:
                # Restore hooks
                if hooks_removed:
                    for model_id, hooks in original_hooks.items():
                        # Find the model by id (this is a bit hacky but works)
                        for model in models_to_check:
                            if id(model) == model_id:
                                model._forward_hooks.update(hooks)
                                logger.debug(f"Restored {len(hooks)} forward hooks")
                                break

        # Last layer hidden states: (batch=1, seq_len, hidden_dim)
        # outputs is a dict when return_dict=True
        if isinstance(outputs, dict):
            hidden_states = outputs.get('hidden_states', None)
            # If hidden_states not found, try to get last_hidden_state and use that
            if hidden_states is None:
                last_hidden_state = outputs.get('last_hidden_state', None)
                if last_hidden_state is not None:
                    # Use last_hidden_state as a single-layer hidden_states
                    hidden_states = (last_hidden_state,)
                else:
                    # Debug: print what keys are available
                    logger.warning(f"Model output keys: {list(outputs.keys())}")
                    raise ValueError(
                        f"Model did not return hidden_states. Available keys: {list(outputs.keys())}. "
                        "Try accessing the base model directly or check model configuration."
                    )
        else:
            # Try attribute access
            hidden_states = getattr(outputs, 'hidden_states', None)
            if hidden_states is None:
                last_hidden_state = getattr(outputs, 'last_hidden_state', None)
                if last_hidden_state is not None:
                    hidden_states = (last_hidden_state,)
                else:
                    raise ValueError("Model did not return hidden_states as attribute.")

        if hidden_states is None or len(hidden_states) == 0:
            raise ValueError("hidden_states is empty or None.")

        # Extract last layer hidden states
        last_layer_hidden = hidden_states[-1]  # Should be (batch, seq_len, hidden_dim) or (seq_len, hidden_dim)

        # Debug: log shapes
        logger.info(f"Sample {sample_idx}: hidden_states tuple length: {len(hidden_states)}, "
                    f"last_layer_hidden shape: {last_layer_hidden.shape}, dim: {last_layer_hidden.dim()}")

        # Handle different shapes
        if last_layer_hidden.dim() == 3:
            # (batch, seq_len, hidden_dim) -> (seq_len, hidden_dim)
            last_hidden = last_layer_hidden[0]
            logger.info(f"Sample {sample_idx}: Extracted last_hidden shape: {last_hidden.shape} from 3D tensor")
        elif last_layer_hidden.dim() == 2:
            # Already (seq_len, hidden_dim)
            last_hidden = last_layer_hidden
            logger.info(f"Sample {sample_idx}: Using last_hidden directly, shape: {last_hidden.shape}")
        else:
            raise ValueError(
                f"Unexpected last_layer_hidden shape: {last_layer_hidden.shape}, "
                f"expected 2D (seq_len, hidden_dim) or 3D (batch, seq_len, hidden_dim)"
            )

        # Also check if we can use last_hidden_state directly (it might have the full sequence)
        if isinstance(outputs, dict):
            last_hidden_state = outputs.get('last_hidden_state', None)
            if last_hidden_state is not None:
                logger.info(f"Sample {sample_idx}: Also found last_hidden_state with shape: {last_hidden_state.shape}")
                # Use last_hidden_state if it has more tokens
                if last_hidden_state.dim() == 3 and last_hidden_state.shape[1] > last_hidden.shape[0]:
                    last_hidden = last_hidden_state[0]
                    logger.info(f"Sample {sample_idx}: Using last_hidden_state instead, new shape: {last_hidden.shape}")
                elif last_hidden_state.dim() == 2 and last_hidden_state.shape[0] > last_hidden.shape[0]:
                    last_hidden = last_hidden_state
                    logger.info(f"Sample {sample_idx}: Using last_hidden_state instead, new shape: {last_hidden.shape}")

        # Build a simple pooled query embedding from token states
        attn_mask = encoded_inputs.get('attention_mask', None)
        if attn_mask is not None:
            if attn_mask.dim() > 1:
                mask = attn_mask[0].unsqueeze(-1).to(last_hidden.device).to(last_hidden.dtype)  # (seq_len, 1)
            else:
                mask = attn_mask.unsqueeze(-1).to(last_hidden.device).to(last_hidden.dtype)
            masked_hidden = last_hidden * mask
            denom = mask.sum() + 1e-8
            query_embedding = masked_hidden.sum(dim=0) / denom
        else:
            query_embedding = last_hidden.mean(dim=0)

        # Build candidate embeddings
        candidates = sample.get('candidates', None)
        ground_truth_text = sample.get('ground_truth', '')
        candidate_embeddings: List[torch.Tensor] = []

        if candidates is None or len(candidates) == 0:
            # Fallback: treat ground truth as the only "candidate"
            if ground_truth_text:
                candidates = [ground_truth_text]
            else:
                candidates = []

        for cand_text in candidates:
            cand_encode_input = {
                'messages': [{'role': 'user', 'content': cand_text}],
            }
            cand_inputs = self.template.encode(cand_encode_input)

            # Move to device
            for key, value in list(cand_inputs.items()):
                if isinstance(value, list):
                    cand_inputs[key] = torch.tensor(value, device=self.device)
                elif isinstance(value, torch.Tensor):
                    cand_inputs[key] = value.to(self.device)

            with torch.no_grad():
                cand_encoded = self.template._post_encode(self.model, cand_inputs)

                # Ensure batch dimension exists
                for key in ('input_ids', 'attention_mask', 'position_ids'):
                    if key in cand_encoded and isinstance(cand_encoded[key], torch.Tensor):
                        if cand_encoded[key].dim() == 1:
                            cand_encoded[key] = cand_encoded[key].unsqueeze(0)

                # Ensure attention_mask exists (required by embedding hook in patcher.py)
                if 'attention_mask' not in cand_encoded or cand_encoded['attention_mask'] is None:
                    if 'input_ids' in cand_encoded:
                        cand_input_ids = cand_encoded['input_ids']
                        if cand_input_ids.dim() == 1:
                            cand_input_ids = cand_input_ids.unsqueeze(0)
                        cand_encoded['attention_mask'] = torch.ones_like(cand_input_ids, dtype=torch.long)
                    else:
                        raise ValueError("Cannot create attention_mask: input_ids not found in cand_encoded")

                # Temporarily remove hooks for candidate encoding too
                cand_original_hooks = {}
                cand_hooks_removed = False
                cand_models_to_check = [model_to_use]
                if hasattr(model_to_use, 'model'):
                    cand_models_to_check.append(model_to_use.model)

                for model in cand_models_to_check:
                    if hasattr(model, '_forward_hooks') and len(model._forward_hooks) > 0:
                        cand_original_hooks[id(model)] = model._forward_hooks.copy()
                        model._forward_hooks.clear()
                        cand_hooks_removed = True

                try:
                    # Use same model component as for query
                    # Filter out image-related parameters when using base model
                    if using_base_model:
                        base_model_params = {
                            'input_ids', 'attention_mask', 'position_ids', 'past_key_values',
                            'inputs_embeds', 'use_cache', 'output_attentions', 'output_hidden_states',
                            'return_dict', 'cache_position'
                        }
                        cand_filtered = {k: v for k, v in cand_encoded.items() if k in base_model_params}
                    else:
                        cand_filtered = cand_encoded

                    cand_outputs = model_to_use(
                        **cand_filtered,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                finally:
                    # Restore hooks
                    if cand_hooks_removed:
                        for model_id, hooks in cand_original_hooks.items():
                            for model in cand_models_to_check:
                                if id(model) == model_id:
                                    model._forward_hooks.update(hooks)
                                    break

            if isinstance(cand_outputs, dict):
                cand_hidden_states = cand_outputs.get('hidden_states', None)
                if cand_hidden_states is None:
                    cand_last_hidden_state = cand_outputs.get('last_hidden_state', None)
                    if cand_last_hidden_state is not None:
                        cand_hidden_states = (cand_last_hidden_state,)
                    else:
                        raise ValueError(
                            f"Model did not return hidden_states for candidate. Keys: {list(cand_outputs.keys())}")
            else:
                cand_hidden_states = getattr(cand_outputs, 'hidden_states', None)
                if cand_hidden_states is None:
                    cand_last_hidden_state = getattr(cand_outputs, 'last_hidden_state', None)
                    if cand_last_hidden_state is not None:
                        cand_hidden_states = (cand_last_hidden_state,)
                    else:
                        raise ValueError("Model did not return hidden_states for candidate.")

            if cand_hidden_states is None or len(cand_hidden_states) == 0:
                raise ValueError("cand_hidden_states is empty or None.")

            # Extract last layer hidden states
            cand_last_layer = cand_hidden_states[-1]
            if cand_last_layer.dim() == 3:
                cand_last_hidden = cand_last_layer[0]  # (batch, seq_len, hidden_dim) -> (seq_len, hidden_dim)
            elif cand_last_layer.dim() == 2:
                cand_last_hidden = cand_last_layer  # Already (seq_len, hidden_dim)
            else:
                raise ValueError(f"Unexpected cand_last_layer shape: {cand_last_layer.shape}")

            cand_attn_mask = cand_encoded.get('attention_mask', None)
            if cand_attn_mask is not None:
                if cand_attn_mask.dim() > 1:
                    c_mask = cand_attn_mask[0].unsqueeze(-1).to(cand_last_hidden.device).to(cand_last_hidden.dtype)
                else:
                    c_mask = cand_attn_mask.unsqueeze(-1).to(cand_last_hidden.device).to(cand_last_hidden.dtype)
                c_masked = cand_last_hidden * c_mask
                c_denom = c_mask.sum() + 1e-8
                cand_emb = c_masked.sum(dim=0) / c_denom
            else:
                cand_emb = cand_last_hidden.mean(dim=0)

            candidate_embeddings.append(cand_emb)

        if not candidate_embeddings:
            logger.warning(f'No candidates available for sample {sample_idx}, skipping TAM analysis.')
            return {}

        # Choose which candidate to analyze against: prefer ground-truth if known
        gt_idx = sample.get('ground_truth_idx', None)
        if gt_idx is None and ground_truth_text and candidates:
            try:
                gt_idx = candidates.index(ground_truth_text)
            except ValueError:
                gt_idx = 0
        if gt_idx is None or not (0 <= int(gt_idx) < len(candidate_embeddings)):
            gt_idx = 0
        gt_idx = int(gt_idx)

        target_candidate_embedding = candidate_embeddings[gt_idx]

        # Optionally compute predicted candidate via cosine similarity
        cand_stack = torch.stack(candidate_embeddings, dim=0)  # (num_cands, hidden_dim)
        with torch.no_grad():
            q_norm = F.normalize(query_embedding.view(1, -1), dim=-1)  # (1, dim)
            c_norm = F.normalize(cand_stack, dim=-1)  # (num_cands, dim)
            sims = torch.matmul(c_norm, q_norm.t()).squeeze(-1)  # (num_cands,)
        pred_idx = int(torch.argmax(sims).item())

        # "Generated" text for reporting: top-ranked candidate
        generated_text = candidates[pred_idx] if candidates else ''

        # Compute TAM activation scores based on cosine similarity
        vision_scores = None
        graph_scores = None

        # Debug: check actual sequence length
        actual_seq_len = last_hidden.shape[0]
        logger.info(f"Sample {sample_idx}: actual_seq_len={actual_seq_len}, token_boundaries={token_boundaries}, "
                    f"has_graph={has_graph}, target_candidate_idx={gt_idx}")

        if token_boundaries['vision_range'] is not None:
            v_start, v_end = token_boundaries['vision_range']
            # Clamp to actual sequence length
            v_start = min(v_start, actual_seq_len)
            v_end = min(v_end, actual_seq_len)
            if v_start < v_end:
                vision_scores = self.tam_computer.compute_activation_scores_from_cosine(
                    hidden_states=last_hidden,
                    candidate_embedding=target_candidate_embedding,
                    token_range=(v_start, v_end),
                )
                if vision_scores.size > 0:
                    if self.tam_computer.use_causal_inference:
                        vision_scores = self.tam_computer.apply_rank_gaussian_filter(vision_scores)
                    vision_contribution = self.tam_computer.aggregate_scores(vision_scores, method='mean')
                else:
                    vision_contribution = np.array([])
                    logger.warning(f"Sample {sample_idx}: vision_scores is empty after computation")
            else:
                vision_contribution = np.array([])
                logger.warning(
                    f"Sample {sample_idx}: vision_range [{v_start}, {v_end}) is invalid for seq_len {actual_seq_len}")
        else:
            vision_contribution = np.array([])

        if token_boundaries['graph_range'] is not None and has_graph:
            g_start, g_end = token_boundaries['graph_range']
            # Clamp to actual sequence length
            g_start = min(g_start, actual_seq_len)
            g_end = min(g_end, actual_seq_len)
            if g_start < g_end:
                graph_scores = self.tam_computer.compute_activation_scores_from_cosine(
                    hidden_states=last_hidden,
                    candidate_embedding=target_candidate_embedding,
                    token_range=(g_start, g_end),
                )
                if graph_scores.size > 0:
                    if self.tam_computer.use_causal_inference:
                        graph_scores = self.tam_computer.apply_rank_gaussian_filter(graph_scores)
                    graph_contribution = self.tam_computer.aggregate_scores(graph_scores, method='mean')
                else:
                    graph_contribution = np.array([])
                    logger.warning(f"Sample {sample_idx}: graph_scores is empty after computation")
            else:
                graph_contribution = np.array([])
                logger.warning(
                    f"Sample {sample_idx}: graph_range [{g_start}, {g_end}) is invalid for seq_len {actual_seq_len}")
        else:
            graph_contribution = np.array([])

        # Compute summary statistics
        vision_total = float(np.sum(vision_contribution)) if vision_contribution.size > 0 else 0.0
        graph_total = float(np.sum(graph_contribution)) if graph_contribution.size > 0 else 0.0
        total = vision_total + graph_total

        vision_ratio = vision_total / total if total > 0 else 0.0
        graph_ratio = graph_total / total if total > 0 else 0.0

        # Spatial decay analysis
        spatial_decay_stats = {}
        if (self.args.compute_spatial_decay and has_graph and graph is not None
                and len(graph_contribution) > 0 and image_coords):
            # Match graph node order with attention scores
            # The graph encoder processes nodes in the order they appear in the PyG Data object
            # which should match the NetworkX graph node order
            # However, the graph encoder may filter or reorder nodes, so we need to be careful

            # For now, we'll match by index assuming the order is preserved
            # In practice, the graph encoder should preserve node order from the graph
            matched_node_ids = None
            if len(node_ids) >= len(graph_contribution):
                # Truncate node_ids to match graph_contribution length
                # This handles cases where graph encoder filters some nodes
                matched_node_ids = node_ids[:len(graph_contribution)]
            elif len(node_ids) < len(graph_contribution):
                # Graph encoder may have added padding or repeated nodes
                # Use available nodes and pad with 'unknown' for extra attention scores
                matched_node_ids = node_ids + ['unknown'] * (len(graph_contribution) - len(node_ids))
            else:
                matched_node_ids = node_ids

            spatial_decay_stats = self.analyze_spatial_decay(
                graph_contribution,
                matched_node_ids,
                node_distances,
                self.args.distance_bins
            )

        # Get node names/attributes for visualization
        node_names = []
        if has_graph and graph is not None and len(node_ids) > 0:
            # Match node_ids with graph_contribution length
            matched_node_ids = node_ids[:len(graph_contribution)] if len(node_ids) >= len(
                graph_contribution) else node_ids + ['unknown'] * (len(graph_contribution) - len(node_ids))

            for node_id in matched_node_ids:
                if node_id == 'unknown':
                    node_names.append('unknown')
                elif graph.has_node(node_id):
                    node_attrs = graph.nodes[node_id]
                    # Try to get node name from various possible attributes
                    node_name = (
                            node_attrs.get('node_text', None) or
                            node_attrs.get('name', None) or
                            node_attrs.get('label', None) or
                            str(node_id)
                    )
                    # Truncate long names
                    if isinstance(node_name, str) and len(node_name) > 50:
                        node_name = node_name[:47] + '...'
                    node_names.append(node_name if node_name else str(node_id))
                else:
                    node_names.append(str(node_id))

        result = {
            'sample_idx': sample_idx,
            'query_type': sample.get('query_type', sample.get('question_type', 'unknown')),
            'generated_text': generated_text,
            'ground_truth': sample.get('ground_truth', ''),
            'is_correct': generated_text.strip().lower() == sample.get('ground_truth', '').strip().lower(),
            'vision_contribution': vision_contribution.tolist() if vision_contribution.size > 0 else [],
            'graph_contribution': graph_contribution.tolist() if graph_contribution.size > 0 else [],
            'vision_total_score': vision_total,
            'graph_total_score': graph_total,
            'vision_ratio': vision_ratio,
            'graph_ratio': graph_ratio,
            'has_graph': has_graph,
            'token_boundaries': token_boundaries,
            'image_path': img_path,
            'image_coords': image_coords,
            'spatial_decay_stats': spatial_decay_stats,
            'matched_node_ids': matched_node_ids,
            'node_names': node_names,
        }

        return result

    def visualize_top_samples(self, tam_results: List[Dict[str, Any]], samples: List[Dict], k: int = 10):
        """Visualize top-k samples with highest graph contributions"""
        if k <= 0:
            return

        # Sort by graph contribution (or vision contribution if you prefer)
        sorted_results = sorted(
            tam_results,
            key=lambda x: x.get('graph_total_score', 0.0),
            reverse=True
        )

        top_k_results = sorted_results[:k]

        logger.info(f"Visualizing top {k} samples with highest graph contributions...")

        for rank, result in enumerate(top_k_results):
            sample_idx = result['sample_idx']
            if sample_idx < len(samples):
                sample = samples[sample_idx]
            else:
                logger.warning(f"Sample index {sample_idx} out of range, skipping visualization")
                continue

            # try:
            # Create visualization
            fig = self._create_tam_visualization(result, sample)

            # Save
            save_path = os.path.join(
                self.args.tam_output_dir,
                f"tam_rank{rank + 1}_sample{sample_idx}.png"
            )
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)

            logger.info(f"Saved visualization: {save_path}")
            #
            # Also save a map view of top-k activated graph nodes (if available).
            # try:
            graph_path = sample.get('graphs', None)
            if isinstance(graph_path, list):
                graph_path = graph_path[0] if graph_path else None

            # Recover image coords (lon, lat) if present
            image_coords = sample.get('image_coords', None) or sample.get('coordinates', None)
            if image_coords and isinstance(image_coords, list) and len(image_coords) >= 2:
                image_coords = (float(image_coords[0]), float(image_coords[1]))


            node_scores = result.get('graph_contribution', None)
            matched_node_ids =  result.get('matched_node_ids', None)

            if graph_path and os.path.exists(graph_path):
                nx_graph = self.load_graph_from_pickle(graph_path)
                # if nx_graph is not None:
                # print('nx_graph: ', nx_graph)
                map_path = os.path.join(
                    self.args.tam_output_dir,
                    f"map_rank{rank + 1}_sample{sample_idx}_top20.png"
                )
                self.plot_topk_activated_nodes_on_map(
                    graph=nx_graph,
                    image_coords=image_coords,
                    node_ids=matched_node_ids,
                    node_scores=node_scores,
                    out_path=map_path,
                    k=50,
                    use_basemap=True,
                )
            #     except Exception as e:
            #     except Exception as e:
            #         logger.debug(f"[MapPlot] Skipped map plotting for sample {sample_idx}: {e}")
            # except Exception as e:
            #     logger.error(f"Error creating visualization for sample {sample_idx}: {e}")
            #     import traceback
            #     traceback.print_exc()
            #     continue

    def _create_tam_visualization(self, result: Dict[str, Any], sample: Dict) -> plt.Figure:
        """Create a simplified TAM visualization for spatial reasoning"""
        fig = plt.figure(figsize=(20, 12))

        # New Layout: 2 rows, 3 columns
        # Row 1: [Input Image] [Vision Token Activation Heatmap] [Graph Token Activation Heatmap]
        # Row 2: [Vision Activation Overlay] [Question & Results] [Top Activated Graph Nodes]
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3, left=0.03, right=0.97, top=0.95, bottom=0.05,
                              width_ratios=[1.2, 1, 1])

        # Load image
        img_path = result.get('image_path', '')
        if img_path and os.path.exists(img_path):
            try:
                img = Image.open(img_path).convert('RGB')
            except Exception as e:
                logger.warning(f"Could not load image {img_path}: {e}")
                img = None
        else:
            img = None

        # 1. Original Image
        ax1 = fig.add_subplot(gs[0, 0])
        if img is not None:
            ax1.imshow(img)
            ax1.set_title("Input Image", fontsize=12, fontweight='bold')
        else:
            ax1.text(0.5, 0.5, "Image Not Available", ha='center', va='center', fontsize=14)
            ax1.set_title("Input Image", fontsize=12, fontweight='bold')
        ax1.axis('off')

        vision_contrib = np.array(result.get('vision_contribution', []))
        graph_contrib = np.array(result.get('graph_contribution', []))
        has_graph = result.get('has_graph', False)

        # 2. Vision Token Activation Heatmap (separate visualization)
        ax2 = fig.add_subplot(gs[0, 1])
        if vision_contrib.size > 0:
            # Reshape vision tokens to 2D grid
            side_len = int(np.sqrt(len(vision_contrib)))
            if side_len * side_len < len(vision_contrib):
                side_len += 1

            padded = np.zeros(side_len * side_len)
            padded[:len(vision_contrib)] = vision_contrib
            vision_map_2d = padded.reshape(side_len, side_len)

            im2 = ax2.imshow(vision_map_2d, cmap='hot', interpolation='bilinear', aspect='auto')
            vision_total = result.get('vision_total_score', 0)
            ax2.set_title(f"Vision Token Activation\n(Total: {vision_total:.3f}, {len(vision_contrib)} tokens)",
                          fontsize=12, fontweight='bold')
            ax2.set_xlabel("Token Position (2D Grid X)", fontsize=10)
            ax2.set_ylabel("Token Position (2D Grid Y)", fontsize=10)
            plt.colorbar(im2, ax=ax2, fraction=0.046, label='Activation Score')
            ax2.tick_params(labelsize=8)
        else:
            ax2.text(0.5, 0.5, "No Vision Tokens Available", ha='center', va='center', fontsize=14)
            ax2.set_title("Vision Token Activation", fontsize=12, fontweight='bold')
            ax2.axis('off')

        # 3. Graph Token Activation Heatmap (separate visualization)
        ax2b = fig.add_subplot(gs[0, 2])
        if has_graph and graph_contrib.size > 0:
            # Reshape graph tokens to 2D grid
            graph_count = len(graph_contrib)
            side_len = int(np.sqrt(graph_count))
            if side_len * side_len < graph_count:
                side_len += 1

            padded = np.zeros(side_len * side_len)
            padded[:graph_count] = graph_contrib
            graph_map_2d = padded.reshape(side_len, side_len)

            im2b = ax2b.imshow(graph_map_2d, cmap='hot', interpolation='bilinear', aspect='auto')
            graph_total = result.get('graph_total_score', 0)
            ax2b.set_title(f"Graph Token Activation\n(Total: {graph_total:.3f}, {graph_count} tokens)",
                           fontsize=12, fontweight='bold')
            ax2b.set_xlabel("Token Position (2D Grid X)", fontsize=10)
            ax2b.set_ylabel("Token Position (2D Grid Y)", fontsize=10)
            plt.colorbar(im2b, ax=ax2b, fraction=0.046, label='Activation Score')
            ax2b.tick_params(labelsize=8)
        else:
            ax2b.text(0.5, 0.5, "No Graph Tokens Available", ha='center', va='center', fontsize=14)
            ax2b.set_title("Graph Token Activation", fontsize=12, fontweight='bold')
            ax2b.axis('off')

        # 3. Vision Overlay on Image (if image available)
        ax3 = fig.add_subplot(gs[1, 0])
        if img is not None and vision_contrib.size > 0:
            # Resize vision map to image size
            img_array = np.array(img)
            h, w = img_array.shape[:2]

            # Reshape vision scores to approximate grid
            side_len = int(np.sqrt(len(vision_contrib)))
            if side_len * side_len < len(vision_contrib):
                side_len += 1
            padded = np.zeros(side_len * side_len)
            padded[:len(vision_contrib)] = vision_contrib
            vision_map_2d = padded.reshape(side_len, side_len)

            # Normalize and resize using PIL
            vision_map_2d = (vision_map_2d - vision_map_2d.min()) / (vision_map_2d.max() - vision_map_2d.min() + 1e-8)
            # Convert to uint8 for PIL
            vision_map_uint8 = (vision_map_2d * 255).astype(np.uint8)
            vision_map_pil = Image.fromarray(vision_map_uint8, mode='L')
            vision_map_resized = np.array(vision_map_pil.resize((w, h), Image.BILINEAR)) / 255.0

            # Create heatmap overlay
            import matplotlib.cm as cm
            heatmap = cm.hot(vision_map_resized)[:, :, :3]
            overlay = (0.5 * img_array / 255.0 + 0.5 * heatmap)

            ax3.imshow(overlay)
            ax3.set_title("Vision Activation Overlay", fontsize=12, fontweight='bold')
        else:
            ax3.text(0.5, 0.5, "No Overlay Available", ha='center', va='center', fontsize=14)
            ax3.set_title("Vision Activation Overlay", fontsize=12, fontweight='bold')
        ax3.axis('off')

        # 4. Question and Ground Truth
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.axis('off')

        # Extract question from messages
        messages = sample.get('messages', [])
        question = ""
        if messages and len(messages) > 0:
            if isinstance(messages[0], dict):
                question = messages[0].get('content', 'N/A')
            else:
                question = str(messages[0])

        generated_text = result.get('generated_text', 'N/A')
        ground_truth = result.get('ground_truth', 'N/A')
        is_correct = result.get('is_correct', False)

        # Format text content
        text_content = f"Question:\n{question}\n\n"
        text_content += f"Generated: {generated_text}\n"
        text_content += f"Ground Truth: {ground_truth}\n"
        text_content += f"Correct: {'✓' if is_correct else '✗'}\n\n"
        text_content += f"Query Type: {result.get('query_type', 'unknown')}\n"
        text_content += f"Sample ID: {result.get('sample_idx', 'N/A')}"

        # Add summary statistics
        vision_total = result.get('vision_total_score', 0)
        graph_total = result.get('graph_total_score', 0)
        vision_ratio = result.get('vision_ratio', 0)
        graph_ratio = result.get('graph_ratio', 0)

        text_content += f"\n\nActivation Summary:\n"
        text_content += f"Vision Total: {vision_total:.3f} ({vision_ratio * 100:.1f}%)\n"
        text_content += f"Graph Total: {graph_total:.3f} ({graph_ratio * 100:.1f}%)"

        ax4.text(0.05, 0.95, text_content,
                 verticalalignment='top',
                 fontsize=10,
                 family='monospace',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        ax4.set_title("Question & Results", fontsize=12, fontweight='bold')

        # 5. Top Activated Graph Nodes (separate panel - only names)
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.axis('off')
        node_names = result.get('node_names', [])
        if has_graph and len(node_names) > 0 and graph_contrib.size > 0:
            # Get more candidates initially to ensure we have enough after filtering
            candidate_k = min(100, len(graph_contrib))  # Get top 100 candidates
            candidate_indices = np.argsort(graph_contrib)[-candidate_k:][::-1]  # Sort descending

            # Filter out nodes with null/empty names or "Unknown location" in name
            filtered_indices = []
            for idx in candidate_indices:
                if idx < len(node_names):
                    node_name = node_names[idx]
                    # Skip if name is empty, None, or contains "Unknown location"
                    if node_name and node_name.strip() and "Unknown location" not in node_name:
                        filtered_indices.append(idx)

            # Take top 20 after filtering
            top_indices = filtered_indices[:20]

            if len(top_indices) > 0:
                # Create text for top nodes - only names, no index, no score
                top_nodes_text = "🔝 Top Activated Graph Nodes:\n" + "=" * 60 + "\n"
                for i, idx in enumerate(top_indices):
                    node_name = node_names[idx]
                    # Truncate very long names for display
                    display_name = node_name if len(node_name) <= 60 else node_name[:57] + "..."
                    top_nodes_text += f"{i + 1:2d}. {display_name}\n"

                ax5.text(
                    0.05, 0.95,
                    top_nodes_text,
                    transform=ax5.transAxes,
                    verticalalignment="top",
                    fontsize=10,
                    fontproperties=CJK_FP,  # ✅ this alone controls font
                    bbox=dict(
                        boxstyle="round",
                        facecolor="lightyellow",
                        alpha=0.9,
                        edgecolor="black",
                        linewidth=1
                    ),
                )

                ax5.set_title("Top Activated Graph Nodes", fontsize=12, fontweight='bold')
            else:
                ax5.text(0.5, 0.5, "No Valid Graph Nodes\n(All filtered out)", ha='center', va='center', fontsize=14)
                ax5.set_title("Top Activated Graph Nodes", fontsize=12, fontweight='bold')
        else:
            ax5.text(0.5, 0.5, "No Graph Nodes Available", ha='center', va='center', fontsize=14)
            ax5.set_title("Top Activated Graph Nodes", fontsize=12, fontweight='bold')

        plt.tight_layout()
        return fig

    def plot_topk_activated_nodes_on_map(
            self,
            graph: nx.Graph,
            image_coords: Optional[Tuple[float, float]],
            node_ids: Optional[List[Any]],
            node_scores: Optional[Union[List[float], np.ndarray]],
            out_path: str,
            k: int = 50,
            use_basemap: bool = True,
            max_label_chars: int = 18,
    ) -> None:
        """
        Plot top-k graph nodes by activation score on an OSM-style basemap.

        - node_ids must align with node_scores (same order)
        - image_coords: (lon, lat) in EPSG:4326
        - If a node does not have explicit coords, we fall back to:
          (1) parsing from node_text "Coordinates: (lon, lat)"; or
          (2) looking up geometry in a city-level nodes.geojson by parsing "ID: <int>" from node_text.
        """
        node_scores = np.asarray(node_scores, dtype=float)
        # if len(node_scores) == 0 or len(node_ids) == 0:
        #     return

        k = min(k, len(node_scores), len(node_ids))
        top_idx = np.argsort(-node_scores)[:k]
        top_nodes = [node_ids[i] for i in top_idx]
        top_scores = node_scores[top_idx]

        # Recover lon/lat for top nodes
        pts: List[Tuple[float, float]] = []
        scores: List[float] = []
        for nid, sc in zip(node_ids, top_scores):
            lonlat = self._get_node_lonlat(graph, nid,)
            if lonlat is None:
                continue
            pts.append(lonlat)
            scores.append(float(sc))

        # print('pts: ', pts)
        if len(pts) < 1:
            logger.warning(f"[MapPlot] No plottable nodes (top-{k} all missing coords). out={out_path}")
            return

        # try:
        import geopandas as gpd
        from shapely.geometry import Point
        import matplotlib.pyplot as plt

        # Project WGS84 -> WebMercator for basemap tiles
        gdf = gpd.GeoDataFrame(
            {"score": scores},
            geometry=[Point(lon, lat) for lon, lat in pts],
            crs="EPSG:4326",
        ).to_crs(epsg=3857)

        # Calculate bounds to ensure square aspect ratio
        bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
        x_range = bounds[2] - bounds[0]
        y_range = bounds[3] - bounds[1]
        max_range = max(x_range, y_range)
        
        # Center the bounds and make them square
        x_center = (bounds[0] + bounds[2]) / 2
        y_center = (bounds[1] + bounds[3]) / 2
        padding = max_range * 0.1  # 10% padding on all sides
        # Add extra padding **only on the left** (~30% of the map extent)
        extra_padding_left = max_range * 0.30
        square_bounds = [
            x_center - max_range / 2 - padding - extra_padding_left,  # left (more padding)
            y_center - max_range / 2 - padding,                       # bottom
            x_center + max_range / 2 + padding,                       # right
            y_center + max_range / 2 + padding,                       # top
        ]
        
        # Calculate label offset as a percentage of the data extent (lower left of node)
        label_offset_x = -max_range * 0.02  # 2% of the data extent (to the left)
        label_offset_y = -max_range * 0.03  # 3% of the data extent (downward, increased for more spacing)

        # Create figure with space for colorbar on the right
        fig, ax = plt.subplots(figsize=(12, 12))
        
        # Set square aspect ratio and bounds BEFORE creating divider
        # This ensures the colorbar height matches the map height
        ax.set_xlim(square_bounds[0], square_bounds[2])
        ax.set_ylim(square_bounds[1], square_bounds[3])
        ax.set_aspect('equal', adjustable='box')
        
        # Adjust subplot to leave space for colorbar (after setting limits/aspect)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.1)

        # Use the same black->red->yellow feel as the token activation maps
        # ('hot' is close: black -> red -> yellow -> white)
        # Plot without automatic legend, we'll create colorbar manually
        # Set high zorder so nodes appear above labels
        gdf.plot(ax=ax, column="score", markersize=80, legend=False, cmap="hot", vmin=min(scores), vmax=max(scores), zorder=10)
        
        # Create colorbar with same height as map
        sm = plt.cm.ScalarMappable(cmap="hot", norm=plt.Normalize(vmin=min(scores), vmax=max(scores)))
        sm.set_array([])
        cbar = plt.colorbar(sm, cax=cax)
        cbar.set_label('Activation Score', rotation=270, labelpad=30, fontsize=20)  # Increased labelpad for more space
        cbar.ax.tick_params(labelsize=20, pad=10)  # Add padding to tick labels

        # Draw edges between plotted top nodes (thin + translucent so they don't overpower the basemap)
        xy3857: Dict[Any, Any] = {}
        for nid, score in zip(node_ids, top_scores):
            # print('nid:', nid)
            lonlat = self._get_node_lonlat(graph, nid,)
            print('lonlat: ', lonlat)
            if lonlat is None:
                continue
            p = gpd.GeoSeries([Point(lonlat[0], lonlat[1])], crs="EPSG:4326").to_crs(epsg=3857).iloc[0]
            xy3857[nid] = p

        for u, v in graph.edges():
            if u in xy3857 and v in xy3857:
                ax.plot(
                    [xy3857[u].x, xy3857[v].x],
                    [xy3857[u].y, xy3857[v].y],
                    linewidth=1.2,
                    alpha=0.35,
                    color="blue",
                    zorder=5,  # Lower zorder so edges appear behind nodes and labels
                )
        if image_coords is not None:
            img_pt = (
                gpd.GeoSeries(
                    [Point(image_coords[0], image_coords[1])],
                    crs="EPSG:4326",
                )
                .to_crs(epsg=3857)
                .iloc[0]
            )

            ax.scatter(
                [img_pt.x],
                [img_pt.y],
                marker="*",
                s=420,  # Increased size for better visibility
                edgecolors="black",
                facecolors="gold",
                linewidths=1.5,
                zorder=11,  # Highest zorder so star appears above nodes and labels
                label="Image location",
            )


        def _shorten(s: str, n: int) -> str:
            s = (s or "").strip()
            return s if len(s) <= n else s[: n - 1] + "…"

        for nid, geom in xy3857.items():
            attrs = graph.nodes[nid] if graph.has_node(nid) else {}
            label = None

            node_text = (attrs.get("node_text", "") or "").strip()
            if node_text:
                m = re.search(r"\bName:\s*([^,\n]+)", node_text)
                if m:
                    label = m.group(1).strip()

            if not label:
                label = str(nid)
            
            ax.text(
                geom.x + label_offset_x,
                geom.y + label_offset_y,
                _shorten(label, max_label_chars),
                fontsize=20,  # Larger font size for KDD paper
                fontproperties=CJK_FP,
                alpha=0.92,
                zorder=7,  # Lower zorder so nodes and star appear above labels
                horizontalalignment='right',  # Align text to end at the offset position (right of text box, so text extends left)
                verticalalignment='bottom',  # Align text bottom at the offset position
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    facecolor="white",
                    alpha=0.75,
                    edgecolor="none",
                ),
            )

            # -------------------------
            # 6. Basemap + legend
            # -------------------------
        if use_basemap:
            cx.add_basemap(ax, source=cx.providers.CartoDB.Positron)
            # Re-apply bounds after basemap to ensure they're maintained
            ax.set_xlim(square_bounds[0], square_bounds[2])
            ax.set_ylim(square_bounds[1], square_bounds[3])
            ax.set_aspect('equal', adjustable='box')

        # Manual legend (node meaning)
        import matplotlib.lines as mlines

        node_handle = mlines.Line2D(
            [], [], marker="o", linestyle="None",
            markersize=8, color="red", label="Graph node (activation-colored)"
        )
        star_handle = mlines.Line2D(
            [], [], marker="*", linestyle="None",
            markersize=12, color="gold", markeredgecolor="black",
            label="Image location"
        )
        edge_handle = mlines.Line2D(
            [], [], linestyle="-", linewidth=1.2,
            alpha=0.35, color="blue", label="Graph edge"
        )

        ax.legend(handles=[node_handle, star_handle, edge_handle], loc="lower left", fontsize=20, frameon=True, fancybox=False)

        ax.set_axis_off()
        plt.tight_layout()

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        # Save with pad_inches to maintain square aspect ratio
        fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.1)
        # Also save as PDF for high-resolution insertion in Overleaf
        pdf_path = out_path.replace('.png', '.pdf')
        fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.1, format='pdf')
        plt.close(fig)

        logger.info(f"[MapPlot] Saved top-{k} activation map: {out_path} and {pdf_path}")

    def concatenate_samples_visualization(
            self,
            tam_results: List[Dict[str, Any]],
            samples: List[Dict[str, Any]],
            sample_indices: List[int] = [0, 3],
    ) -> None:
        """
        Concatenate only the input images and corresponding map plots for
        the specified samples into a single PNG + PDF figure.

        This ignores the full TAM panels (tam_rank*_sampleX.png) and uses
        only the map visualizations (map_rank*_sampleX_top20.png).
        """
        import glob

        # Map from sample_idx -> result for quick lookup
        results_dict = {r["sample_idx"]: r for r in tam_results}

        sample_images: List[Optional[Image.Image]] = []
        sample_map_plots: List[Optional[Image.Image]] = []

        for sample_idx in sample_indices:
            if sample_idx >= len(samples):
                logger.warning(f"Sample index {sample_idx} out of range, skipping")
                continue
            if sample_idx not in results_dict:
                logger.warning(f"No TAM result found for sample {sample_idx}, skipping")
                continue

            sample = samples[sample_idx]
            result = results_dict[sample_idx]

            # ---- Input image ----
            img_path = result.get("image_path", "")
            if (not img_path) or (not os.path.exists(img_path)):
                img_path = sample.get("images", "")
                if isinstance(img_path, list):
                    img_path = img_path[0] if img_path else ""
                if img_path and not os.path.isabs(img_path):
                    img_path = os.path.join(self.args.image_root, os.path.basename(img_path))

            if img_path and os.path.exists(img_path):
                try:
                    input_img = Image.open(img_path).convert("RGB")
                    sample_images.append(input_img)
                except Exception as e:
                    logger.warning(f"Could not load image for sample {sample_idx}: {e}")
                    sample_images.append(None)
            else:
                logger.warning(f"Image not found for sample {sample_idx}: {img_path}")
                sample_images.append(None)

            # ---- Map plot (only) ----
            map_plot: Optional[Image.Image] = None
            map_pattern = os.path.join(
                self.args.tam_output_dir,
                f"map_rank*_sample{sample_idx}_top20.png",
            )
            map_matches = glob.glob(map_pattern)
            if map_matches:
                map_path = map_matches[0]
                if os.path.exists(map_path):
                    try:
                        map_plot = Image.open(map_path).convert("RGB")
                    except Exception as e:
                        logger.warning(f"Could not load map plot for sample {sample_idx}: {e}")

            if map_plot is None:
                logger.warning(
                    f"No map plot found for sample {sample_idx}, this sample will be skipped"
                )
            sample_map_plots.append(map_plot)

        if not sample_images or all(img is None for img in sample_images):
            logger.error("No valid input images found for concatenation")
            return
        if not sample_map_plots or all(mp is None for mp in sample_map_plots):
            logger.error("No valid map plots found for concatenation")
            return

        # Standard display height for both input images and maps
        target_height = 800

        # First pass: resize map plots by height while preserving aspect ratio.
        # Track the maximum width so we can pad all maps to a common size.
        resized_map_plots: List[Optional[Image.Image]] = []
        max_map_width = 0
        for mp in sample_map_plots:
            if mp is None:
                resized_map_plots.append(None)
                continue
            map_aspect = mp.width / mp.height
            map_target_width = int(target_height * map_aspect)
            mp_resized = mp.resize((map_target_width, target_height), Image.Resampling.LANCZOS)
            resized_map_plots.append(mp_resized)
            max_map_width = max(max_map_width, mp_resized.width)

        if max_map_width == 0:
            logger.error("All map plots are invalid after resizing")
            return

        # Build one row per sample: [cropped input image | padded map]
        concatenated_rows: List[Image.Image] = []
        for sample_idx, input_img, map_resized in zip(
                sample_indices, sample_images, resized_map_plots
        ):
            if input_img is None or map_resized is None:
                continue

            # Center-crop input image to a square, then resize to target_height.
            w, h = input_img.width, input_img.height
            side = min(w, h)
            left = (w - side) // 2
            upper = (h - side) // 2
            right = left + side
            lower = upper + side
            input_img_cropped = input_img.crop((left, upper, right, lower))

            target_width = target_height  # square after cropping
            input_img_resized = input_img.resize(
                (target_width, target_height), Image.Resampling.LANCZOS
            )

            # Pad map to common (max_map_width, target_height) canvas
            map_canvas = Image.new("RGB", (max_map_width, target_height), color="white")
            x_off = (max_map_width - map_resized.width) // 2
            map_canvas.paste(map_resized, (x_off, 0))

            row_width = input_img_resized.width + map_canvas.width
            row_height = target_height
            row_img = Image.new("RGB", (row_width, row_height), color="white")
            row_img.paste(input_img_resized, (0, 0))
            row_img.paste(map_canvas, (input_img_resized.width, 0))

            concatenated_rows.append(row_img)

        if not concatenated_rows:
            logger.error("No valid rows to concatenate")
            return

        # Stack rows vertically
        final_width = max(row.width for row in concatenated_rows)
        final_height = sum(row.height for row in concatenated_rows)

        final_img = Image.new("RGB", (final_width, final_height), color="white")
        y_offset = 0
        for row in concatenated_rows:
            x_offset = (final_width - row.width) // 2  # center each row
            final_img.paste(row, (x_offset, y_offset))
            y_offset += row.height

        # Save PNG
        png_output_path = os.path.join(
            self.args.tam_output_dir,
            f"concatenated_samples_{'_'.join(map(str, sample_indices))}.png",
        )
        final_img.save(png_output_path)
        logger.info(f"Saved concatenated visualization (PNG): {png_output_path}")

        # Save PDF (via matplotlib)
        pdf_output_path = png_output_path.replace(".png", ".pdf")
        fig, ax = plt.subplots(figsize=(final_width / 100, final_height / 100), dpi=100)
        ax.imshow(final_img)
        ax.axis("off")
        plt.tight_layout(pad=0)
        fig.savefig(pdf_output_path, format="pdf", bbox_inches="tight", pad_inches=0, dpi=300)
        plt.close(fig)
        logger.info(f"Saved concatenated visualization (PDF): {pdf_output_path}")


    def run(self) -> Dict[str, Any]:
        """Main entry point for spatial reasoning TAM evaluation"""
        args = self.args
        logger.info("Starting spatial reasoning TAM evaluation...")

        # Load dataset
        samples = self.load_spatial_reasoning_dataset()
        logger.info(f"Loaded {len(samples)} samples")

        # Run TAM analysis on each sample
        tam_results = []
        for idx, sample in enumerate(tqdm(samples, desc="Spatial Reasoning TAM Evaluation")):
            try:
                result = self.analyze_sample_with_tam(sample, idx)
                if result:
                    tam_results.append(result)
            except Exception as e:
                logger.error(f"Error processing sample {idx}: {e}")
                import traceback
                traceback.print_exc()
                continue

        # Aggregate results
        aggregate_stats = self.compute_aggregate_statistics(tam_results)

        # Create evaluation report
        eval_report = {
            'tam_analysis': aggregate_stats,
            'time': args.time,
            'model': args.model,
            'adapters': args.adapters,
            'num_samples': len(tam_results),
            'spatial_reasoning_data_path': args.spatial_reasoning_data_path,
            'tam_output_dir': args.tam_output_dir,
        }

        # Save results
        if args.result_jsonl:
            append_to_jsonl(args.result_jsonl, eval_report)
            logger.info(f'Spatial reasoning TAM results saved to: {args.result_jsonl}')

        # Visualize top samples if requested
        if args.visualize_top_k > 0 and args.save_attention_maps:
            try:
                # If we have results, visualize them (even if just 1 sample)
                if tam_results:
                    k = min(args.visualize_top_k, len(tam_results))
                    self.visualize_top_samples(tam_results, samples, k=k)
                    # Additionally, concatenate samples 0 and 3 input images + map plots
                    try:
                        if len(tam_results) > 3:
                            self.concatenate_samples_visualization(
                                tam_results, samples, sample_indices=[0, 3]
                            )
                    except Exception as concat_e:
                        logger.warning(f"Error creating concatenated visualization: {concat_e}")
            except Exception as e:
                logger.error(f"Error during visualization: {e}")
                import traceback
                traceback.print_exc()

        # Run Stage 1 vs Stage 2 comparison if requested
        if args.compare_stage1_stage2:
            try:
                logger.info("=" * 80)
                logger.info("Running Stage 1 vs Stage 2 comparison on image-only samples...")
                logger.info("=" * 80)
                comparison_results = self.compare_stage1_stage2_attention(samples)

                # Add comparison results to eval report
                eval_report['stage_comparison'] = comparison_results

                # Save comparison results
                if args.result_jsonl:
                    # Update the existing entry
                    import json
                    with open(args.result_jsonl, 'r') as f:
                        lines = f.readlines()
                    if lines:
                        # Update last line
                        last_entry = json.loads(lines[-1])
                        last_entry['stage_comparison'] = comparison_results
                        with open(args.result_jsonl, 'w') as f:
                            for line in lines[:-1]:
                                f.write(line)
                            f.write(json.dumps(last_entry) + '\n')
            except Exception as e:
                logger.error(f"Error during Stage 1 vs Stage 2 comparison: {e}")
                import traceback
                traceback.print_exc()

        return eval_report

    def _create_comparison_model(self, adapter_path: str, stage_name: str):
        """Create a second model instance for comparison"""
        from swift.llm.infer import prepare_model_template
        from copy import deepcopy

        # Create new args with different adapter path
        comparison_args = deepcopy(self.args)
        comparison_args.adapters = adapter_path

        logger.info(f"Loading {stage_name} model from: {adapter_path}")
        model, template = prepare_model_template(comparison_args)
        template.model = model

        # Load graph encoder if needed
        if comparison_args.template in ('qwen2_vl_graph', 'qwen2_5_vl_graph') and adapter_path:
            try:
                from swift.llm.model.model.graph_encoder_spatial import TextAttributedGraphEncoderSpatial

                graph_encoder_path = os.path.join(adapter_path, 'graph_encoder.bin')
                graph_config_path = os.path.join(adapter_path, 'graph_encoder_config.json')

                if os.path.exists(graph_encoder_path):
                    # Load checkpoint early to inspect shapes for parameter inference
                    graph_encoder_state = torch.load(graph_encoder_path, map_location='cpu')

                    if os.path.exists(graph_config_path):
                        with open(graph_config_path, 'r') as f:
                            saved_config = json.load(f)
                    else:
                        saved_config = {}

                    # Defaults mirror training-time init (see init_graph_encoder)
                    defaults = {
                        'hidden_dim': model.config.hidden_size,
                        'output_dim': model.config.hidden_size,
                        'num_layers': 2,
                        'edge_dim': 64,
                        'use_spatial_encoding': True,
                        'spatial_embed_dim': 128,
                        'spatial_frequency_num': 8,
                        'use_edge_features': True,
                        'use_gat': True,
                        'gat_heads': 4,
                        'edge_use_distance': True,
                        'edge_use_direction': True,
                        'edge_use_displacement': True,
                    }

                    # Merge with defaults but ignore None values and deprecated parameters
                    graph_config = defaults.copy()
                    for k, v in saved_config.items():
                        if k != 'use_spatial_auxiliary' and v is not None:
                            graph_config[k] = v

                    # Infer parameters from checkpoint to ensure compatibility
                    try:
                        # Infer spatial_frequency_num from spatial encoder projection
                        spatial_proj_key = "spatial_encoder.projection.0.weight"
                        if spatial_proj_key in graph_encoder_state:
                            weight_shape = graph_encoder_state[spatial_proj_key].shape
                            if len(weight_shape) == 2:
                                input_dim = weight_shape[1]
                                inferred_freq_num = input_dim // 4
                                if inferred_freq_num > 0 and input_dim % 4 == 0:
                                    graph_config['spatial_frequency_num'] = inferred_freq_num
                                    logger.info(
                                        "Inferred spatial_frequency_num=%d from checkpoint "
                                        "(spatial_encoder.projection.0.weight shape: %s)",
                                        inferred_freq_num, weight_shape
                                    )

                        # Infer gat_heads from GNN layer attention or linear layer shapes
                        att_key = "convs.0.att"
                        if att_key in graph_encoder_state:
                            att_shape = graph_encoder_state[att_key].shape
                            if len(att_shape) >= 2:
                                inferred_gat_heads = att_shape[1]
                                graph_config['gat_heads'] = int(inferred_gat_heads)
                                logger.info(
                                    "Inferred gat_heads=%d from checkpoint (convs.0.att shape: %s)",
                                    inferred_gat_heads, att_shape
                                )
                        else:
                            # Fallback: infer from linear layer shape
                            lin_l_key = "convs.0.lin_l.weight"
                            if lin_l_key in graph_encoder_state:
                                lin_l_shape = graph_encoder_state[lin_l_key].shape
                                if len(lin_l_shape) == 2 and lin_l_shape[1] > 0:
                                    hidden_dim = lin_l_shape[1]
                                    if lin_l_shape[0] % hidden_dim == 0:
                                        inferred_gat_heads = lin_l_shape[0] // hidden_dim
                                        graph_config['gat_heads'] = int(inferred_gat_heads)
                                        logger.info(
                                            "Inferred gat_heads=%d from checkpoint "
                                            "(convs.0.lin_l.weight shape: %s, hidden_dim: %d)",
                                            inferred_gat_heads, lin_l_shape, hidden_dim
                                        )

                        # Infer edge_dim and enabled components from edge_combiner layer shape
                        edge_combiner_key = "edge_encoder.edge_combiner.0.weight"
                        if edge_combiner_key in graph_encoder_state:
                            combiner_shape = graph_encoder_state[edge_combiner_key].shape
                            logger.info("Found edge_combiner.0.weight in checkpoint with shape: %s", combiner_shape)
                            if len(combiner_shape) == 2:
                                output_dim = combiner_shape[0]  # edge_embed_dim * 2
                                input_dim = combiner_shape[1]  # edge_embed_dim * enabled_components

                                # Infer edge_dim from output_dim
                                candidate_edge_dim = output_dim // 2
                                if candidate_edge_dim > 0 and output_dim % 2 == 0:
                                    # Infer enabled_components from input_dim
                                    enabled_comp = input_dim // candidate_edge_dim
                                    if enabled_comp > 0 and input_dim % candidate_edge_dim == 0:
                                        graph_config['edge_dim'] = candidate_edge_dim
                                        logger.info(
                                            "Inferred edge_dim=%d from checkpoint "
                                            "(edge_encoder.edge_combiner.0.weight shape: %s)",
                                            candidate_edge_dim, combiner_shape
                                        )

                                        # Infer which edge components were enabled
                                        if enabled_comp == 1:
                                            graph_config['edge_use_distance'] = True
                                            graph_config['edge_use_direction'] = False
                                            graph_config['edge_use_displacement'] = False
                                            logger.info("Inferred enabled_components=1: only distance enabled")
                                        elif enabled_comp == 2:
                                            # Common case: distance + direction (displacement disabled)
                                            graph_config['edge_use_distance'] = True
                                            graph_config['edge_use_direction'] = True
                                            graph_config['edge_use_displacement'] = False
                                            logger.info(
                                                "Inferred enabled_components=2: distance + direction enabled "
                                                "(displacement disabled)"
                                            )
                                        elif enabled_comp == 3:
                                            # All enabled
                                            graph_config['edge_use_distance'] = True
                                            graph_config['edge_use_direction'] = True
                                            graph_config['edge_use_displacement'] = True
                                            logger.info("Inferred enabled_components=3: all edge features enabled")
                    except Exception as exc:
                        logger.warning("Could not infer some parameters from checkpoint: %s", exc)

                    language_model = model.model if hasattr(model, 'model') else model
                    graph_encoder = TextAttributedGraphEncoderSpatial(
                        qwen_model=language_model,
                        tokenizer=template.tokenizer,
                        **graph_config,
                        training_phase="frozen",
                    )

                    graph_encoder.load_state_dict(graph_encoder_state, strict=False)
                    graph_encoder.eval()
                    graph_encoder.to(self.device)
                    model.add_module('graph_encoder', graph_encoder)
            except Exception as e:
                logger.warning(f"Could not load graph encoder for {stage_name}: {e}")

        template.set_mode('pt')

        if hasattr(model, 'config'):
            model.config.output_hidden_states = True

        return model, template

    def compare_stage1_stage2_attention(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compare attention patterns between Stage 1 and Stage 2 models on image-only samples"""
        if not self.args.stage1_adapters or not self.args.stage2_adapters:
            raise ValueError("Both stage1_adapters and stage2_adapters must be provided for comparison")

        # Create Stage 1 model
        stage1_model, stage1_template = self._create_comparison_model(
            self.args.stage1_adapters, "Stage 1"
        )

        # Create Stage 2 model
        stage2_model, stage2_template = self._create_comparison_model(
            self.args.stage2_adapters, "Stage 2"
        )

        # Run TAM on both models for the same samples
        stage1_results = []
        stage2_results = []

        logger.info(f"Running TAM analysis on {len(samples)} image-only samples...")

        for idx, sample in enumerate(tqdm(samples, desc="Stage 1 vs Stage 2 TAM")):
            try:
                # Stage 1 TAM
                result1 = self.analyze_sample_with_tam_model(
                    sample, idx, stage1_model, stage1_template, "stage1"
                )
                if result1:
                    result1['stage'] = 'stage1'
                    stage1_results.append(result1)

                # Stage 2 TAM
                result2 = self.analyze_sample_with_tam_model(
                    sample, idx, stage2_model, stage2_template, "stage2"
                )
                if result2:
                    result2['stage'] = 'stage2'
                    stage2_results.append(result2)
            except Exception as e:
                logger.error(f"Error processing sample {idx} for comparison: {e}")
                continue

        # Compute comparison statistics
        comparison_stats = self._compute_stage_comparison_stats(stage1_results, stage2_results)

        # Create comparison visualizations
        if self.args.visualize_top_k > 0 and self.args.save_attention_maps:
            try:
                self.visualize_stage_comparison(stage1_results, stage2_results, samples)
            except Exception as e:
                logger.error(f"Error creating comparison visualizations: {e}")
                import traceback
                traceback.print_exc()

        # Visualize embeddings if requested
        if self.args.visualize_embeddings:
            try:
                logger.info("Extracting embeddings for visualization...")
                self.visualize_embedding_space(samples, stage1_model, stage1_template,
                                               stage2_model, stage2_template)
            except Exception as e:
                logger.error(f"Error visualizing embeddings: {e}")
                import traceback
                traceback.print_exc()

        return comparison_stats

    def extract_embeddings(self, sample: Dict[str, Any], model: torch.nn.Module,
                           template: Any, stage_name: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Extract query embedding and ground truth candidate embedding for a sample.

        Returns:
            (query_embedding, ground_truth_embedding) or (None, None) if extraction fails
        """
        try:
            # Get image path
            img_path = sample.get('images', '')
            if isinstance(img_path, list):
                img_path = img_path[0] if img_path else ''

            if not os.path.isabs(img_path):
                img_path = os.path.join(self.args.image_root, os.path.basename(img_path))

            if not os.path.exists(img_path):
                return None, None

            # Get graph path if available
            graph_path = sample.get('graphs', '')
            if isinstance(graph_path, list):
                graph_path = graph_path[0] if graph_path else ''

            has_graph = bool(graph_path and os.path.exists(graph_path))

            # Prepare query input
            encode_input = {
                'messages': sample['messages'],
                'images': [img_path]
            }
            if has_graph:
                encode_input['graphs'] = [graph_path]

            # Encode query
            inputs = template.encode(encode_input)

            # Convert to tensors
            tensor_keys = {'input_ids', 'attention_mask', 'labels', 'position_ids',
                           'image_grid_thw', 'video_grid_thw', 'pixel_values', 'image_sizes'}

            for key, value in inputs.items():
                if key in tensor_keys and isinstance(value, list):
                    inputs[key] = torch.tensor(value, device=self.device)
                elif isinstance(value, torch.Tensor):
                    inputs[key] = value.to(self.device)

            # Extract query embedding
            with torch.no_grad():
                encoded_inputs = template._post_encode(model, inputs)

                # Ensure batch dimension
                for key in ('input_ids', 'attention_mask', 'position_ids'):
                    if key in encoded_inputs and isinstance(encoded_inputs[key], torch.Tensor):
                        if encoded_inputs[key].dim() == 1:
                            encoded_inputs[key] = encoded_inputs[key].unsqueeze(0)

                if 'attention_mask' not in encoded_inputs or encoded_inputs['attention_mask'] is None:
                    if 'input_ids' in encoded_inputs:
                        input_ids = encoded_inputs['input_ids']
                        if input_ids.dim() == 1:
                            input_ids = input_ids.unsqueeze(0)
                        encoded_inputs['attention_mask'] = torch.ones_like(input_ids, dtype=torch.long)

                # Temporarily remove hooks to get full sequence
                model_to_use = model
                using_base_model = False
                if hasattr(model, 'model'):
                    model_to_use = model.model
                    using_base_model = True

                original_hooks = {}
                hooks_removed = False
                models_to_check = [model_to_use]
                if hasattr(model_to_use, 'model'):
                    models_to_check.append(model_to_use.model)

                for m in models_to_check:
                    if hasattr(m, '_forward_hooks') and len(m._forward_hooks) > 0:
                        original_hooks[id(m)] = m._forward_hooks.copy()
                        m._forward_hooks.clear()
                        hooks_removed = True

                try:
                    # Filter out image-related parameters when using base model
                    if using_base_model:
                        base_model_params = {
                            'input_ids', 'attention_mask', 'position_ids', 'past_key_values',
                            'inputs_embeds', 'use_cache', 'output_attentions', 'output_hidden_states',
                            'return_dict', 'cache_position'
                        }
                        filtered_inputs = {k: v for k, v in encoded_inputs.items() if k in base_model_params}
                    else:
                        filtered_inputs = encoded_inputs

                    outputs = model_to_use(
                        **filtered_inputs,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                finally:
                    if hooks_removed:
                        for model_id, hooks in original_hooks.items():
                            for m in models_to_check:
                                if id(m) == model_id:
                                    m._forward_hooks.update(hooks)
                                    break

                # Extract last hidden state
                if isinstance(outputs, dict):
                    hidden_states = outputs.get('hidden_states', None)
                    if hidden_states is None:
                        last_hidden_state = outputs.get('last_hidden_state', None)
                        if last_hidden_state is not None:
                            hidden_states = (last_hidden_state,)
                else:
                    hidden_states = getattr(outputs, 'hidden_states', None)
                    if hidden_states is None:
                        last_hidden_state = getattr(outputs, 'last_hidden_state', None)
                        if last_hidden_state is not None:
                            hidden_states = (last_hidden_state,)

                if hidden_states is None or len(hidden_states) == 0:
                    return None, None

                last_layer_hidden = hidden_states[-1]
                if last_layer_hidden.dim() == 3:
                    last_hidden = last_layer_hidden[0]  # (seq_len, hidden_dim)
                elif last_layer_hidden.dim() == 2:
                    last_hidden = last_layer_hidden
                else:
                    return None, None

                # Pool query embedding (mean pooling with attention mask)
                attn_mask = encoded_inputs.get('attention_mask', None)
                if attn_mask is not None:
                    if attn_mask.dim() > 1:
                        mask = attn_mask[0].unsqueeze(-1).to(last_hidden.device).to(last_hidden.dtype)
                    else:
                        mask = attn_mask.unsqueeze(-1).to(last_hidden.device).to(last_hidden.dtype)
                    masked_hidden = last_hidden * mask
                    denom = mask.sum() + 1e-8
                    query_embedding = masked_hidden.sum(dim=0) / denom
                else:
                    query_embedding = last_hidden.mean(dim=0)

                query_embedding = F.normalize(query_embedding, p=2, dim=0)

            # Extract ground truth candidate embedding
            ground_truth_text = sample.get('ground_truth', '')
            if not ground_truth_text:
                return query_embedding.detach().cpu().numpy(), None

            # Encode ground truth text
            cand_inputs = {
                'messages': [{'role': 'user', 'content': ground_truth_text}]
            }

            with torch.no_grad():
                cand_encoded = template._post_encode(model, cand_inputs)

                for key in ('input_ids', 'attention_mask', 'position_ids'):
                    if key in cand_encoded and isinstance(cand_encoded[key], torch.Tensor):
                        if cand_encoded[key].dim() == 1:
                            cand_encoded[key] = cand_encoded[key].unsqueeze(0)

                if 'attention_mask' not in cand_encoded or cand_encoded['attention_mask'] is None:
                    if 'input_ids' in cand_encoded:
                        cand_input_ids = cand_encoded['input_ids']
                        if cand_input_ids.dim() == 1:
                            cand_input_ids = cand_input_ids.unsqueeze(0)
                        cand_encoded['attention_mask'] = torch.ones_like(cand_input_ids, dtype=torch.long)

                # Remove hooks for candidate too
                cand_original_hooks = {}
                cand_hooks_removed = False
                cand_models_to_check = [model_to_use]
                if hasattr(model_to_use, 'model'):
                    cand_models_to_check.append(model_to_use.model)

                for m in cand_models_to_check:
                    if hasattr(m, '_forward_hooks') and len(m._forward_hooks) > 0:
                        cand_original_hooks[id(m)] = m._forward_hooks.copy()
                        m._forward_hooks.clear()
                        cand_hooks_removed = True

                try:
                    # Filter out image-related parameters when using base model
                    if using_base_model:
                        base_model_params = {
                            'input_ids', 'attention_mask', 'position_ids', 'past_key_values',
                            'inputs_embeds', 'use_cache', 'output_attentions', 'output_hidden_states',
                            'return_dict', 'cache_position'
                        }
                        cand_filtered = {k: v for k, v in cand_encoded.items() if k in base_model_params}
                    else:
                        cand_filtered = cand_encoded

                    cand_outputs = model_to_use(
                        **cand_filtered,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                finally:
                    if cand_hooks_removed:
                        for model_id, hooks in cand_original_hooks.items():
                            for m in cand_models_to_check:
                                if id(m) == model_id:
                                    m._forward_hooks.update(hooks)
                                    break

                # Extract candidate embedding
                if isinstance(cand_outputs, dict):
                    cand_hidden_states = cand_outputs.get('hidden_states', None)
                    if cand_hidden_states is None:
                        cand_last_hidden_state = cand_outputs.get('last_hidden_state', None)
                        if cand_last_hidden_state is not None:
                            cand_hidden_states = (cand_last_hidden_state,)
                else:
                    cand_hidden_states = getattr(cand_outputs, 'hidden_states', None)
                    if cand_hidden_states is None:
                        cand_last_hidden_state = getattr(cand_outputs, 'last_hidden_state', None)
                        if cand_last_hidden_state is not None:
                            cand_hidden_states = (cand_last_hidden_state,)

                if cand_hidden_states is None or len(cand_hidden_states) == 0:
                    return query_embedding.detach().cpu().numpy(), None

                cand_last_layer = cand_hidden_states[-1]
                if cand_last_layer.dim() == 3:
                    cand_last_hidden = cand_last_layer[0]
                elif cand_last_layer.dim() == 2:
                    cand_last_hidden = cand_last_layer
                else:
                    return query_embedding.detach().cpu().numpy(), None

                # Pool candidate embedding
                cand_attn_mask = cand_encoded.get('attention_mask', None)
                if cand_attn_mask is not None:
                    if cand_attn_mask.dim() > 1:
                        cand_mask = cand_attn_mask[0].unsqueeze(-1).to(cand_last_hidden.device).to(
                            cand_last_hidden.dtype)
                    else:
                        cand_mask = cand_attn_mask.unsqueeze(-1).to(cand_last_hidden.device).to(cand_last_hidden.dtype)
                    cand_masked = cand_last_hidden * cand_mask
                    cand_denom = cand_mask.sum() + 1e-8
                    gt_embedding = cand_masked.sum(dim=0) / cand_denom
                else:
                    gt_embedding = cand_last_hidden.mean(dim=0)

                gt_embedding = F.normalize(gt_embedding, p=2, dim=0)

            return (query_embedding.detach().cpu().numpy(),
                    gt_embedding.detach().cpu().numpy())

        except Exception as e:
            logger.warning(f"Error extracting embeddings for {stage_name}: {e}")
            return None, None

    def visualize_embedding_space(self, samples: List[Dict[str, Any]],
                                  stage1_model: torch.nn.Module, stage1_template: Any,
                                  stage2_model: torch.nn.Module, stage2_template: Any):
        """Visualize query and ground truth embeddings in embedding space for Stage 1 vs Stage 2"""
        logger.info("Extracting embeddings from Stage 1 and Stage 2 models...")

        # Extract embeddings
        stage1_query_embs = []
        stage1_gt_embs = []
        stage2_query_embs = []
        stage2_gt_embs = []
        valid_indices = []

        for idx, sample in enumerate(tqdm(samples, desc="Extracting embeddings")):
            # Stage 1 embeddings
            q1, gt1 = self.extract_embeddings(sample, stage1_model, stage1_template, "Stage 1")
            # Stage 2 embeddings
            q2, gt2 = self.extract_embeddings(sample, stage2_model, stage2_template, "Stage 2")

            if q1 is not None and gt1 is not None and q2 is not None and gt2 is not None:
                stage1_query_embs.append(q1)
                stage1_gt_embs.append(gt1)
                stage2_query_embs.append(q2)
                stage2_gt_embs.append(gt2)
                valid_indices.append(idx)

        if len(stage1_query_embs) == 0:
            logger.warning("No valid embeddings extracted. Cannot create visualization.")
            return

        # Convert to numpy arrays
        stage1_query_embs = np.array(stage1_query_embs)
        stage1_gt_embs = np.array(stage1_gt_embs)
        stage2_query_embs = np.array(stage2_query_embs)
        stage2_gt_embs = np.array(stage2_gt_embs)

        logger.info(f"Extracted {len(stage1_query_embs)} valid embeddings")

        # Compute alignment metrics
        stage1_similarities = np.array([np.dot(q, gt) for q, gt in zip(stage1_query_embs, stage1_gt_embs)])
        stage2_similarities = np.array([np.dot(q, gt) for q, gt in zip(stage2_query_embs, stage2_gt_embs)])

        logger.info(
            f"Stage 1 mean query-GT similarity: {stage1_similarities.mean():.4f} ± {stage1_similarities.std():.4f}")
        logger.info(
            f"Stage 2 mean query-GT similarity: {stage2_similarities.mean():.4f} ± {stage2_similarities.std():.4f}")
        logger.info(f"Improvement: {stage2_similarities.mean() - stage1_similarities.mean():.4f}")

        # Combine all embeddings for dimensionality reduction
        all_embs = np.vstack([
            stage1_query_embs,
            stage1_gt_embs,
            stage2_query_embs,
            stage2_gt_embs
        ])

        # Dimensionality reduction
        logger.info(f"Performing {self.args.embedding_reduction_method.upper()} dimensionality reduction...")
        if self.args.embedding_reduction_method.lower() == 'umap' and HAS_UMAP:
            reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
            reduced = reducer.fit_transform(all_embs)
        elif self.args.embedding_reduction_method.lower() == 'pca':
            reducer = PCA(n_components=2, random_state=42)
            reduced = reducer.fit_transform(all_embs)
        else:  # t-SNE (default)
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_embs) - 1))
            reduced = reducer.fit_transform(all_embs)

        n_samples = len(stage1_query_embs)
        stage1_query_2d = reduced[:n_samples]
        stage1_gt_2d = reduced[n_samples:2 * n_samples]
        stage2_query_2d = reduced[2 * n_samples:3 * n_samples]
        stage2_gt_2d = reduced[3 * n_samples:]

        # Create visualization
        fig = plt.figure(figsize=(16, 8))
        gs = fig.add_gridspec(1, 2, hspace=0.3, wspace=0.3)

        # Stage 1 plot
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.scatter(stage1_query_2d[:, 0], stage1_query_2d[:, 1],
                    c='blue', alpha=0.6, s=50, label='Query Embeddings', marker='o')
        ax1.scatter(stage1_gt_2d[:, 0], stage1_gt_2d[:, 1],
                    c='red', alpha=0.6, s=50, label='Ground Truth Embeddings', marker='s')

        # Draw lines connecting query-GT pairs
        for i in range(n_samples):
            ax1.plot([stage1_query_2d[i, 0], stage1_gt_2d[i, 0]],
                     [stage1_query_2d[i, 1], stage1_gt_2d[i, 1]],
                     'gray', alpha=0.3, linewidth=0.5)

        ax1.set_title(f'Stage 1 Embedding Space\n(Mean Query-GT Similarity: {stage1_similarities.mean():.4f})',
                      fontsize=12, fontweight='bold')
        ax1.set_xlabel(f'{self.args.embedding_reduction_method.upper()} Component 1', fontsize=10)
        ax1.set_ylabel(f'{self.args.embedding_reduction_method.upper()} Component 2', fontsize=10)
        ax1.legend()
        ax1.grid(alpha=0.3)

        # Stage 2 plot
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.scatter(stage2_query_2d[:, 0], stage2_query_2d[:, 1],
                    c='blue', alpha=0.6, s=50, label='Query Embeddings', marker='o')
        ax2.scatter(stage2_gt_2d[:, 0], stage2_gt_2d[:, 1],
                    c='red', alpha=0.6, s=50, label='Ground Truth Embeddings', marker='s')

        # Draw lines connecting query-GT pairs
        for i in range(n_samples):
            ax2.plot([stage2_query_2d[i, 0], stage2_gt_2d[i, 0]],
                     [stage2_query_2d[i, 1], stage2_gt_2d[i, 1]],
                     'gray', alpha=0.3, linewidth=0.5)

        ax2.set_title(
            f'Stage 2 Embedding Space (Graph-Conditioned)\n(Mean Query-GT Similarity: {stage2_similarities.mean():.4f})',
            fontsize=12, fontweight='bold')
        ax2.set_xlabel(f'{self.args.embedding_reduction_method.upper()} Component 1', fontsize=10)
        ax2.set_ylabel(f'{self.args.embedding_reduction_method.upper()} Component 2', fontsize=10)
        ax2.legend()
        ax2.grid(alpha=0.3)

        # Add text box with statistics
        stats_text = (
            f"Embedding Alignment Statistics:\n"
            f"Stage 1: {stage1_similarities.mean():.4f} ± {stage1_similarities.std():.4f}\n"
            f"Stage 2: {stage2_similarities.mean():.4f} ± {stage2_similarities.std():.4f}\n"
            f"Improvement: {stage2_similarities.mean() - stage1_similarities.mean():.4f}\n"
            f"Better alignment: {sum(stage2_similarities > stage1_similarities)}/{n_samples} samples"
        )
        fig.text(0.5, 0.02, stats_text, ha='center', fontsize=10,
                 family='monospace',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.tight_layout()

        # Save visualization
        output_path = os.path.join(self.args.tam_output_dir, 'embedding_space_comparison.png')
        os.makedirs(self.args.tam_output_dir, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved embedding space visualization to: {output_path}")
        plt.close(fig)

    def analyze_sample_with_tam_model(
            self,
            sample: Dict[str, Any],
            sample_idx: int,
            model: torch.nn.Module,
            template: Any,
            stage_name: str
    ) -> Dict[str, Any]:
        """Run TAM analysis using a specific model (for Stage 1/Stage 2 comparison)"""
        # This is essentially the same as analyze_sample_with_tam but uses provided model/template
        # We'll reuse most of the logic but with different model

        # Get image path
        img_path = sample.get('images', '')
        if isinstance(img_path, list):
            img_path = img_path[0] if img_path else ''

        if not os.path.isabs(img_path):
            img_path = os.path.join(self.args.image_root, os.path.basename(img_path))

        if not os.path.exists(img_path):
            logger.warning(f"Image not found: {img_path}")
            return {}

        # Prepare input (image-only, no graph)
        encode_input = {
            'messages': sample['messages'],
            'images': [img_path]
        }

        # Encode input
        inputs = template.encode(encode_input)

        # Convert to tensors
        tensor_keys = {'input_ids', 'attention_mask', 'labels', 'position_ids',
                       'image_grid_thw', 'video_grid_thw', 'pixel_values', 'image_sizes'}

        for key, value in inputs.items():
            if key in tensor_keys and isinstance(value, list):
                inputs[key] = torch.tensor(value, device=self.device)
            elif isinstance(value, torch.Tensor):
                inputs[key] = value.to(self.device)

        # Get token boundaries
        token_boundaries = self._identify_token_boundaries(inputs, has_graph=False)

        # Embedding-style forward pass
        with torch.no_grad():
            encoded_inputs = template._post_encode(model, inputs)

            # Ensure batch dimension
            for key in ('input_ids', 'attention_mask', 'position_ids'):
                if key in encoded_inputs and isinstance(encoded_inputs[key], torch.Tensor):
                    if encoded_inputs[key].dim() == 1:
                        encoded_inputs[key] = encoded_inputs[key].unsqueeze(0)

            # Ensure attention_mask exists
            if 'attention_mask' not in encoded_inputs or encoded_inputs['attention_mask'] is None:
                if 'input_ids' in encoded_inputs:
                    input_ids = encoded_inputs['input_ids']
                    if input_ids.dim() == 1:
                        input_ids = input_ids.unsqueeze(0)
                    encoded_inputs['attention_mask'] = torch.ones_like(input_ids, dtype=torch.long)

            # Temporarily remove hooks to get full sequence hidden states
            model_to_use = model
            using_base_model = False
            if hasattr(model, 'model'):
                model_to_use = model.model
                using_base_model = True

            original_hooks = {}
            hooks_removed = False
            models_to_check = [model_to_use]
            if hasattr(model_to_use, 'model'):
                models_to_check.append(model_to_use.model)

            for m in models_to_check:
                if hasattr(m, '_forward_hooks') and len(m._forward_hooks) > 0:
                    original_hooks[id(m)] = m._forward_hooks.copy()
                    m._forward_hooks.clear()
                    hooks_removed = True

            try:
                # Filter out image-related parameters when using base model
                if using_base_model:
                    base_model_params = {
                        'input_ids', 'attention_mask', 'position_ids', 'past_key_values',
                        'inputs_embeds', 'use_cache', 'output_attentions', 'output_hidden_states',
                        'return_dict', 'cache_position'
                    }
                    filtered_inputs = {k: v for k, v in encoded_inputs.items() if k in base_model_params}
                else:
                    filtered_inputs = encoded_inputs

                outputs = model_to_use(
                    **filtered_inputs,
                    output_hidden_states=True,
                    return_dict=True,
                )
            finally:
                if hooks_removed:
                    for model_id, hooks in original_hooks.items():
                        for m in models_to_check:
                            if id(m) == model_id:
                                m._forward_hooks.update(hooks)
                                break

        if isinstance(outputs, dict):
            hidden_states = outputs.get('hidden_states', None)
            if hidden_states is None:
                last_hidden_state = outputs.get('last_hidden_state', None)
                if last_hidden_state is not None:
                    hidden_states = (last_hidden_state,)
                else:
                    raise ValueError(f"Model did not return hidden_states. Keys: {list(outputs.keys())}")
        else:
            hidden_states = getattr(outputs, 'hidden_states', None)
            if hidden_states is None:
                last_hidden_state = getattr(outputs, 'last_hidden_state', None)
                if last_hidden_state is not None:
                    hidden_states = (last_hidden_state,)
                else:
                    raise ValueError("Model did not return hidden_states.")

        if hidden_states is None or len(hidden_states) == 0:
            raise ValueError("hidden_states is empty or None.")

        # Extract last layer hidden states
        last_layer_hidden = hidden_states[-1]  # Should be (batch, seq_len, hidden_dim) or (seq_len, hidden_dim)

        # Handle different shapes
        if last_layer_hidden.dim() == 3:
            # (batch, seq_len, hidden_dim) -> (seq_len, hidden_dim)
            last_hidden = last_layer_hidden[0]
        elif last_layer_hidden.dim() == 2:
            # Already (seq_len, hidden_dim)
            last_hidden = last_layer_hidden
        else:
            raise ValueError(
                f"Unexpected last_layer_hidden shape: {last_layer_hidden.shape}, "
                f"expected 2D (seq_len, hidden_dim) or 3D (batch, seq_len, hidden_dim)"
            )

        # Build query embedding
        attn_mask = encoded_inputs.get('attention_mask', None)
        if attn_mask is not None:
            if attn_mask.dim() > 1:
                mask = attn_mask[0].unsqueeze(-1).to(last_hidden.device).to(last_hidden.dtype)
            else:
                mask = attn_mask.unsqueeze(-1).to(last_hidden.device).to(last_hidden.dtype)
            masked_hidden = last_hidden * mask
            denom = mask.sum() + 1e-8
            query_embedding = masked_hidden.sum(dim=0) / denom
        else:
            query_embedding = last_hidden.mean(dim=0)

        # Build candidate embeddings
        candidates = sample.get('candidates', None)
        ground_truth_text = sample.get('ground_truth', '')
        candidate_embeddings = []

        if candidates is None or len(candidates) == 0:
            if ground_truth_text:
                candidates = [ground_truth_text]
            else:
                candidates = []

        for cand_text in candidates:
            cand_encode_input = {
                'messages': [{'role': 'user', 'content': cand_text}],
            }
            cand_inputs = template.encode(cand_encode_input)

            for key, value in list(cand_inputs.items()):
                if isinstance(value, list):
                    cand_inputs[key] = torch.tensor(value, device=self.device)
                elif isinstance(value, torch.Tensor):
                    cand_inputs[key] = value.to(self.device)

            with torch.no_grad():
                cand_encoded = template._post_encode(model, cand_inputs)

                for key in ('input_ids', 'attention_mask', 'position_ids'):
                    if key in cand_encoded and isinstance(cand_encoded[key], torch.Tensor):
                        if cand_encoded[key].dim() == 1:
                            cand_encoded[key] = cand_encoded[key].unsqueeze(0)

                if 'attention_mask' not in cand_encoded or cand_encoded['attention_mask'] is None:
                    if 'input_ids' in cand_encoded:
                        cand_input_ids = cand_encoded['input_ids']
                        if cand_input_ids.dim() == 1:
                            cand_input_ids = cand_input_ids.unsqueeze(0)
                        cand_encoded['attention_mask'] = torch.ones_like(cand_input_ids, dtype=torch.long)

                cand_outputs = model(
                    **cand_encoded,
                    output_hidden_states=True,
                    return_dict=True,
                )

            if isinstance(cand_outputs, dict):
                cand_hidden_states = cand_outputs.get('hidden_states', None)
                if cand_hidden_states is None:
                    cand_last_hidden_state = cand_outputs.get('last_hidden_state', None)
                    if cand_last_hidden_state is not None:
                        cand_hidden_states = (cand_last_hidden_state,)
                    else:
                        raise ValueError(
                            f"Model did not return hidden_states for candidate. Keys: {list(cand_outputs.keys())}")
            else:
                cand_hidden_states = getattr(cand_outputs, 'hidden_states', None)
                if cand_hidden_states is None:
                    cand_last_hidden_state = getattr(cand_outputs, 'last_hidden_state', None)
                    if cand_last_hidden_state is not None:
                        cand_hidden_states = (cand_last_hidden_state,)
                    else:
                        raise ValueError("Model did not return hidden_states for candidate.")

            if cand_hidden_states is None or len(cand_hidden_states) == 0:
                raise ValueError("cand_hidden_states is empty or None.")

            # Extract last layer hidden states
            cand_last_layer = cand_hidden_states[-1]
            if cand_last_layer.dim() == 3:
                cand_last_hidden = cand_last_layer[0]  # (batch, seq_len, hidden_dim) -> (seq_len, hidden_dim)
            elif cand_last_layer.dim() == 2:
                cand_last_hidden = cand_last_layer  # Already (seq_len, hidden_dim)
            else:
                raise ValueError(f"Unexpected cand_last_layer shape: {cand_last_layer.shape}")

            cand_attn_mask = cand_encoded.get('attention_mask', None)
            if cand_attn_mask is not None:
                if cand_attn_mask.dim() > 1:
                    c_mask = cand_attn_mask[0].unsqueeze(-1).to(cand_last_hidden.device).to(cand_last_hidden.dtype)
                else:
                    c_mask = cand_attn_mask.unsqueeze(-1).to(cand_last_hidden.device).to(cand_last_hidden.dtype)
                c_masked = cand_last_hidden * c_mask
                c_denom = c_mask.sum() + 1e-8
                cand_emb = c_masked.sum(dim=0) / c_denom
            else:
                cand_emb = cand_last_hidden.mean(dim=0)

            candidate_embeddings.append(cand_emb)

        if not candidate_embeddings:
            return {}

        # Choose target candidate
        gt_idx = sample.get('ground_truth_idx', None)
        if gt_idx is None and ground_truth_text and candidates:
            try:
                gt_idx = candidates.index(ground_truth_text)
            except ValueError:
                gt_idx = 0
        if gt_idx is None or not (0 <= int(gt_idx) < len(candidate_embeddings)):
            gt_idx = 0
        gt_idx = int(gt_idx)

        target_candidate_embedding = candidate_embeddings[gt_idx]

        # Compute predicted candidate
        cand_stack = torch.stack(candidate_embeddings, dim=0)
        with torch.no_grad():
            q_norm = F.normalize(query_embedding.view(1, -1), dim=-1)
            c_norm = F.normalize(cand_stack, dim=-1)
            sims = torch.matmul(c_norm, q_norm.t()).squeeze(-1)
        pred_idx = int(torch.argmax(sims).item())
        generated_text = candidates[pred_idx] if candidates else ''

        # Compute TAM activation scores
        vision_scores = None
        if token_boundaries['vision_range'] is not None:
            vision_scores = self.tam_computer.compute_activation_scores_from_cosine(
                hidden_states=last_hidden,
                candidate_embedding=target_candidate_embedding,
                token_range=token_boundaries['vision_range'],
            )
            if self.tam_computer.use_causal_inference:
                vision_scores = self.tam_computer.apply_rank_gaussian_filter(vision_scores[None, :])[0]
            vision_contribution = self.tam_computer.aggregate_scores(vision_scores[None, :], method='mean')
        else:
            vision_contribution = np.array([])

        # Compute summary statistics
        vision_total = float(np.sum(vision_contribution)) if vision_contribution.size > 0 else 0.0

        result = {
            'sample_idx': sample_idx,
            'query_type': sample.get('query_type', sample.get('question_type', 'unknown')),
            'generated_text': generated_text,
            'ground_truth': sample.get('ground_truth', ''),
            'is_correct': generated_text.strip().lower() == sample.get('ground_truth', '').strip().lower(),
            'vision_contribution': vision_contribution.tolist() if vision_contribution.size > 0 else [],
            'vision_total_score': vision_total,
            'vision_ratio': 1.0,  # Image-only, so 100% vision
            'has_graph': False,
            'token_boundaries': token_boundaries,
            'image_path': img_path,
            'stage': stage_name,
        }

        return result

    def _compute_stage_comparison_stats(
            self,
            stage1_results: List[Dict[str, Any]],
            stage2_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Compute statistics comparing Stage 1 vs Stage 2 attention patterns"""
        if not stage1_results or not stage2_results:
            return {}

        # Match results by sample_idx
        stage1_dict = {r['sample_idx']: r for r in stage1_results}
        stage2_dict = {r['sample_idx']: r for r in stage2_results}

        common_samples = set(stage1_dict.keys()) & set(stage2_dict.keys())

        vision_total_diff = []
        vision_contribution_diffs = []
        attention_shift_magnitudes = []

        for sample_idx in common_samples:
            s1 = stage1_dict[sample_idx]
            s2 = stage2_dict[sample_idx]

            # Difference in total vision scores
            v1_total = s1.get('vision_total_score', 0.0)
            v2_total = s2.get('vision_total_score', 0.0)
            vision_total_diff.append(v2_total - v1_total)

            # Difference in per-token contributions
            v1_contrib = np.array(s1.get('vision_contribution', []))
            v2_contrib = np.array(s2.get('vision_contribution', []))

            if len(v1_contrib) > 0 and len(v2_contrib) > 0:
                min_len = min(len(v1_contrib), len(v2_contrib))
                v1_contrib = v1_contrib[:min_len]
                v2_contrib = v2_contrib[:min_len]

                diff = v2_contrib - v1_contrib
                vision_contribution_diffs.append(diff)

                # Magnitude of attention shift
                shift_magnitude = np.linalg.norm(diff)
                attention_shift_magnitudes.append(float(shift_magnitude))

        stats = {
            'num_common_samples': len(common_samples),
            'vision_total_score_diff': {
                'mean': float(np.mean(vision_total_diff)) if vision_total_diff else 0.0,
                'std': float(np.std(vision_total_diff)) if vision_total_diff else 0.0,
                'median': float(np.median(vision_total_diff)) if vision_total_diff else 0.0,
            },
            'attention_shift_magnitude': {
                'mean': float(np.mean(attention_shift_magnitudes)) if attention_shift_magnitudes else 0.0,
                'std': float(np.std(attention_shift_magnitudes)) if attention_shift_magnitudes else 0.0,
                'median': float(np.median(attention_shift_magnitudes)) if attention_shift_magnitudes else 0.0,
            },
            'stage1_stats': self.compute_aggregate_statistics(stage1_results),
            'stage2_stats': self.compute_aggregate_statistics(stage2_results),
        }

        return stats

    def visualize_stage_comparison(
            self,
            stage1_results: List[Dict[str, Any]],
            stage2_results: List[Dict[str, Any]],
            samples: List[Dict]
    ):
        """Create side-by-side visualizations comparing Stage 1 vs Stage 2 attention"""
        # Match results by sample_idx
        stage1_dict = {r['sample_idx']: r for r in stage1_results}
        stage2_dict = {r['sample_idx']: r for r in stage2_results}

        common_samples = sorted(set(stage1_dict.keys()) & set(stage2_dict.keys()))
        k = min(self.args.visualize_top_k, len(common_samples))

        logger.info(f"Creating comparison visualizations for {k} samples...")

        for rank, sample_idx in enumerate(common_samples[:k]):
            if sample_idx >= len(samples):
                continue

            try:
                s1_result = stage1_dict[sample_idx]
                s2_result = stage2_dict[sample_idx]
                sample = samples[sample_idx]

                fig = self._create_stage_comparison_visualization(s1_result, s2_result, sample)

                save_path = os.path.join(
                    self.args.tam_output_dir,
                    f"stage_comparison_rank{rank + 1}_sample{sample_idx}.png"
                )
                fig.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close(fig)

                logger.info(f"Saved comparison visualization: {save_path}")
            except Exception as e:
                logger.error(f"Error creating comparison visualization for sample {sample_idx}: {e}")
                continue

    def _create_stage_comparison_visualization(
            self,
            stage1_result: Dict[str, Any],
            stage2_result: Dict[str, Any],
            sample: Dict
    ) -> plt.Figure:
        """Create side-by-side visualization comparing Stage 1 vs Stage 2"""
        fig = plt.figure(figsize=(20, 12))

        # Layout: 2 rows, 4 columns
        # Row 1: [Image] [Stage1 Vision Heatmap] [Stage2 Vision Heatmap] [Difference]
        # Row 2: [Stage1 Vision Overlay] [Stage2 Vision Overlay] [Attention Shift] [Stats]

        img_path = stage1_result.get('image_path', '')
        if img_path and os.path.exists(img_path):
            try:
                img = Image.open(img_path).convert('RGB')
            except:
                img = None
        else:
            img = None

        # 1. Original Image
        ax1 = plt.subplot(2, 4, 1)
        if img is not None:
            ax1.imshow(img)
        else:
            ax1.text(0.5, 0.5, "Image Not Available", ha='center', va='center')
        ax1.set_title("Input Image", fontsize=12, fontweight='bold')
        ax1.axis('off')

        # 2. Stage 1 Vision Heatmap
        ax2 = plt.subplot(2, 4, 2)
        v1_contrib = np.array(stage1_result.get('vision_contribution', []))
        if v1_contrib.size > 0:
            side_len = int(np.sqrt(len(v1_contrib)))
            if side_len * side_len < len(v1_contrib):
                side_len += 1
            padded = np.zeros(side_len * side_len)
            padded[:len(v1_contrib)] = v1_contrib
            v1_map = padded.reshape(side_len, side_len)
            im2 = ax2.imshow(v1_map, cmap='hot', interpolation='bilinear')
            ax2.set_title(f"Stage 1 Vision Activation\n(Total: {stage1_result.get('vision_total_score', 0):.3f})",
                          fontsize=12, fontweight='bold')
            plt.colorbar(im2, ax=ax2, fraction=0.046)
        else:
            ax2.text(0.5, 0.5, "No Vision Tokens", ha='center', va='center')
            ax2.set_title("Stage 1 Vision Activation", fontsize=12, fontweight='bold')
        ax2.axis('off')

        # 3. Stage 2 Vision Heatmap
        ax3 = plt.subplot(2, 4, 3)
        v2_contrib = np.array(stage2_result.get('vision_contribution', []))
        if v2_contrib.size > 0:
            side_len = int(np.sqrt(len(v2_contrib)))
            if side_len * side_len < len(v2_contrib):
                side_len += 1
            padded = np.zeros(side_len * side_len)
            padded[:len(v2_contrib)] = v2_contrib
            v2_map = padded.reshape(side_len, side_len)
            im3 = ax3.imshow(v2_map, cmap='hot', interpolation='bilinear')
            ax3.set_title(f"Stage 2 Vision Activation\n(Total: {stage2_result.get('vision_total_score', 0):.3f})",
                          fontsize=12, fontweight='bold')
            plt.colorbar(im3, ax=ax3, fraction=0.046)
        else:
            ax3.text(0.5, 0.5, "No Vision Tokens", ha='center', va='center')
            ax3.set_title("Stage 2 Vision Activation", fontsize=12, fontweight='bold')
        ax3.axis('off')

        # 4. Difference Map
        ax4 = plt.subplot(2, 4, 4)
        if v1_contrib.size > 0 and v2_contrib.size > 0:
            min_len = min(len(v1_contrib), len(v2_contrib))
            v1_contrib = v1_contrib[:min_len]
            v2_contrib = v2_contrib[:min_len]
            diff = v2_contrib - v1_contrib

            side_len = int(np.sqrt(len(diff)))
            if side_len * side_len < len(diff):
                side_len += 1
            padded = np.zeros(side_len * side_len)
            padded[:len(diff)] = diff
            diff_map = padded.reshape(side_len, side_len)
            im4 = ax4.imshow(diff_map, cmap='RdBu_r', interpolation='bilinear',
                             vmin=-np.abs(diff_map).max(), vmax=np.abs(diff_map).max())
            ax4.set_title("Stage 2 - Stage 1\n(Red=More, Blue=Less)", fontsize=12, fontweight='bold')
            plt.colorbar(im4, ax=ax4, fraction=0.046)
        else:
            ax4.text(0.5, 0.5, "No Difference Data", ha='center', va='center')
            ax4.set_title("Difference Map", fontsize=12, fontweight='bold')
        ax4.axis('off')

        # 5. Stage 1 Vision Overlay
        ax5 = plt.subplot(2, 4, 5)
        if img is not None and v1_contrib.size > 0:
            img_array = np.array(img)
            h, w = img_array.shape[:2]
            side_len = int(np.sqrt(len(v1_contrib)))
            if side_len * side_len < len(v1_contrib):
                side_len += 1
            padded = np.zeros(side_len * side_len)
            padded[:len(v1_contrib)] = v1_contrib
            v1_map_2d = padded.reshape(side_len, side_len)
            v1_map_2d = (v1_map_2d - v1_map_2d.min()) / (v1_map_2d.max() - v1_map_2d.min() + 1e-8)
            v1_map_uint8 = (v1_map_2d * 255).astype(np.uint8)
            v1_map_pil = Image.fromarray(v1_map_uint8, mode='L')
            v1_map_resized = np.array(v1_map_pil.resize((w, h), Image.BILINEAR)) / 255.0
            import matplotlib.cm as cm
            heatmap1 = cm.hot(v1_map_resized)[:, :, :3]
            overlay1 = (0.5 * img_array / 255.0 + 0.5 * heatmap1)
            ax5.imshow(overlay1)
        else:
            ax5.text(0.5, 0.5, "No Overlay", ha='center', va='center')
        ax5.set_title("Stage 1 Overlay", fontsize=12, fontweight='bold')
        ax5.axis('off')

        # 6. Stage 2 Vision Overlay
        ax6 = plt.subplot(2, 4, 6)
        if img is not None and v2_contrib.size > 0:
            img_array = np.array(img)
            h, w = img_array.shape[:2]
            side_len = int(np.sqrt(len(v2_contrib)))
            if side_len * side_len < len(v2_contrib):
                side_len += 1
            padded = np.zeros(side_len * side_len)
            padded[:len(v2_contrib)] = v2_contrib
            v2_map_2d = padded.reshape(side_len, side_len)
            v2_map_2d = (v2_map_2d - v2_map_2d.min()) / (v2_map_2d.max() - v2_map_2d.min() + 1e-8)
            v2_map_uint8 = (v2_map_2d * 255).astype(np.uint8)
            v2_map_pil = Image.fromarray(v2_map_uint8, mode='L')
            v2_map_resized = np.array(v2_map_pil.resize((w, h), Image.BILINEAR)) / 255.0
            heatmap2 = cm.hot(v2_map_resized)[:, :, :3]
            overlay2 = (0.5 * img_array / 255.0 + 0.5 * heatmap2)
            ax6.imshow(overlay2)
        else:
            ax6.text(0.5, 0.5, "No Overlay", ha='center', va='center')
        ax6.set_title("Stage 2 Overlay", fontsize=12, fontweight='bold')
        ax6.axis('off')

        # 7. Attention Shift Magnitude
        ax7 = plt.subplot(2, 4, 7)
        if v1_contrib.size > 0 and v2_contrib.size > 0:
            min_len = min(len(v1_contrib), len(v2_contrib))
            v1_contrib = v1_contrib[:min_len]
            v2_contrib = v2_contrib[:min_len]
            diff = v2_contrib - v1_contrib
            shift_magnitude = np.linalg.norm(diff)

            # Bar chart showing shift magnitude
            ax7.barh(['Attention Shift'], [shift_magnitude], color='purple', alpha=0.7)
            ax7.set_xlabel("Magnitude", fontsize=10)
            ax7.set_title(f"Attention Shift\nMagnitude: {shift_magnitude:.3f}", fontsize=12, fontweight='bold')
            ax7.grid(axis='x', alpha=0.3)
        else:
            ax7.text(0.5, 0.5, "No Data", ha='center', va='center')
            ax7.set_title("Attention Shift", fontsize=12, fontweight='bold')

        # 8. Statistics
        ax8 = plt.subplot(2, 4, 8)
        ax8.axis('off')

        messages = sample.get('messages', [])
        question = messages[0].get('content', 'N/A') if messages and isinstance(messages[0], dict) else str(
            messages[0]) if messages else 'N/A'

        stats_text = f"Sample: {stage1_result.get('sample_idx', 'N/A')}\n\n"
        stats_text += f"Stage 1 Vision Total: {stage1_result.get('vision_total_score', 0):.3f}\n"
        stats_text += f"Stage 2 Vision Total: {stage2_result.get('vision_total_score', 0):.3f}\n"
        if v1_contrib.size > 0 and v2_contrib.size > 0:
            diff_total = stage2_result.get('vision_total_score', 0) - stage1_result.get('vision_total_score', 0)
            stats_text += f"Difference: {diff_total:+.3f}\n\n"
        stats_text += f"Stage 1 Correct: {'✓' if stage1_result.get('is_correct') else '✗'}\n"
        stats_text += f"Stage 2 Correct: {'✓' if stage2_result.get('is_correct') else '✗'}\n\n"
        stats_text += f"Question:\n{question[:100]}{'...' if len(question) > 100 else ''}"

        ax8.text(0.05, 0.95, stats_text,
                 verticalalignment='top',
                 fontsize=9,
                 family='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
        ax8.set_title("Comparison Stats", fontsize=12, fontweight='bold')

        plt.tight_layout()
        return fig

    def compute_aggregate_statistics(self, tam_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute aggregate statistics across all samples"""
        vision_totals = []
        graph_totals = []
        vision_ratios = []
        graph_ratios = []
        spatial_decay_correlations = []

        # Group by query type
        query_type_stats = {}

        for result in tam_results:
            if 'vision_total_score' in result:
                vision_totals.append(result['vision_total_score'])
            if 'graph_total_score' in result:
                graph_totals.append(result['graph_total_score'])
            if 'vision_ratio' in result:
                vision_ratios.append(result['vision_ratio'])
            if 'graph_ratio' in result:
                graph_ratios.append(result['graph_ratio'])

            # Spatial decay correlation
            if 'spatial_decay_stats' in result and result['spatial_decay_stats']:
                corr = result['spatial_decay_stats'].get('distance_attention_correlation', 0.0)
                if not np.isnan(corr):
                    spatial_decay_correlations.append(corr)

            # Query type statistics
            query_type = result.get('query_type', 'unknown')
            if query_type not in query_type_stats:
                query_type_stats[query_type] = {
                    'vision_ratios': [],
                    'graph_ratios': [],
                    'count': 0,
                }
            query_type_stats[query_type]['vision_ratios'].append(result.get('vision_ratio', 0.0))
            query_type_stats[query_type]['graph_ratios'].append(result.get('graph_ratio', 0.0))
            query_type_stats[query_type]['count'] += 1

        # Compute per-query-type averages
        query_type_summary = {}
        for qtype, stats in query_type_stats.items():
            query_type_summary[qtype] = {
                'mean_vision_ratio': float(np.mean(stats['vision_ratios'])) if stats['vision_ratios'] else 0.0,
                'mean_graph_ratio': float(np.mean(stats['graph_ratios'])) if stats['graph_ratios'] else 0.0,
                'count': stats['count'],
            }

        stats = {
            'vision_mean': float(np.mean(vision_totals)) if vision_totals else 0.0,
            'vision_std': float(np.std(vision_totals)) if vision_totals else 0.0,
            'graph_mean': float(np.mean(graph_totals)) if graph_totals else 0.0,
            'graph_std': float(np.std(graph_totals)) if graph_totals else 0.0,
            'vision_ratio_mean': float(np.mean(vision_ratios)) if vision_ratios else 0.0,
            'vision_ratio_std': float(np.std(vision_ratios)) if vision_ratios else 0.0,
            'graph_ratio_mean': float(np.mean(graph_ratios)) if graph_ratios else 0.0,
            'graph_ratio_std': float(np.std(graph_ratios)) if graph_ratios else 0.0,
            'spatial_decay_correlation_mean': float(
                np.mean(spatial_decay_correlations)) if spatial_decay_correlations else 0.0,
            'spatial_decay_correlation_std': float(
                np.std(spatial_decay_correlations)) if spatial_decay_correlations else 0.0,
            'query_type_statistics': query_type_summary,
        }

        return stats


def spatial_reasoning_tam_eval_main(args: Union[List[str], SpatialReasoningTAMEvalArguments, None] = None):
    """Main entry point for spatial reasoning TAM evaluation"""
    return SpatialReasoningTAMEval(args).run()


if __name__ == '__main__':
    spatial_reasoning_tam_eval_main()

