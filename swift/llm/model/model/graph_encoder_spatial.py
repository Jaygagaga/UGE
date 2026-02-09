"""
Enhanced Graph Encoder with Spatial Features

This is the ENHANCED version with:
- Spatial positional encoding (PE-GNN style)
- Geodesic edge features (GeoGNN style)
- GATv2 support for edge-aware message passing

Created: 2025-11-04
Based on: graph_encoder1.py (original) + spatial_encoders.py

To use this enhanced version:
1. Update imports in your code to use graph_encoder_spatial
2. Ensure graph.coords attribute exists in your PyG graphs
3. Configure spatial features via train arguments
"""

from torch_geometric.nn import GCNConv, GATv2Conv
from swift.utils import get_logger
logger = get_logger()

import torch
import torch.nn as nn
import torch.nn.functional as F
import re
from typing import Optional, List, Tuple

# Import spatial encoding modules
from .spatial_encoders import (
    SpatialPositionalEncoder,
    GeodesicEdgeEncoder
)


class QwenNodeEncoder(nn.Module):
    """Node text encoder using shared Qwen2VL language model"""
    
    def __init__(self, qwen_model=None, tokenizer=None, hidden_dim=1536, training_phase="frozen"):
        super(QwenNodeEncoder, self).__init__()
        
        # Use shared Qwen2VL language model components
        self.qwen_model = qwen_model  # Reference to main model
        self.tokenizer = tokenizer    # Shared tokenizer
        self.hidden_dim = hidden_dim  # Qwen2VL hidden dimension (1536)
        self.training_phase = training_phase
        
        # Node-specific projection layer (trainable)
        self.node_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize node head weights"""
        for module in self.node_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, texts):
        """Encode node texts using shared Qwen2VL encoder"""
        if not texts or len(texts) == 0:
            return torch.zeros((0, self.hidden_dim), device=next(self.parameters()).device)
            
        # Tokenize using shared Qwen2VL tokenizer
        # For Qwen2VL, tokenizer is a processor; use .tokenizer for text-only encoding
        device = next(self.parameters()).device
        text_tokenizer = self.tokenizer.tokenizer if hasattr(self.tokenizer, 'tokenizer') else self.tokenizer
        encoded_input = text_tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,  # Reasonable length for node descriptions
            return_tensors='pt'
        ).to(device)
        
        # Get embeddings from shared language model embedding layer
        # Support both Qwen (embed_tokens) and InternVL (get_input_embeddings) models
        input_ids = encoded_input.input_ids
        token_embeddings = None
        
        if self.qwen_model is not None:
            # Try Qwen-style: direct embed_tokens attribute
            if hasattr(self.qwen_model, 'embed_tokens'):
                token_embeddings = self.qwen_model.embed_tokens(input_ids)
            # Try InternVL-style: get_input_embeddings() method
            elif hasattr(self.qwen_model, 'get_input_embeddings'):
                embedding_layer = self.qwen_model.get_input_embeddings()
                if embedding_layer is not None:
                    token_embeddings = embedding_layer(input_ids)
            # Try model.language_model.get_input_embeddings() for InternVL
            elif hasattr(self.qwen_model, 'language_model') and hasattr(self.qwen_model.language_model, 'get_input_embeddings'):
                embedding_layer = self.qwen_model.language_model.get_input_embeddings()
                if embedding_layer is not None:
                    token_embeddings = embedding_layer(input_ids)
        
        if token_embeddings is None:
            # Fallback: create embeddings (should not happen in normal use)
            logger.warning(f"No shared language model embeddings available (model type: {type(self.qwen_model).__name__}), using random embeddings")
            text_tokenizer = self.tokenizer.tokenizer if hasattr(self.tokenizer, 'tokenizer') else self.tokenizer
            vocab_size = len(text_tokenizer) if text_tokenizer else 50000
            embedding_layer = nn.Embedding(vocab_size, self.hidden_dim).to(device)
            token_embeddings = embedding_layer(encoded_input.input_ids)
        
        # Mean pooling over sequence length
        attention_mask = encoded_input['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float() #to("cuda:0")
        pooled_embeddings = torch.sum(token_embeddings * attention_mask, dim=1) / torch.clamp(
            attention_mask.sum(dim=1), min=1e-9
        )  # [batch_size, hidden_dim]
        
        return pooled_embeddings
    
    def set_training_phase(self, phase="frozen"):
        """Set training phase for the encoder"""
        self.training_phase = phase
        
        # Node head is always trainable
        for param in self.node_head.parameters():
            param.requires_grad = True
            
        logger.info(f"QwenNodeEncoder set to {phase} phase")


class TextAttributedGraphEncoderSpatial(nn.Module):
    """
    Enhanced Graph Encoder with Spatial Features
    
    Key enhancements over original:
    1. Spatial positional encoding for geographic coordinates
    2. Geodesic edge features (distance, direction, displacement)
    3. GATv2 support for edge-aware message passing
    
    Args:
        qwen_model: Shared Qwen2VL model for text encoding
        tokenizer: Shared tokenizer
        hidden_dim: Hidden dimension (1536 for Qwen2VL)
        output_dim: Output dimension
        num_layers: Number of GNN layers (default: 2)
        edge_dim: Edge feature dimension (default: 64)
        training_phase: "frozen" or "joint"
        
        # NEW spatial arguments
        use_spatial_encoding: Whether to use spatial positional encoding
        spatial_embed_dim: Dimension of spatial embeddings (default: 128)
        spatial_frequency_num: Number of frequency bands for spatial encoding (default: 16)
        use_edge_features: Whether to use geodesic edge features
        use_gat: Whether to use GATv2 instead of GCN (recommended if using edge features)
        gat_heads: Number of attention heads for GATv2 (default: 4)
    """

    def __init__(
        self, 
        qwen_model=None, 
        tokenizer=None, 
        hidden_dim=1536, 
        output_dim=1536,
        num_layers=2, 
        edge_dim=64, 
        training_phase="frozen",
        # NEW: Spatial encoding parameters
        use_spatial_encoding=True,
        spatial_embed_dim=128,
        spatial_frequency_num=16,
        use_edge_features=True,
        use_gat=True,  # Use GAT if edge features enabled
        gat_heads=4,
        edge_use_distance=True,
        edge_use_direction=True,
        edge_use_displacement=True
    ):
        super().__init__()
        
        # Store configuration
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.edge_dim = edge_dim
        self.training_phase = training_phase
        self.num_layers = num_layers
        self.tokenizer = tokenizer
        
        # Spatial feature flags
        self.use_spatial_encoding = use_spatial_encoding
        self.spatial_embed_dim = spatial_embed_dim
        self.use_edge_features = use_edge_features
        self.use_gat = use_gat
        self.edge_use_distance = edge_use_distance
        self.edge_use_direction = edge_use_direction
        self.edge_use_displacement = edge_use_displacement
        
        # Text encoder (unchanged)
        self.node_encoder = QwenNodeEncoder(
            qwen_model=qwen_model,
            tokenizer=tokenizer, 
            hidden_dim=hidden_dim,
            training_phase=training_phase
        )
        
        # ===== NEW: Spatial Positional Encoder =====
        if self.use_spatial_encoding:
            self.spatial_encoder = SpatialPositionalEncoder(
                spa_embed_dim=spatial_embed_dim,
                frequency_num=spatial_frequency_num,
                max_radius=360.0,  # Covers full longitude range [-180, +180] for cross-city evaluation (matching PE-GNN)
                min_radius=1e-6,   # Matching PE-GNN's min_radius
                use_ffn=True       # Learnable projection
            )
            # Project concatenated features (text + spatial) back to hidden_dim
            self.node_projection = nn.Linear(hidden_dim + spatial_embed_dim, hidden_dim)
            logger.info(f"✓ Spatial positional encoding enabled: {spatial_embed_dim}D, {spatial_frequency_num} frequencies")
        else:
            self.node_projection = nn.Identity()
            logger.info("✗ Spatial positional encoding disabled")
        
        # ===== NEW: Geodesic Edge Encoder =====
        if self.use_edge_features:
            self.edge_encoder = GeodesicEdgeEncoder(
                edge_embed_dim=edge_dim,
                distance_units='meters',
                use_log_distance=True,
                use_distance=edge_use_distance,
                use_direction=edge_use_direction,
                use_displacement=edge_use_displacement
            )
            enabled_components = [
                name for flag, name in [
                    (edge_use_distance, "distance"),
                    (edge_use_direction, "direction"),
                    (edge_use_displacement, "displacement"),
                ] if flag
            ]
            logger.info(
                f"✓ Geodesic edge features enabled: {edge_dim}D ({', '.join(enabled_components)})"
            )
        else:
            self.edge_encoder = None
            logger.info("✗ Geodesic edge features disabled")
        
        # ===== GNN Layers: GATv2 or GCN =====
        self.convs = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        
        # Use GAT if use_gat is True (independent of edge features)
        use_gat_actual = self.use_gat
        
        for i in range(num_layers):
            if use_gat_actual:
                # GATv2 - can work with or without edge features
                if self.use_edge_features:
                    # GATv2 with edge features
                    self.convs.append(GATv2Conv(
                        in_channels=hidden_dim,
                        out_channels=hidden_dim,
                        heads=gat_heads,
                        concat=False,  # Average heads to maintain hidden_dim
                        edge_dim=edge_dim,
                        dropout=0.1,
                        add_self_loops=True,
                        share_weights=False
                    ))
                    logger.info(f"  Layer {i}: GATv2Conv ({gat_heads} heads, with edge features, edge_dim={edge_dim})")
                else:
                    # GATv2 without edge features (attention over nodes only)
                    self.convs.append(GATv2Conv(
                        in_channels=hidden_dim,
                        out_channels=hidden_dim,
                        heads=gat_heads,
                        concat=False,  # Average heads to maintain hidden_dim
                        dropout=0.1,
                        add_self_loops=True,
                        share_weights=False
                    ))
                    logger.info(f"  Layer {i}: GATv2Conv ({gat_heads} heads, no edge features)")
            else:
                # Standard GCN (no edge features)
                self.convs.append(GCNConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    improved=False,
                    cached=False,
                    add_self_loops=True,
                    normalize=True,
                    bias=True
                ))
                logger.info(f"  Layer {i}: GCNConv (no edge features)")
            
            self.layer_norms.append(nn.LayerNorm(hidden_dim))
        
        # ===== Output Projection =====
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        logger.info(f"Enhanced Graph Encoder initialized:")
        logger.info(f"  - Spatial encoding: {self.use_spatial_encoding}")
        logger.info(f"  - Edge features: {self.use_edge_features}")
        logger.info(f"  - GNN type: {'GATv2' if use_gat_actual else 'GCN'}")

    def set_training_phase(self, phase="frozen"):
        """Switch between training phases"""
        self.training_phase = phase
        
        # Set training phase for node encoder
        self.node_encoder.set_training_phase(phase)
        
        # Make graph modules trainable
        trainable_modules = [self.convs, self.layer_norms, self.output_proj]
        
        # Add spatial modules if enabled
        if self.use_spatial_encoding:
            trainable_modules.append(self.spatial_encoder)
        if self.use_edge_features:
            trainable_modules.append(self.edge_encoder)
        
        for module in trainable_modules:
            for param in module.parameters():
                param.requires_grad = True
        
        trainable_count = sum(1 for p in self.parameters() if p.requires_grad)
        total_count = sum(1 for p in self.parameters())
        
        logger.info(f"GraphEncoder (Spatial): {phase} phase - {trainable_count}/{total_count} params trainable")

    def parse_node_text(self, node_text: str, node_id: str, central_node_id=None) -> tuple:
        """
        Parse node_text string to extract coordinates and filtered text features.
        
        Args:
            node_text: Original node_text string
            node_id: Current node ID
            central_node_id: Central node ID for sanitization (None if not applicable)
            
        Returns:
            (filtered_text, coordinates) where:
            - filtered_text: String with only Category, Name, Planning area, District
            - coordinates: Tuple of (lon, lat) or None if not found
        """
        import re
        
        # Extract coordinates using regex
        # Format: ", Coordinates: (103.123456, 1.234567)" - note the leading comma and 6 decimal places
        # Also supports: "Coordinates: [103.123, 1.456]" or "Coordinates: 103.123, 1.456"
        coords = None
        # Improved regex: handles optional leading comma, parentheses/brackets, and proper floats
        coord_match = re.search(r',?\s*Coordinates:\s*[\(\[]?\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*[\)\]]?', node_text, re.IGNORECASE)
        if coord_match:
            # try:
            lon = float(coord_match.group(1))
            lat = float(coord_match.group(2))
            coords = (lon, lat)
            # except ValueError:
            #     logger.warning(f"Failed to parse coordinates from: {coord_match.group(0)}")
        
        # Extract filtered attributes (Category, Name, Planning area/Neighborhood, District/Borough)
        # Supports both Singapore and New York data formats
        filtered_parts = []
        
        # Extract Category
        category_match = re.search(r'Category:\s*([^,]+)', node_text, re.IGNORECASE)
        if category_match:
            filtered_parts.append(f"Category: {category_match.group(1).strip()}")
        
        # Extract Name
        name_match = re.search(r'Name:\s*([^,]+)', node_text, re.IGNORECASE)
        if name_match:
            filtered_parts.append(f"Name: {name_match.group(1).strip()}")
        
        # Extract Planning area (Singapore) or Neighborhood (New York)
        planning_match = re.search(r'Planning area:\s*([^,]+)', node_text, re.IGNORECASE)
        if planning_match:
            filtered_parts.append(f"Planning area: {planning_match.group(1).strip()}")
        else:
            # Try Neighborhood (New York format)
            neighborhood_match = re.search(r'Neighborhood:\s*([^,]+)', node_text, re.IGNORECASE)
            if neighborhood_match:
                filtered_parts.append(f"Neighborhood: {neighborhood_match.group(1).strip()}")
        
        # Extract District (Singapore) or Borough (New York)
        district_match = re.search(r'District:\s*([^,]+)', node_text, re.IGNORECASE)
        if district_match:
            filtered_parts.append(f"District: {district_match.group(1).strip()}")
        else:
            # Try Borough (New York format)
            borough_match = re.search(r'Borough:\s*([^,]+)', node_text, re.IGNORECASE)
            if borough_match:
                filtered_parts.append(f"Borough: {borough_match.group(1).strip()}")
        
        # Build filtered text
        if filtered_parts:
            filtered_text = ", ".join(filtered_parts)
        else:
            # Fallback if no attributes found
            filtered_text = f"Node {node_id}"
        
        # Sanitize during inference (prevent data leakage)
        if not self.training and central_node_id is not None:
            if str(node_id) == str(central_node_id):
                # Remove category from central node only
                filtered_text = re.sub(r',?\s*Category:\s*[^,]+,?\s*', '', filtered_text, flags=re.IGNORECASE)
                filtered_text = re.sub(r',\s*,', ',', filtered_text)
                filtered_text = re.sub(r'^,\s*|,\s*$', '', filtered_text)  # Remove leading/trailing commas
                filtered_text = re.sub(r'\s+', ' ', filtered_text).strip()
                if not filtered_text:
                    filtered_text = f"Location {node_id}"
        
        return filtered_text, coords

    def embed_node_text(self, graph):
        """
        Embed node_text strings from the graph using Qwen2VL encoder.
        
        Uses node_text as-is (no parsing). Coordinates are extracted directly from
        the 'coords' node attribute. After graph cleaning, all nodes should have
        both node_text and coords attributes.
        
        Returns:
            tuple: (text_embeddings, coordinates_list) where:
            - text_embeddings: [num_nodes, hidden_dim] tensor
            - coordinates_list: List of (lon, lat) tuples or None for each node
        """
        device = next(self.parameters()).device

        # Resolve original node order to align with PyG indexing
        if hasattr(graph, 'original_id_to_idx'):
            original_ids = [node_id for node_id, _idx in sorted(graph.original_id_to_idx.items(), key=lambda x: x[1])]
        else:
            original_ids = []

        # Load underlying NetworkX graph to access per-node 'node_text' and 'coords'
        nx_graph = None
        if hasattr(graph, 'nx_graph'):
            nx_graph = graph.nx_graph
        elif hasattr(graph, 'source_file'):
            # try:
            import pickle
            with open(graph.source_file, 'rb') as f:
                nx_graph = pickle.load(f)
            # except Exception as e:
            #     logger.warning(f"Failed to load source graph: {e}")

        central_node_id = getattr(graph, 'central_node_id', None)
        
        # Check if we need to clean Singapore strings from node_text
        # This is set via environment variable from eval_urban_ranking_multiview.py
        # when evaluating beijing/paris datasets that may have graphs with Singapore data
        import os
        clean_singapore = os.environ.get("CLEAN_SINGAPORE_FROM_GRAPHS") == "1"
        
        # Build node_text list and extract coordinates
        node_texts = []
        coordinates_list = []
        _COORD_RE = re.compile(
            r"Coordinates:\s*\(\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)\s*\)",
            re.IGNORECASE
        )
        
        for node_id in original_ids:
            raw_text = None
            coords = None
            
            if nx_graph is not None and hasattr(nx_graph, 'nodes') and nx_graph.has_node(node_id):
                attrs = nx_graph.nodes[node_id]
                
                # Get coords directly from node attributes (required after graph cleaning)
                # Format: (lon, lat) tuple as stored by clean_graph_attributes.py
                coords_attr = attrs.get('coords', None)
                # logger.info(f"coords_attr: {coords_attr}")
                # print('coords_attr: ', coords_attr)
                if coords_attr == None:
                    node_text = attrs.get("node_text", "") or ""
                    m = _COORD_RE.search(node_text)
                    if m:
                        # print('m: ', m)
                        # try:
                        lon = float(m.group(1))
                        lat = float(m.group(2))
                else:
                    lon = float(coords_attr[0])
                    lat = float(coords_attr[1])
                    coords = (lon, lat)
                    # except (ValueError, TypeError, IndexError):
                    #     coords = None
                # else:
                #     coords = None
                
                # Get node_text (use as-is, no parsing needed)
                raw_text = attrs.get('node_text', None)
                if isinstance(raw_text, str):
                    raw_text = raw_text.strip()
                    
                    # Clean Singapore strings if requested (for beijing/paris datasets)
                    if clean_singapore:
                        # Remove word "Singapore" with word boundaries (case-insensitive)
                        raw_text = re.sub(r'\bSingapore\b', '', raw_text, flags=re.IGNORECASE)
                        # Clean up multiple spaces
                        raw_text = re.sub(r'\s+', ' ', raw_text)
                        # Clean up comma patterns
                        raw_text = re.sub(r',\s*,', ',', raw_text)  # Remove ", ," or ",,"
                        raw_text = re.sub(r',\s+$', '', raw_text)   # Remove trailing ", "
                        raw_text = re.sub(r'^\s+,', '', raw_text)   # Remove leading ", "
                        raw_text = re.sub(r',\s*$', '', raw_text)   # Remove trailing comma
                        raw_text = raw_text.strip()
                else:
                    raw_text = None
            
            # Fallback if no node_text found
            # if raw_text is None:
            #     raw_text = f"Node {node_id}"
            
            # Sanitize central node text during inference (minimal processing)
            # logger.info(f"graph_encoder_spatial.py: raw_text: {raw_text}")
            processed_text = raw_text
            if not self.training and central_node_id is not None and str(node_id) == str(central_node_id):
                # Remove category info from central node during inference
                processed_text = re.sub(r',?\s*Category:\s*[^,]+,?\s*', '', processed_text, flags=re.IGNORECASE)
                processed_text = re.sub(r',\s*,', ',', processed_text)
                processed_text = re.sub(r'^,\s*|,\s*$', '', processed_text)
                processed_text = re.sub(r'\s+', ' ', processed_text).strip()
                if not processed_text:
                    processed_text = f"Location {node_id}"
            
            node_texts.append(processed_text)
            coordinates_list.append(coords)  # None if coords not found (should not happen with cleaned graphs)

        # Encode node_text using Qwen2VL
        embeddings = self.node_encoder(node_texts)  # [num_nodes, hidden_dim=1536]
        # logger.info(f"graph_encoder_spatial.py: coordinates_list: {coordinates_list[:2]}")
        return embeddings, coordinates_list

    def forward(self, batch_graphs):
        """
        Process batch of graphs with spatial features.
        
        Coordinates are extracted directly from node 'coords' attribute.
        Node text is used as-is without parsing (graphs should be cleaned and standardized).
        
        Returns:
            List of [num_nodes, output_dim] tensors
        """
        device = next(self.convs[0].parameters()).device
        batch_node_embeddings = []

        for graph in batch_graphs:
            # 1. Get text embeddings and parsed coordinates
            text_embeddings, coordinates_list = self.embed_node_text(graph)  # [num_nodes, 1536], List[(lon, lat)]
            text_embeddings = text_embeddings.to(device)
            # logger.info(f"coordinates_list: {coordinates_list}")
            # 2. Convert coordinates to tensor (format: [lon, lat] as stored in graph nodes)
            # Coords are already in (lon, lat) format from node attributes
            coords_tensor = torch.tensor(
                [[c[0], c[1]] if c is not None else [0.0, 0.0] for c in coordinates_list],
                dtype=torch.float32,
                device=device
            )  # [num_nodes, 2] as [lon, lat]
            # 3. Add spatial embeddings if enabled
            if self.use_spatial_encoding:
                # logger.info(f"graph_encoder_spatial.py: use_spatial_encoding: {self.use_spatial_encoding}")
                # if coords_tensor is not None and coords_tensor.size(0) > 0:
                    # try:
                spatial_embeddings = self.spatial_encoder(coords_tensor)  # [num_nodes, spatial_embed_dim]
                
                # Concatenate text + spatial
                combined = torch.cat([text_embeddings, spatial_embeddings], dim=-1)
                x = self.node_projection(combined)  # [num_nodes, hidden_dim]
                # logger.info(f"graph_encoder_spatial.py: x after node_projection using use_spatial_encoding: {x[:10]}")
                    # except Exception as e:
                    #     logger.warning(f"Failed to encode spatial features: {e}, using text only")
                    #     x = text_embeddings
                # else:
                #     logger.warning(f"Spatial encoding enabled but no valid coordinates found in node_text! node_id: {graph.central_node_id}")
                #     x = text_embeddings
            else:
                x = text_embeddings
                # logger.info(f"graph_encoder_spatial.py: node_embeddings: {x[:10]}")
            
            # 4. Get edge index
            edge_index = graph.edge_index.to(device)
            # logger.info(f"graph_encoder_spatial.py: edge_index : {edge_index}")
            # 5. Compute edge features if enabled
            edge_attr = None
            if self.use_edge_features: # and coords_tensor is not None:
                # try:
                source_coords = coords_tensor[edge_index[0]]  # [num_edges, 2]
                target_coords = coords_tensor[edge_index[1]]  # [num_edges, 2]
                edge_attr = self.edge_encoder(source_coords, target_coords)  # [num_edges, edge_dim]
                # except Exception as e:
                #     logger.warning(f"Failed to encode edge features: {e}, using no edge features")
                #     edge_attr = None
            
            # 6. Apply GNN layers
            for i, (conv, norm) in enumerate(zip(self.convs, self.layer_norms)):
                # Check if conv supports edge_attr
                if edge_attr is not None and hasattr(conv, 'edge_dim'):
                    x_new = conv(x, edge_index, edge_attr=edge_attr)
                else:
                    x_new = conv(x, edge_index)
                
                x_new = norm(x_new)
                
                if i < len(self.convs) - 1:
                    x_new = F.relu(x_new)
                    x = x_new + x  # Residual connection
                else:
                    x = x_new
            # logger.info(f"graph_encoder_spatial.py: after GNN layers x shape: {x.shape}")
            # 7. Output projection
            x = self.output_proj(x)  # [num_nodes, output_dim]
            # logger.info(f"graph_encoder_spatial.py: after output projection x shape: {x.shape}")
            batch_node_embeddings.append(x)
            # logger.info(f"graph_encoder_spatial.py : x after output projection using edge_features: {x[:10]}")
            
            return batch_node_embeddings

    def get_trainable_parameters(self):
        """Get parameters that should be trained"""
        trainable_params = []

        # Always trainable
        trainable_params.extend(self.node_encoder.node_head.parameters())
        trainable_params.extend(self.convs.parameters())
        trainable_params.extend(self.layer_norms.parameters()) 
        trainable_params.extend(self.output_proj.parameters())
        
        # Spatial modules
        if self.use_spatial_encoding:
            trainable_params.extend(self.spatial_encoder.parameters())
        if self.use_edge_features:
            trainable_params.extend(self.edge_encoder.parameters())
        
        total_params = sum(p.numel() for p in trainable_params)
        logger.info(f"GraphEncoder (Spatial) trainable parameters: {total_params:,}")
        return trainable_params


# Backward compatibility: alias to match original name
TextAttributedGraphEncoder = TextAttributedGraphEncoderSpatial


def setup_graph_encoder_training(graph_encoder, learning_rate=1e-4, phase="frozen"):
    """Setup optimizer for graph encoder components"""
    graph_encoder.set_training_phase(phase)

    trainable_params = graph_encoder.get_trainable_parameters()
    
    if len(trainable_params) == 0:
        logger.warning("No trainable parameters found in graph encoder!")
        return None
    
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=learning_rate,
        weight_decay=1e-2
    )
    
    logger.info(f"Graph encoder optimizer setup: {phase} phase, lr={learning_rate}")
    
    return optimizer





