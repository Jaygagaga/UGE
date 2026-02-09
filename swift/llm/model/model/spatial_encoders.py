"""
Spatial Encoding Modules for Geographic Graph Neural Networks

Based on:
1. PE-GNN (Positional Encoder GNN): Multi-frequency sinusoidal encoding
2. GeoGNN: Geodesic-based edge features
3. GeoToken: S2 hierarchical token prediction for geolocalization

Author: Enhanced for MS-SWIFT spatial reasoning tasks
Date: 2025-11
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from swift.utils import get_logger

try:
    from s2sphere import LatLng, CellId
    S2_AVAILABLE = True
except ImportError:
    S2_AVAILABLE = False
    import warnings
    warnings.warn("s2sphere not available. S2 token functions will not work. Install with: pip install s2sphere")

logger = get_logger()


class SpatialPositionalEncoder(nn.Module):
    """
    Encode geographic coordinates (lat, lon) using multi-frequency sinusoidal functions.
    
    Based on PE-GNN's GridCellSpatialRelationEncoder with adaptations for:
    - Geographic coordinates (latitude/longitude in degrees)
    - Integration with Qwen2VL embeddings
    - Learnable projection for task-specific adaptation
    
    Args:
        spa_embed_dim: Output spatial embedding dimension
        coord_dim: Input coordinate dimensions (2 for lat/lon)
        frequency_num: Number of frequency bands (default: 16)
        max_radius: Maximum spatial scale in degrees (default: 180° for global)
        min_radius: Minimum spatial scale in degrees (default: 1e-4°, ~11 meters)
        use_ffn: Whether to use learnable FFN projection (recommended)
    
    Example:
        >>> encoder = SpatialPositionalEncoder(spa_embed_dim=128, frequency_num=16)
        >>> coords = torch.tensor([[1.3521, 103.8198], [1.2897, 103.8501]])  # Singapore
        >>> spatial_embeds = encoder(coords)  # [2, 128]
    """
    
    def __init__(
        self,
        spa_embed_dim: int = 128,
        coord_dim: int = 2,
        frequency_num: int = 16,
        max_radius: float = 180.0,
        min_radius: float = 1e-4,
        use_ffn: bool = True
    ):
        super().__init__()
        self.spa_embed_dim = spa_embed_dim
        self.coord_dim = coord_dim
        self.frequency_num = frequency_num
        self.max_radius = max_radius
        self.min_radius = min_radius
        self.use_ffn = use_ffn
        
        # Calculate frequency list (geometric progression for multi-scale encoding)
        self.freq_list = self._cal_freq_list()
        
        # Frequency matrix for broadcasting: [frequency_num, 2] for (sin, cos)
        self.freq_mat = np.repeat(
            np.expand_dims(self.freq_list, axis=1), 
            2,  # For sin and cos
            axis=1
        )
        
        # Input dimension: coord_dim * frequency_num * 2 (sin + cos)
        # Example: 2 coords * 16 frequencies * 2 = 64 dimensions
        self.input_dim = coord_dim * frequency_num * 2
        
        # Learnable projection (recommended for task adaptation)
        if self.use_ffn:
            self.projection = nn.Sequential(
                nn.Linear(self.input_dim, spa_embed_dim * 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(spa_embed_dim * 2, spa_embed_dim),
                nn.LayerNorm(spa_embed_dim)
            )
        else:
            # Simple linear projection if no FFN
            self.projection = nn.Linear(self.input_dim, spa_embed_dim)
    
    def _cal_freq_list(self) -> np.ndarray:
        """
        Calculate frequency list using geometric progression.
        
        Geometric progression ensures frequencies cover multiple spatial scales:
        - Low frequencies: capture global patterns (country/city level)
        - High frequencies: capture local patterns (neighborhood/street level)
        
        Returns:
            freq_list: [frequency_num] array of frequencies
        """
        log_timescale_increment = (
            np.log(float(self.max_radius) / float(self.min_radius)) / 
            (self.frequency_num - 1)
        )
        timescales = self.min_radius * np.exp(
            np.arange(self.frequency_num).astype(float) * log_timescale_increment
        )
        # Frequency = 1 / wavelength
        freq_list = 1.0 / timescales
        return freq_list
    
    def make_sinusoidal_embeds(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Create sinusoidal embeddings from coordinates.
        
        For each coordinate and each frequency:
            - Apply sin for even indices
            - Apply cos for odd indices
        
        This creates a unique, continuous embedding that preserves distances.
        
        Args:
            coords: [batch_size, coord_dim] coordinates (e.g., [lat, lon])
        
        Returns:
            embeds: [batch_size, input_dim] sinusoidal embeddings
        """
        device = coords.device
        coords_np = coords.cpu().numpy()
        batch_size = coords_np.shape[0]
        
        # Expand coordinates for broadcasting with frequencies
        # [batch_size, coord_dim, 1, 1]
        coords_expanded = np.expand_dims(coords_np, axis=2)
        coords_expanded = np.expand_dims(coords_expanded, axis=3)
        
        # Repeat for all frequencies: [batch_size, coord_dim, frequency_num, 1]
        coords_expanded = np.repeat(coords_expanded, self.frequency_num, axis=2)
        
        # Repeat for sin and cos: [batch_size, coord_dim, frequency_num, 2]
        coords_expanded = np.repeat(coords_expanded, 2, axis=3)
        
        # Apply frequency scaling: multiply by frequency matrix
        spr_embeds = coords_expanded * self.freq_mat  # Broadcasting
        
        # Apply sinusoidal functions
        # sin for even indices (0, 2, 4, ...), cos for odd indices (1, 3, 5, ...)
        spr_embeds[:, :, :, 0::2] = np.sin(spr_embeds[:, :, :, 0::2])
        spr_embeds[:, :, :, 1::2] = np.cos(spr_embeds[:, :, :, 1::2])
        
        # Flatten to [batch_size, coord_dim * frequency_num * 2]
        spr_embeds = spr_embeds.reshape(batch_size, -1)
        
        # Convert back to tensor
        spr_embeds = torch.FloatTensor(spr_embeds).to(device)
        
        return spr_embeds
    
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Encode coordinates into spatial embeddings.
        
        Args:
            coords: [num_nodes, coord_dim] tensor of coordinates
                    For geographic data: [num_nodes, 2] where each row is [lon, lat]
                    Note: Input format is [lon, lat] to match graph_encoder_spatial.py
        
        Returns:
            spatial_embeddings: [num_nodes, spa_embed_dim] encoded features
        """
        # Create sinusoidal embeddings
        # print(f"coords: {coords}")
        sinusoidal_embeds = self.make_sinusoidal_embeds(coords)
        # print(f"sinusoidal_embeds: {sinusoidal_embeds}")
        # Project to output dimension
        spatial_embeddings = self.projection(sinusoidal_embeds)
        
        return spatial_embeddings


class GeodesicEdgeEncoder(nn.Module):
    """
    Encode edge features based on geodesic relationships between nodes.
    
    Inspired by GeoGNN's edge feature design. Captures:
    1. Distance: How far apart are the nodes? (haversine distance)
    2. Direction: What direction from source to target? (bearing angle)
    3. Displacement: Relative position in lat/lon space (Δlat, Δlon)
    
    These features provide rich spatial context for graph message passing.
    
    Args:
        edge_embed_dim: Output edge embedding dimension
        distance_units: 'meters' or 'kilometers' for distance encoding
        use_log_distance: Apply log transform to distances (recommended)
    
    Example:
        >>> encoder = GeodesicEdgeEncoder(edge_embed_dim=64)
        >>> src = torch.tensor([[1.35, 103.82], [1.29, 103.85]])  # Singapore locations
        >>> tgt = torch.tensor([[1.30, 103.83], [1.36, 103.86]])
        >>> edge_feats = encoder(src, tgt)  # [2, 64]
    """
    
    def __init__(
        self,
        edge_embed_dim: int = 64,
        distance_units: str = 'meters',
        use_log_distance: bool = True,
        use_distance: bool = True,
        use_direction: bool = True,
        use_displacement: bool = True
    ):
        super().__init__()
        self.edge_embed_dim = edge_embed_dim
        self.distance_units = distance_units
        self.use_log_distance = use_log_distance
        self.use_distance = use_distance
        self.use_direction = use_direction
        self.use_displacement = use_displacement
        
        if not (self.use_distance or self.use_direction or self.use_displacement):
            raise ValueError("At least one edge feature component must be enabled.")
        
        # Distance encoder: handles varying distance scales
        if self.use_distance:
            self.distance_encoder = nn.Sequential(
                nn.Linear(1, edge_embed_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(edge_embed_dim, edge_embed_dim)
            )
        else:
            self.distance_encoder = None
        
        # Direction encoder: encodes bearing as [sin(θ), cos(θ)]
        if self.use_direction:
            self.direction_encoder = nn.Sequential(
                nn.Linear(2, edge_embed_dim),  # Input: [sin(bearing), cos(bearing)]
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(edge_embed_dim, edge_embed_dim)
            )
        else:
            self.direction_encoder = None
        
        # Displacement encoder: encodes relative position (Δlat, Δlon)
        if self.use_displacement:
            self.displacement_encoder = nn.Sequential(
                nn.Linear(2, edge_embed_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(edge_embed_dim, edge_embed_dim)
            )
        else:
            self.displacement_encoder = None
        
        # Combine enabled edge features
        enabled_components = (
            int(self.use_distance) + int(self.use_direction) + int(self.use_displacement)
        )
        combiner_input_dim = edge_embed_dim * enabled_components
        self.edge_combiner = nn.Sequential(
            nn.Linear(combiner_input_dim, edge_embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(edge_embed_dim * 2, edge_embed_dim),
            nn.LayerNorm(edge_embed_dim)
        )
    
    def haversine_distance(
        self, 
        lon1: torch.Tensor, 
        lat1: torch.Tensor,
        lon2: torch.Tensor, 
        lat2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute haversine distance (great circle distance on Earth's surface).
        
        More accurate than Euclidean distance for geographic coordinates.
        
        Args:
            lon1, lat1: Source coordinates in degrees
            lon2, lat2: Target coordinates in degrees
        
        Returns:
            distances: Great circle distances in meters (or km if distance_units='kilometers')
        """
        # Convert degrees to radians
        lon1_rad = lon1 * np.pi / 180.0
        lat1_rad = lat1 * np.pi / 180.0
        lon2_rad = lon2 * np.pi / 180.0
        lat2_rad = lat2 * np.pi / 180.0
        
        # Haversine formula
        dlon = lon2_rad - lon1_rad
        dlat = lat2_rad - lat1_rad
        
        a = torch.sin(dlat/2)**2 + \
            torch.cos(lat1_rad) * torch.cos(lat2_rad) * torch.sin(dlon/2)**2
        # Use atan2 for better numerical stability (especially for large distances)
        # This is more stable than arcsin when a is close to 1
        c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a + 1e-8))
        
        # Earth's radius
        r = 6371000 if self.distance_units == 'meters' else 6371  # meters or km
        
        distances = c * r
        return distances
    
    def compute_bearing(
        self,
        lon1: torch.Tensor,
        lat1: torch.Tensor,
        lon2: torch.Tensor,
        lat2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute bearing (direction) from point 1 to point 2.
        
        Bearing is the angle (in radians) measured clockwise from north.
        Range: [-π, π]
        
        Uses the standard spherical trigonometry formula:
        y = sin(Δlon) * cos(lat2)
        x = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(Δlon)
        bearing = atan2(y, x)
        
        Args:
            lon1, lat1: Source coordinates in degrees [lon, lat]
            lon2, lat2: Target coordinates in degrees [lon, lat]
        
        Returns:
            bearings: Bearing angles in radians [-π, π]
        """
        # Convert to radians
        lat1_rad = lat1 * np.pi / 180.0
        lon1_rad = lon1 * np.pi / 180.0
        lat2_rad = lat2 * np.pi / 180.0
        lon2_rad = lon2 * np.pi / 180.0
        
        dlon = lon2_rad - lon1_rad
        
        # Standard bearing formula (spherical trigonometry)
        y = torch.sin(dlon) * torch.cos(lat2_rad)
        x = torch.cos(lat1_rad) * torch.sin(lat2_rad) - \
            torch.sin(lat1_rad) * torch.cos(lat2_rad) * torch.cos(dlon)
        
        bearing = torch.atan2(y, x)  # Returns angle in [-π, π]
        
        return bearing
    
    def forward(
        self,
        source_coords: torch.Tensor,
        target_coords: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode spatial relationships between source and target nodes.
        
        Args:
            source_coords: [num_edges, 2] coordinates of source nodes [lon, lat]
            target_coords: [num_edges, 2] coordinates of target nodes [lon, lat]
        
        Returns:
            edge_features: [num_edges, edge_embed_dim] encoded edge features
        """
        device = source_coords.device
        
        # Extract lon/lat (standardized format: [lon, lat])
        src_lon, src_lat = source_coords[:, 0], source_coords[:, 1]
        tgt_lon, tgt_lat = target_coords[:, 0], target_coords[:, 1]
        
        features = []
        
        if self.use_distance:
            distances = self.haversine_distance(src_lon, src_lat, tgt_lon, tgt_lat)
            # logger.info(f"spatial_encoders.py: distances: {distances,src_lon, src_lat, tgt_lon, tgt_lat}")
            # print(f"distances: {distances}")
            if self.use_log_distance:
                distances = torch.log1p(distances)
            distance_feats = self.distance_encoder(distances.unsqueeze(-1))
            # print(f"distance_feats: {distance_feats}")
            features.append(distance_feats)
        
        if self.use_direction:
            bearings = self.compute_bearing(src_lon, src_lat, tgt_lon, tgt_lat)
            # print(f"bearings: {bearings}")
            # logger.info(f"spatial_encoders.py: bearings: {bearings,src_lon, src_lat, tgt_lon, tgt_lat}")
            direction_input = torch.stack([
                torch.sin(bearings),
                torch.cos(bearings)
            ], dim=-1)
            # print(f"direction_input: {direction_input}")
            direction_feats = self.direction_encoder(direction_input)
            # print(f"direction_feats: {direction_feats}")
            features.append(direction_feats)
        
        if self.use_displacement:
            displacement = target_coords - source_coords
            displacement_feats = self.displacement_encoder(displacement)
            # print(f"displacement_feats: {displacement_feats}")
            features.append(displacement_feats)
        
        combined_feats = torch.cat(features, dim=-1)
        
        # Final projection and normalization
        edge_features = self.edge_combiner(combined_feats)
        
        return edge_features


# ===== S2 Token Utilities (from GeoToken) =====

def latlng_to_s2_tokens(lat: float, lng: float, level: int) -> List[int]:
    """
    Convert a (lat, lng) pair to a full S2 token sequence.
    
    Based on GeoToken paper: https://arxiv.org/pdf/2511.01082
    
    Args:
        lat: Latitude in degrees
        lng: Longitude in degrees
        level: S2 level (e.g., 20 for fine-grained, 10 for coarse)
    
    Returns:
        List of length (level+1):
        - Token 0: Cube face (an integer in [0,5])
        - Tokens 1..level: Each token is in [0,3] representing the child index at that level.
    """
    if not S2_AVAILABLE:
        raise ImportError("s2sphere is required for S2 token functions. Install with: pip install s2sphere")
    
    latlng = LatLng.from_degrees(lat, lng)
    cell = CellId.from_lat_lng(latlng).parent(level)
    face = cell.face()
    tokens = [face]
    cell_id_int = cell.id()
    # Extract the 2*k child tokens.
    # The Hilbert position is encoded in the next 2*level bits.
    shift = 61 - (2 * level)
    pos_bits = (cell_id_int >> shift) & ((1 << (2 * level)) - 1)
    for i in range(level):
        shift_i = 2 * (level - i - 1)
        token = (pos_bits >> shift_i) & 0x3
        tokens.append(token)
    return tokens


def group_s2_tokens(full_tokens: List[int], group_size: int = 2) -> List[int]:
    """
    Group the child tokens into groups of size `group_size`. The first token (face) remains unchanged.
    
    Args:
        full_tokens: Full S2 token sequence (length = level + 1)
        group_size: Number of tokens to group together (default: 2)
    
    Returns:
        Grouped token sequence: [face_token, grouped_token_1, grouped_token_2, ...]
        Length: 1 + (level // group_size)
    """
    if len(full_tokens) < 1 or (len(full_tokens) - 1) % group_size != 0:
        raise ValueError(f"Full token sequence length minus one ({len(full_tokens)-1}) must be divisible by group_size ({group_size})")
    grouped = [full_tokens[0]]  # Keep face token as-is
    for i in range(1, len(full_tokens), group_size):
        group = full_tokens[i:i+group_size]
        val = 0
        for digit in group:
            val = val * 4 + digit
        grouped.append(val)
    return grouped


def ungroup_s2_tokens(grouped_tokens: List[int], group_size: int = 2) -> List[int]:
    """
    Reconstruct the full token sequence from the grouped token sequence.
    
    Args:
        grouped_tokens: Grouped token sequence
        group_size: Number of tokens per group
    
    Returns:
        Full token sequence (length = level + 1)
    """
    full = [grouped_tokens[0]]  # Face token
    for val in grouped_tokens[1:]:
        group = []
        for i in range(group_size):
            digit = (val // (4 ** (group_size - i - 1))) % 4
            group.append(digit)
        full.extend(group)
    return full


class S2TokenHead(nn.Module):
    """
    Auxiliary head for predicting S2 hierarchical tokens from node embeddings.
    
    Based on GeoToken paper: https://arxiv.org/pdf/2511.01082
    Predicts grouped S2 tokens for each node, enabling hierarchical geolocalization.
    
    Args:
        hidden_dim: Input embedding dimension from graph encoder
        s2_level: S2 level for tokenization (e.g., 20 for fine-grained)
        group_size: Number of tokens to group together (default: 2)
        dropout: Dropout rate for regularization
    
    Example:
        >>> head = S2TokenHead(hidden_dim=1536, s2_level=20, group_size=2)
        >>> node_embeds = torch.randn(100, 1536)  # From graph encoder
        >>> logits_list = head(node_embeds)  # List of [100, vocab_size] tensors
    """
    
    def __init__(
        self, 
        hidden_dim: int, 
        s2_level: int = 20,
        group_size: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.s2_level = s2_level
        self.group_size = group_size
        
        # Compute sequence length: 1 (face) + (level // group_size) grouped tokens
        if s2_level % group_size != 0:
            raise ValueError(f"s2_level ({s2_level}) must be divisible by group_size ({group_size})")
        self.seq_length = 1 + (s2_level // group_size)
        
        # Vocabulary sizes:
        # Position 0 (face): 6 possibilities (cube faces)
        # Positions 1..seq_length-1: 4^group_size possibilities each
        self.vocab_sizes = [6] + [4 ** group_size] * (self.seq_length - 1)
        
        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Separate output heads for each token position
        self.output_heads = nn.ModuleList([
            nn.Linear(hidden_dim // 4, vocab_size)
            for vocab_size in self.vocab_sizes
        ])
    
    def forward(self, node_embeddings: torch.Tensor) -> List[torch.Tensor]:
        """
        Predict S2 tokens from node embeddings.
        
        Args:
            node_embeddings: [num_nodes, hidden_dim]
        
        Returns:
            List of logits tensors, one per token position:
            - logits[0]: [num_nodes, 6] for face token
            - logits[1:]: [num_nodes, 4^group_size] for grouped tokens
        """
        features = self.feature_extractor(node_embeddings)  # [num_nodes, hidden_dim//4]
        logits_list = [head(features) for head in self.output_heads]
        return logits_list


class SpatialAutocorrelationHead(nn.Module):
    """
    Auxiliary head for predicting local Moran's I (spatial autocorrelation).
    
    Moran's I measures whether a value at a location is similar to values at
    neighboring locations. This auxiliary task encourages the model to learn
    spatially smooth representations.
    
    Usage in training:
        1. Compute true local Moran's I from ground truth values
        2. Predict Moran's I from learned node embeddings
        3. Add MSE loss between predicted and true Moran's I
    
    Args:
        hidden_dim: Input embedding dimension from graph encoder
        dropout: Dropout rate for regularization
    
    Example:
        >>> head = SpatialAutocorrelationHead(hidden_dim=1536)
        >>> node_embeds = torch.randn(100, 1536)  # From graph encoder
        >>> morans_pred = head(node_embeds)  # [100, 1]
    """
    
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1)
        )
    
    def forward(self, node_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Predict local Moran's I from node embeddings.
        
        Args:
            node_embeddings: [num_nodes, hidden_dim]
        
        Returns:
            morans_pred: [num_nodes, 1] predicted local Moran's I values
        """
        return self.predictor(node_embeddings)
    
    @staticmethod
    def compute_local_morans(
        values: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute local Moran's I statistic for ground truth values.
        
        PE-GNN formula: mi_i = (n-1) * z_i * (Σ_j w_ij * z_j) / Σ_k(z_k^2)
        where:
        - z_i = standardized value at location i: z = (y - mean) / std
        - w_ij = spatial weight between i and j (e.g., inverse distance)
        - Σ_j(w_ij * z_j) = spatially lagged value (weighted sum of neighbors' z-scores)
        - Σ_k(z_k^2) = sum of squared z-scores (normalization factor)
        - (n-1) = scaling factor
        
        Positive values indicate spatial clustering (similar values nearby).
        Negative values indicate spatial dispersion (dissimilar values nearby).
        
        Reference: PE-GNN spatial-utils.py lw_tensor_local_moran()
        
        Args:
            values: [num_nodes] target values (e.g., perception scores)
            edge_index: [2, num_edges] graph connectivity
            edge_weights: [num_edges] optional spatial weights (default: uniform)
        
        Returns:
            local_morans: [num_nodes] local Moran's I for each node
        """
        num_nodes = values.size(0)
        device = values.device
        n_1 = num_nodes - 1  # PE-GNN: n-1 scaling factor
        
        # 1. Standardize values: z = (y - mean) / std
        mean_val = values.mean()
        std_val = values.std()
        z = (values - mean_val) / (std_val + 1e-8)
        
        # 2. Calculate denominator: sum of squared z-scores
        # PE-GNN: den = (z * z).sum()
        den = (z * z).sum()
        
        # 3. Default to uniform weights if not provided
        if edge_weights is None:
            edge_weights = torch.ones(edge_index.size(1), device=device)
        
        # 4. Compute spatially lagged values: zl = Σ_j(w_ij * z_j)
        # This is the weighted sum of neighbors' z-scores
        zl = torch.zeros(num_nodes, device=device)
        
        # Accumulate weighted neighbor z-scores
        for i in range(edge_index.size(1)):
            src_idx = edge_index[0, i]
            tgt_idx = edge_index[1, i]
            weight = edge_weights[i]
            
            # Add weighted neighbor's z-score
            zl[src_idx] += z[tgt_idx] * weight
        
        # 5. Local Moran's I formula (PE-GNN):
        # mi_i = (n-1) * z_i * zl_i / sum(z^2)
        local_morans = n_1 * z * zl / den
        
        # 6. Handle NaN values (e.g., isolated nodes)
        local_morans = torch.nan_to_num(local_morans, nan=0.0)
        
        return local_morans


def compute_spatial_auxiliary_loss(
    node_embeddings: torch.Tensor,
    target_values: torch.Tensor,
    edge_index: torch.Tensor,
    coords: torch.Tensor,
    morans_head: SpatialAutocorrelationHead,
    distance_units: str = 'meters'
) -> Tuple[torch.Tensor, dict]:
    """
    Compute auxiliary loss for spatial autocorrelation prediction.
    
    This function:
    1. Computes distance-based edge weights
    2. Calculates true local Moran's I from target values
    3. Predicts Moran's I from learned embeddings
    4. Returns MSE loss between predicted and true values
    
    Args:
        node_embeddings: [num_nodes, hidden_dim] from graph encoder
        target_values: [num_nodes] ground truth (e.g., safety scores)
        edge_index: [2, num_edges] graph connectivity
        coords: [num_nodes, 2] node coordinates [lon, lat]
        morans_head: SpatialAutocorrelationHead module
        distance_units: 'meters' or 'kilometers'
    
    Returns:
        loss: Scalar MSE loss
        metrics: Dict with detailed metrics for logging
    """
    device = node_embeddings.device
    
    # Compute edge weights based on distance (inverse distance weighting)
    source_coords = coords[edge_index[0]]
    target_coords = coords[edge_index[1]]
    
    # Haversine distance (coords format: [lon, lat])
    lon1, lat1 = source_coords[:, 0], source_coords[:, 1]
    lon2, lat2 = target_coords[:, 0], target_coords[:, 1]
    
    # Haversine distance calculation (using atan2 for numerical stability)
    lon1_rad = lon1 * np.pi / 180.0
    lat1_rad = lat1 * np.pi / 180.0
    lon2_rad = lon2 * np.pi / 180.0
    lat2_rad = lat2 * np.pi / 180.0
    
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = torch.sin(dlat/2)**2 + torch.cos(lat1_rad) * torch.cos(lat2_rad) * torch.sin(dlon/2)**2
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a + 1e-8))
    r = 6371000 if distance_units == 'meters' else 6371
    distances = c * r
    
    # PE-GNN edge weighting: edge_weight = (max_val - dist) / range
    # This gives higher weights to closer nodes (max weight = 1.0 for closest)
    max_dist = distances.max()
    min_dist = distances.min()
    dist_range = max_dist - min_dist + 1e-8  # Add epsilon to avoid division by zero
    edge_weights = (max_dist - distances) / dist_range
    
    # Compute true local Moran's I
    true_morans = SpatialAutocorrelationHead.compute_local_morans(
        target_values,
        edge_index,
        edge_weights
    )
    
    # Predict Moran's I from embeddings
    pred_morans = morans_head(node_embeddings).squeeze(-1)
    
    # MSE loss
    loss = F.mse_loss(pred_morans, true_morans)
    
    # Compute metrics for logging
    with torch.no_grad():
        correlation = torch.corrcoef(
            torch.stack([pred_morans, true_morans])
        )[0, 1]
        mae = (pred_morans - true_morans).abs().mean()
    
    metrics = {
        'spatial_loss': loss.item(),
        'morans_correlation': correlation.item(),
        'morans_mae': mae.item()
    }
    
    return loss, metrics


def compute_s2_auxiliary_loss(
    node_embeddings: torch.Tensor,
    coords: torch.Tensor,
    s2_token_head: S2TokenHead,
    s2_level: int = 20,
    group_size: int = 2,
    ignore_index: int = -100
) -> Tuple[torch.Tensor, dict]:
    """
    Compute auxiliary loss for S2 hierarchical token prediction.
    
    Based on GeoToken paper: https://arxiv.org/pdf/2511.01082
    
    This function:
    1. Generates ground truth S2 tokens from node coordinates
    2. Predicts S2 tokens from learned node embeddings
    3. Returns cross-entropy loss for each token position
    
    Args:
        node_embeddings: [num_nodes, hidden_dim] from graph encoder
        coords: [num_nodes, 2] node coordinates [lat, lon]
        s2_token_head: S2TokenHead module
        s2_level: S2 level for tokenization (must match head)
        group_size: Grouping size (must match head)
        ignore_index: Index to ignore in loss computation
    
    Returns:
        loss: Scalar cross-entropy loss (averaged over all positions)
        metrics: Dict with detailed metrics for logging
    """
    if not S2_AVAILABLE:
        raise ImportError("s2sphere is required for S2 token functions. Install with: pip install s2sphere")
    
    device = node_embeddings.device
    num_nodes = node_embeddings.size(0)
    
    # 1. Generate ground truth S2 tokens for each node
    target_tokens_list = []  # List of [seq_length] tensors, one per node
    valid_nodes = []
    
    for i in range(num_nodes):
        lat, lon = coords[i, 0].item(), coords[i, 1].item()
        # try:
            # Convert coordinates to S2 tokens
        full_tokens = latlng_to_s2_tokens(lat, lon, s2_level)
        grouped_tokens = group_s2_tokens(full_tokens, group_size=group_size)
        target_tokens_list.append(torch.tensor(grouped_tokens, dtype=torch.long, device=device))
        valid_nodes.append(i)
        # logger.info(f"[DEBUG] compute_s2_auxiliary_loss: full_tokens: {full_tokens}")
        # logger.info(f"[DEBUG] compute_s2_auxiliary_loss: grouped_tokens: {grouped_tokens}")
        # except Exception as e:
        #     # Skip invalid coordinates
        #     continue
    
    if len(valid_nodes) == 0:
        # Return zero loss if no valid coordinates
        return torch.tensor(0.0, device=device, requires_grad=True), {
            's2_loss': 0.0,
            's2_accuracy': 0.0,
            's2_valid_nodes': 0
        }
    
    # Stack target tokens: [num_valid_nodes, seq_length]
    target_tokens = torch.stack(target_tokens_list)  # [num_valid_nodes, seq_length]
    
    # Get embeddings for valid nodes only
    valid_embeddings = node_embeddings[valid_nodes]  # [num_valid_nodes, hidden_dim]
    
    # 2. Predict S2 tokens from embeddings
    pred_logits_list = s2_token_head(valid_embeddings)  # List of [num_valid_nodes, vocab_size]
    
    # 3. Compute cross-entropy loss for each token position
    losses = []
    correct_predictions = []
    total_predictions = []
    
    for pos in range(len(pred_logits_list)):
        logits = pred_logits_list[pos]  # [num_valid_nodes, vocab_size]
        targets = target_tokens[:, pos]  # [num_valid_nodes]
        
        # Cross-entropy loss
        loss = F.cross_entropy(logits, targets, ignore_index=ignore_index, reduction='mean')
        losses.append(loss)
        
        # Compute accuracy
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            correct = (preds == targets).sum().item()
            correct_predictions.append(correct)
            total_predictions.append(targets.size(0))
    
    # Average loss across all token positions
    total_loss = sum(losses) / len(losses)
    
    # Compute metrics
    with torch.no_grad():
        total_correct = sum(correct_predictions)
        total_count = sum(total_predictions)
        accuracy = total_correct / total_count if total_count > 0 else 0.0
        
        # Per-position accuracy
        per_position_acc = [
            correct / total if total > 0 else 0.0
            for correct, total in zip(correct_predictions, total_predictions)
        ]
    
    metrics = {
        's2_loss': total_loss.item(),
        's2_accuracy': accuracy,
        's2_valid_nodes': len(valid_nodes),
        's2_total_nodes': num_nodes,
        's2_per_position_acc': per_position_acc,
    }
    
    return total_loss, metrics


# Example usage and testing
if __name__ == '__main__':
    print("Testing Spatial Encoding Modules...")
    
    # Test 1: Spatial Positional Encoder
    print("\n1. Testing SpatialPositionalEncoder...")
    spatial_encoder = SpatialPositionalEncoder(
        spa_embed_dim=128,
        frequency_num=16
    )
    
    # Example coordinates (Singapore locations)
    coords = torch.tensor([
        [1.3521, 103.8198],  # Marina Bay
        [1.2897, 103.8501],  # Sentosa
        [1.3644, 103.9915],  # Changi
    ])
    
    spatial_embeds = spatial_encoder(coords)
    print(f"   Input shape: {coords.shape}")
    print(f"   Output shape: {spatial_embeds.shape}")
    print(f"   ✓ Spatial encoding successful")
    
    # Test 2: Geodesic Edge Encoder
    print("\n2. Testing GeodesicEdgeEncoder...")
    edge_encoder = GeodesicEdgeEncoder(edge_embed_dim=64)
    
    source_coords = coords[:2]  # Marina Bay, Sentosa
    target_coords = coords[1:]  # Sentosa, Changi
    
    edge_feats = edge_encoder(source_coords, target_coords)
    print(f"   Source coords: {source_coords.shape}")
    print(f"   Target coords: {target_coords.shape}")
    print(f"   Output shape: {edge_feats.shape}")
    print(f"   ✓ Edge encoding successful")
    
    # Test 3: Spatial Autocorrelation
    print("\n3. Testing SpatialAutocorrelationHead...")
    morans_head = SpatialAutocorrelationHead(hidden_dim=128)
    
    # Dummy embeddings
    node_embeds = torch.randn(3, 128)
    morans_pred = morans_head(node_embeds)
    print(f"   Input shape: {node_embeds.shape}")
    print(f"   Output shape: {morans_pred.shape}")
    
    # Test Moran's I computation
    values = torch.tensor([0.8, 0.7, 0.9])  # Similar values
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])  # Ring graph
    true_morans = SpatialAutocorrelationHead.compute_local_morans(values, edge_index)
    print(f"   True Moran's I: {true_morans}")
    print(f"   ✓ Spatial autocorrelation computation successful")
    
    print("\n✅ All tests passed!")





