"""
Orthogonal Gradient Descent (OGD) for Continual Learning

This module implements a FEATURE-BASED variant of OGD to preserve learned 
image-text similarity when adding new modalities (e.g., graph) to a 
pretrained multimodal model.

IMPORTANT: This is a feature-based implementation, which differs from 
standard OGD+ that stores gradients. We store feature vectors (activations) 
from forward passes, which is more memory-efficient but may be less precise 
than gradient-based OGD.

Based on:
- Farajtabar et al. (2019). "Orthogonal Gradient Descent for Continual Learning"
- Abbana Bennani & Sugiyama (2020). "Generalisation Guarantees for Continual Learning with Orthogonal Gradient Descent"
- Standard OGD+ implementation: https://github.com/MehdiAbbanaBennani/continual-learning-ogdplus

Key Features:
- Projects gradients orthogonally to previously learned task representations
- Uses FEATURE VECTORS (activations) to build projection matrix (not gradients)
- Preserves learned image-text alignment when adding graph modality
- Memory efficient (stores feature vectors, which are smaller than gradients)
- Simple to integrate with existing training pipelines

Note: Standard OGD+ stores gradients from previous tasks. Our implementation
stores feature vectors instead, which is more memory-efficient for large models
but may be less theoretically precise.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import numpy as np
import os
import json
import csv
from swift.utils import get_logger

logger = get_logger()


class OGDMemory:
    """
    Stores feature vectors (activations) from previous tasks for gradient projection.
    
    NOTE: This is a feature-based approach, not standard gradient-based OGD.
    Standard OGD+ stores gradients, but we store features for memory efficiency.
    """
    
    def __init__(self, memory_size: int = 1000):
        self.memory_size = memory_size
        self.feature_mat: Dict[str, torch.Tensor] = {}  # layer_name -> [memory_size, feature_dim]
        self.current_task = 0
    
    def update(self, layer_name: str, features: torch.Tensor):
        """
        Update memory for a layer with new features.
        
        Args:
            layer_name: Name of the layer
            features: Feature tensor of shape [batch_size, feature_dim] or [feature_dim]
        """
        if features.numel() == 0:
            return
        
        # Flatten features if needed
        if features.dim() > 2:
            features = features.view(-1, features.shape[-1])
        elif features.dim() == 1:
            features = features.unsqueeze(0)
        
        # features shape: [num_samples, feature_dim]
        num_samples, feature_dim = features.shape
        
        if layer_name not in self.feature_mat:
            # Initialize memory matrix
            self.feature_mat[layer_name] = torch.zeros(
                self.memory_size, feature_dim,
                device=features.device,
                dtype=features.dtype
            )
            self._current_idx = {layer_name: 0}
        
        # Store features (FIFO if memory is full)
        current_idx = self._current_idx[layer_name]
        num_to_store = min(num_samples, self.memory_size - current_idx)
        
        if num_to_store > 0:
            self.feature_mat[layer_name][current_idx:current_idx + num_to_store] = features[:num_to_store]
            self._current_idx[layer_name] = current_idx + num_to_store
        
        # If we have more features than remaining space, replace oldest
        if num_samples > num_to_store:
            remaining = num_samples - num_to_store
            replace_idx = min(remaining, self.memory_size)
            self.feature_mat[layer_name][:replace_idx] = features[num_to_store:num_to_store + replace_idx]
            self._current_idx[layer_name] = replace_idx
    
    def get_projection_matrix(self, layer_name: str) -> Optional[torch.Tensor]:
        """
        Get projection matrix P for a layer.
        
        Returns:
            P: [feature_dim, memory_size] matrix, or None if no memory stored
        """
        if layer_name not in self.feature_mat:
            return None
        
        P = self.feature_mat[layer_name]  # [memory_size, feature_dim]
        current_size = self._current_idx.get(layer_name, self.memory_size)
        
        if current_size == 0:
            return None
        
        # Use only stored features
        P = P[:current_size]  # [actual_size, feature_dim]
        
        # Transpose to [feature_dim, actual_size] for projection
        return P.t()  # [feature_dim, actual_size]
    
    def get_stored_count(self, layer_name: str) -> int:
        """Get number of features stored for a layer."""
        if layer_name not in self.feature_mat:
            return 0
        return self._current_idx.get(layer_name, 0)


class OGDGradientMemory:
    """
    Stores gradients from previous tasks for gradient projection.
    
    This is the standard OGD+ approach, storing actual gradients
    instead of features. Use this for more precise gradient projection.
    
    NOTE: Gradients are much larger than features, so use smaller memory_size.
    """
    
    def __init__(self, memory_size: int = 100):
        # Smaller memory_size for gradients (they're larger)
        self.memory_size = memory_size
        # layer_name -> [memory_size, param_dim] (flattened gradients)
        self.gradient_mat: Dict[str, torch.Tensor] = {}
        self._param_shapes: Dict[str, Tuple] = {}  # Store original shapes
        self._current_idx: Dict[str, int] = {}
    
    def update(self, layer_name: str, grad: torch.Tensor):
        """
        Update memory with gradient.
        
        Args:
            layer_name: Name of the layer
            grad: Gradient tensor (any shape)
        """
        if grad is None or grad.numel() == 0:
            return
        
        # Flatten gradient to 1D
        original_shape = grad.shape
        grad_flat = grad.detach().clone().flatten()  # [param_dim]
        param_dim = grad_flat.shape[0]
        
        # Store original shape for reshaping later
        if layer_name not in self._param_shapes:
            self._param_shapes[layer_name] = original_shape
        
        # Initialize if needed
        if layer_name not in self.gradient_mat:
            self.gradient_mat[layer_name] = torch.zeros(
                self.memory_size, param_dim,
                device=grad.device,
                dtype=grad.dtype
            )
            self._current_idx[layer_name] = 0
        
        # Store gradient (FIFO if memory is full)
        current_idx = self._current_idx[layer_name]
        if current_idx < self.memory_size:
            self.gradient_mat[layer_name][current_idx] = grad_flat
            self._current_idx[layer_name] = current_idx + 1
        else:
            # FIFO: replace oldest
            replace_idx = current_idx % self.memory_size
            self.gradient_mat[layer_name][replace_idx] = grad_flat
            self._current_idx[layer_name] = replace_idx + 1
    
    def get_projection_matrix(self, layer_name: str) -> Optional[torch.Tensor]:
        """
        Get projection matrix P built from stored gradients.
        
        Returns:
            P: [param_dim, memory_size] matrix, or None if no memory stored
        """
        if layer_name not in self.gradient_mat:
            return None
        
        G = self.gradient_mat[layer_name]  # [memory_size, param_dim]
        current_size = min(self._current_idx.get(layer_name, 0), self.memory_size)
        
        if current_size == 0:
            return None
        
        # Use only stored gradients
        G = G[:current_size]  # [actual_size, param_dim]
        
        # Transpose to [param_dim, actual_size] for projection
        return G.t()  # [param_dim, actual_size]
    
    def get_stored_count(self, layer_name: str) -> int:
        """Get number of gradients stored for a layer."""
        if layer_name not in self.gradient_mat:
            return 0
        return min(self._current_idx.get(layer_name, 0), self.memory_size)


class OGDHook:
    """Forward hook to capture activations from protected layers."""
    
    def __init__(self, layer_name: str, memory: OGDMemory):
        self.layer_name = layer_name
        self.memory = memory
        self.activations = None
    
    def __call__(self, module, input, output):
        """Capture activations during forward pass."""
        # Handle different output types
        if isinstance(output, torch.Tensor):
            self.activations = output.detach()
        elif isinstance(output, tuple):
            # Use first element (usually the main output)
            self.activations = output[0].detach() if len(output) > 0 else None
        else:
            self.activations = None
        
        # Update memory if we have activations
        if self.activations is not None:
            self.memory.update(self.layer_name, self.activations)


class OGDTrainer:
    """
    OGD Trainer for continual learning.
    
    This class manages:
    1. Registering hooks on protected layers
    2. Updating memory from pretrained model
    3. Projecting gradients during training
    """
    
    def __init__(
        self,
        model: nn.Module,
        protected_layers: List[str],
        memory_size: int = 1000,
        use_ogd_plus: bool = False,
        use_gradients: bool = False,
        device: Optional[torch.device] = None
    ):
        """
        Initialize OGD Trainer.
        
        Args:
            model: The model to protect
            protected_layers: List of layer names to protect (e.g., ['visual.transformer.resblocks.23'])
            memory_size: Number of feature vectors (or gradients) to store per layer
            use_ogd_plus: Whether to use OGD+ (stores all previous tasks, not just one)
            use_gradients: If True, use gradient-based OGD (standard OGD+). If False, use feature-based (default).
            device: Device to store memory on (defaults to model device)
        """
        self.model = model
        self.protected_layers = protected_layers
        self.memory_size = memory_size
        self.use_ogd_plus = use_ogd_plus
        self.use_gradients = use_gradients
        self.device = device or next(model.parameters()).device
        
        # Choose memory type based on use_gradients flag
        if use_gradients:
            # Gradient-based OGD (standard OGD+)
            # Use smaller memory_size for gradients (they're larger)
            gradient_memory_size = min(memory_size, 100)  # Cap at 100 for gradients
            if memory_size > 100:
                logger.warning(f"[OGD] Gradient-based mode: reducing memory_size from {memory_size} to {gradient_memory_size} (gradients are larger)")
            self.memory = OGDGradientMemory(memory_size=gradient_memory_size)
            self.memory_type = "gradients"
        else:
            # Feature-based OGD (current default)
            self.memory = OGDMemory(memory_size=memory_size)
            self.memory_type = "features"
        
        self.hooks: Dict[str, torch.utils.hooks.RemovableHandle] = {}
        self._gradient_capture_hooks: Dict[str, torch.utils.hooks.RemovableHandle] = {}  # For gradient collection
        self.projection_enabled = False
        self._projection_step = 0
        self._projection_log_interval = 100
        
        # Accumulate projection stats for logging
        self._projection_stats: Dict[str, List[float]] = {}  # layer_name -> list of reduction ratios
        self._projection_count = 0  # Total number of projections performed
        
        # For saving stats to file
        self._output_dir: Optional[str] = None
        self._stats_history: List[Dict[str, Any]] = []  # Store all stats for saving
        
        print(f"[OGD] Initialized with {len(protected_layers)} protected layers", flush=True)
        print(f"[OGD] Memory type: {self.memory_type}", flush=True)
        print(f"[OGD] Memory size: {self.memory_size} {self.memory_type} per layer", flush=True)
        print(f"[OGD] OGD+ mode: {use_ogd_plus}", flush=True)
        logger.info(f"[OGD] Initialized with {len(protected_layers)} protected layers")
        logger.info(f"[OGD] Memory type: {self.memory_type}")
        logger.info(f"[OGD] Memory size: {self.memory_size} {self.memory_type} per layer")
        logger.info(f"[OGD] OGD+ mode: {use_ogd_plus}")
    
    def _register_hooks(self):
        """Register forward hooks on protected layers."""
        self._remove_hooks()  # Remove existing hooks first
        
        # Debug: collect all available module names to help diagnose layer name issues
        all_module_names = [name for name, _ in self.model.named_modules()]
        logger.debug(f"[OGD] Total modules in model: {len(all_module_names)}")
        
        for layer_name in self.protected_layers:
            # Find the layer in the model
            module = self._get_module_by_name(layer_name)
            if module is None:
                logger.warning(f"[OGD] Layer '{layer_name}' not found in model, skipping")
                # Try to find similar layer names to help user debug
                layer_parts = layer_name.split('.')
                # Search for layers containing the last 2-3 parts of the layer name
                search_parts = layer_parts[-2:] if len(layer_parts) >= 2 else layer_parts
                similar = [name for name in all_module_names if all(part in name for part in search_parts)]
                if similar:
                    logger.warning(f"[OGD] Similar layer names found (showing first 5): {similar[:5]}")
                continue
            
            # Create and register hook
            hook = OGDHook(layer_name, self.memory)
            handle = module.register_forward_hook(hook)
            self.hooks[layer_name] = handle
            logger.info(f"[OGD] Registered hook on '{layer_name}'")
    
    def _get_module_by_name(self, layer_name: str) -> Optional[nn.Module]:
        """
        Get module by name (supports nested names like 'model.layers.31').
        Handles wrapped models (PeftModel, DeepSpeed, etc.)
        
        Supports layer names like:
        - 'base_model.model.model.visual.blocks.30.attn.qkv' (PeftModel wrapped)
        - 'model.model.visual.blocks.30.attn.qkv' (direct model)
        """
        # Try to find the module, handling various model wrappers
        parts = layer_name.split('.')
        module = self.model
        
        # Handle DeepSpeed-wrapped models
        if hasattr(module, 'module'):
            module = module.module
        
        # Handle PeftModel wrapper - if layer_name starts with 'base_model', skip it
        if parts[0] == 'base_model':
            # Skip 'base_model' prefix and continue with the rest
            parts = parts[1:]
            # Now get to the actual base model
            if hasattr(module, 'base_model'):
                module = module.base_model
            elif hasattr(module, 'get_base_model'):
                module = module.get_base_model()
        
        # Handle PeftModel wrapper (if not already handled above)
        if hasattr(module, 'base_model') and hasattr(module.base_model, 'model'):
            module = module.base_model.model
        elif hasattr(module, 'get_base_model'):
            module = module.get_base_model()
        
        # Now traverse the path
        for part in parts:
            if hasattr(module, part):
                module = getattr(module, part)
            else:
                # Try to access as integer index (for list-like modules)
                try:
                    idx = int(part)
                    if isinstance(module, (list, nn.ModuleList, nn.Sequential)):
                        module = module[idx]
                    elif hasattr(module, '__getitem__'):
                        module = module[idx]
                    else:
                        # Try alternative: maybe it's a dict-like access
                        if hasattr(module, '__contains__') and part in module:
                            module = module[part]
                        else:
                            return None
                except (ValueError, IndexError, TypeError, KeyError):
                    return None
        
        return module
    
    def update_memory(self, dataloader, num_batches: int = 100, loss_fn=None, print_weights_metrics: bool = False):
        """
        Update memory from pretrained model using validation/training data.
        
        For feature-based OGD: captures activations from forward pass.
        For gradient-based OGD: captures gradients from backward pass (requires loss_fn).
        
        Args:
            dataloader: DataLoader to iterate over
            num_batches: Number of batches to process
            loss_fn: Loss function (required for gradient-based OGD, optional for feature-based)
            print_weights_metrics: If True, print weights metrics after memory update (useful for Zero3 debugging)
        """
        if self.use_gradients:
            # Gradient-based: need loss function
            if loss_fn is None:
                logger.warning("[OGD] Gradient-based mode requires loss_fn. Falling back to feature-based collection.")
                self._update_memory_features(dataloader, num_batches)
            else:
                self._update_memory_gradients(dataloader, num_batches, loss_fn)
        else:
            # Feature-based: just forward pass
            self._update_memory_features(dataloader, num_batches)
        
        # Print weights metrics if requested (useful for debugging Zero3 weight gathering)
        print_weights_metrics = True
        if print_weights_metrics:
            self.print_protected_layers_weights_metrics()
    
    def _update_memory_features(self, dataloader, num_batches: int = 100):
        """
        Update memory with features (activations) from forward pass.
        This is the original feature-based approach.
        """
        logger.info(f"[OGD] Updating feature memory from {num_batches} batches...")
        
        # Register forward hooks
        self._register_hooks()
        
        # Set model to eval mode
        was_training = self.model.training
        self.model.eval()
        
        # Check if using DeepSpeed ZeRO-3 (need to gather parameters)
        try:
            import deepspeed
            is_deepspeed_zero3 = (
                hasattr(self.model, 'module') and 
                hasattr(self.model, 'is_zero3_enabled') and 
                self.model.is_zero3_enabled()
            )
        except (ImportError, AttributeError):
            is_deepspeed_zero3 = False
        
        try:
            batch_count = 0
            consecutive_errors = 0
            max_consecutive_errors = 10  # Stop if too many consecutive errors
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(dataloader):
                    if batch_count >= num_batches:
                        break
                    
                    # Forward pass (hooks will capture activations)
                    try:
                        # For DeepSpeed ZeRO-3, DeepSpeed handles parameter gathering automatically
                        # The 'weight' must be 2-D error might be from a different issue
                        # Let's try the forward pass directly - DeepSpeed will gather as needed
                        if isinstance(batch, dict):
                            self.model(**batch)
                        elif isinstance(batch, (list, tuple)):
                            self.model(*batch)
                        else:
                            self.model(batch)
                        
                        # Reset error counter on success
                        consecutive_errors = 0
                        batch_count += 1
                        
                        if (batch_count + 1) % 10 == 0:
                            logger.info(f"[OGD] Processed {batch_count + 1}/{num_batches} batches")
                    
                    except Exception as e:
                        consecutive_errors += 1
                        error_msg = str(e)
                        # Only log first few errors to avoid spam
                        if consecutive_errors <= 3 or batch_idx % 20 == 0:
                            logger.warning(f"[OGD] Error processing batch {batch_idx}: {error_msg}")
                        
                        # Stop if too many consecutive errors (likely a systematic issue)
                        if consecutive_errors >= max_consecutive_errors:
                            logger.error(f"[OGD] Too many consecutive errors ({consecutive_errors}), stopping memory update")
                            break
                        continue
            
            # Log memory status
            for layer_name in self.protected_layers:
                stored_count = self.memory.get_stored_count(layer_name)
                if stored_count > 0:
                    logger.info(f"[OGD] {layer_name}: stored {stored_count} features")
                else:
                    logger.warning(f"[OGD] {layer_name}: stored 0 features (check layer name)")
        
        finally:
            # Remove hooks
            self._remove_hooks()
            
            # Restore training mode
            if was_training:
                self.model.train()
    
    def _update_memory_gradients(self, dataloader, num_batches: int = 100, loss_fn=None):
        """
        Update memory with gradients from backward pass.
        This is the standard OGD+ gradient-based approach.
        
        Args:
            dataloader: DataLoader to iterate over
            num_batches: Number of batches to process
            loss_fn: Loss function to compute gradients
        """
        logger.info(f"[OGD] Updating gradient memory from {num_batches} batches...")
        
        # Register gradient capture hooks
        self._register_gradient_capture_hooks()
        
        was_training = self.model.training
        self.model.train()  # Need gradients, so use train mode
        
        # Check if using DeepSpeed ZeRO-3 (need to gather parameters)
        try:
            from transformers.integrations import is_deepspeed_zero3_enabled
            is_deepspeed_zero3 = is_deepspeed_zero3_enabled()
        except (ImportError, AttributeError):
            # Fallback: check model structure
            try:
                import deepspeed
                is_deepspeed_zero3 = (
                    hasattr(self.model, 'module') and 
                    hasattr(self.model, 'is_zero3_enabled') and 
                    self.model.is_zero3_enabled()
                )
            except (ImportError, AttributeError):
                is_deepspeed_zero3 = False
        
        try:
            batch_count = 0
            consecutive_errors = 0
            max_consecutive_errors = 10  # Stop if too many consecutive errors
            
            for batch_idx, batch in enumerate(dataloader):
                if batch_count >= num_batches:
                    break
                
                try:
                    # For DeepSpeed ZeRO-3, we don't need to gather all parameters
                    # DeepSpeed handles parameter gathering automatically during forward pass
                    # The error might be coming from something else - let's try without explicit gathering first
                    # Forward pass
                    if isinstance(batch, dict):
                        outputs = self.model(**batch)
                    elif isinstance(batch, (list, tuple)):
                        outputs = self.model(*batch)
                    else:
                        outputs = self.model(batch)
                    
                    # Compute loss
                    if loss_fn is not None:
                        loss = loss_fn(outputs, batch)
                    elif hasattr(outputs, 'loss') and outputs.loss is not None:
                        loss = outputs.loss
                    else:
                        logger.warning(f"[OGD] No loss available for batch {batch_idx}, skipping")
                        continue
                    
                    # Backward pass (gradients will be captured by hooks)
                    loss.backward()
                    
                    # Clear gradients for next iteration
                    self.model.zero_grad()
                    
                    # Reset error counter on success
                    consecutive_errors = 0
                    batch_count += 1
                    
                    if (batch_count + 1) % 10 == 0:
                        logger.info(f"[OGD] Processed {batch_count + 1}/{num_batches} batches")
                
                except Exception as e:
                    consecutive_errors += 1
                    error_msg = str(e)
                    # Only log first few errors to avoid spam
                    if consecutive_errors <= 3 or batch_idx % 20 == 0:
                        logger.warning(f"[OGD] Error processing batch {batch_idx}: {error_msg}")
                    
                    # Stop if too many consecutive errors (likely a systematic issue)
                    if consecutive_errors >= max_consecutive_errors:
                        logger.error(f"[OGD] Too many consecutive errors ({consecutive_errors}), stopping memory update")
                        break
                    
                    self.model.zero_grad()  # Clear gradients on error
                    continue
            
            # Log memory status
            for layer_name in self.protected_layers:
                stored_count = self.memory.get_stored_count(layer_name)
                if stored_count > 0:
                    logger.info(f"[OGD] {layer_name}: stored {stored_count} gradients")
                else:
                    logger.warning(f"[OGD] {layer_name}: stored 0 gradients (check layer name)")
        
        finally:
            # Remove gradient capture hooks
            self._remove_gradient_capture_hooks()
            
            # Restore original mode
            if not was_training:
                self.model.eval()
    
    def _register_gradient_capture_hooks(self):
        """Register hooks to capture gradients during backward pass."""
        self._remove_gradient_capture_hooks()  # Remove existing first
        
        for layer_name in self.protected_layers:
            module = self._get_module_by_name(layer_name)
            if module is None:
                continue
            
            for param_name, param in module.named_parameters():
                if not param.requires_grad:
                    continue
                
                def make_hook(name):
                    def grad_hook(grad):
                        if grad is not None:
                            # Store gradient in memory
                            self.memory.update(name, grad)
                        return grad  # Don't modify, just capture
                    return grad_hook
                
                handle = param.register_hook(make_hook(layer_name))
                full_name = f"{layer_name}.{param_name}" if param_name else layer_name
                self._gradient_capture_hooks[full_name] = handle
    
    def _remove_gradient_capture_hooks(self):
        """Remove gradient capture hooks."""
        for handle in self._gradient_capture_hooks.values():
            handle.remove()
        self._gradient_capture_hooks.clear()
    
    def enable_projection(self):
        """Enable gradient projection during training."""
        self.projection_enabled = True
        logger.info("[OGD] Gradient projection enabled")
    
    def disable_projection(self):
        """Disable gradient projection."""
        self.projection_enabled = False
        logger.info("[OGD] Gradient projection disabled")
    
    def project_gradient(self, layer_name: str, grad: torch.Tensor) -> torch.Tensor:
        """
        Project gradient orthogonally to previous task representations.
        
        Formula: g_projected = g - P @ P^T @ g
        
        For feature-based: P is built from features, grad is reshaped to match feature_dim
        For gradient-based: P is built from gradients, grad is flattened to param_dim
        
        Args:
            layer_name: Name of the layer
            grad: Gradient tensor
            
        Returns:
            Projected gradient
        """
        if not self.projection_enabled:
            return grad
        
        P = self.memory.get_projection_matrix(layer_name)
        if P is None:
            return grad  # No memory stored, return original gradient
        
        original_shape = grad.shape
        
        if self.use_gradients:
            # Gradient-based: P shape is [param_dim, memory_size]
            # Flatten gradient to match
            grad_flat = grad.flatten()  # [param_dim]
            
            # Project: g_projected = g - P @ P^T @ g
            P_T_g = torch.matmul(P.t(), grad_flat.unsqueeze(-1))  # [memory_size, 1]
            P_P_T_g = torch.matmul(P, P_T_g).squeeze(-1)  # [param_dim]
            grad_projected_flat = grad_flat - P_P_T_g  # [param_dim]
            
            # Reshape to original shape
            grad_projected = grad_projected_flat.view(original_shape)
        else:
            # Feature-based: P shape is [feature_dim, memory_size]
            # grad shape: [..., feature_dim] (flattened)
            grad_flat = grad.view(-1, original_shape[-1])  # [num_params, feature_dim]
            original_norm = torch.norm(grad_flat, dim=-1).mean()
            
            # Project: g_projected = g - P @ P^T @ g
            # P^T @ g: [memory_size, num_params]
            # P @ (P^T @ g): [feature_dim, num_params]
            P_T_g = torch.matmul(P.t(), grad_flat.t())  # [memory_size, num_params]
            P_P_T_g = torch.matmul(P, P_T_g)  # [feature_dim, num_params]
            grad_projected = grad_flat.t() - P_P_T_g  # [feature_dim, num_params]
            grad_projected = grad_projected.t()  # [num_params, feature_dim]
            projected_norm = torch.norm(grad_projected, dim=-1).mean()
            
            # Reshape to original shape
            grad_projected = grad_projected.view(original_shape)

        # Calculate reduction ratio
        reduction = (original_norm - projected_norm) / (original_norm + 1e-9)
        reduction_val = reduction.item() if isinstance(reduction, torch.Tensor) else reduction
        
        # Accumulate stats for logging
        if layer_name not in self._projection_stats:
            self._projection_stats[layer_name] = []
        self._projection_stats[layer_name].append(reduction_val)
        self._projection_count += 1
        
        self._projection_step += 1
        if self._projection_step % self._projection_log_interval == 0:
            stored = self.memory.get_stored_count(layer_name)
            logger.info(
                f"[OGD] Projection stats | layer={layer_name} | "
                f"stored={stored} | Δnorm={reduction_val:.4f}"
            )
        
        return grad_projected
    
    def get_projection_stats(self) -> Dict[str, float]:
        """
        Get aggregated projection statistics for logging.
        
        Returns:
            Dictionary with aggregated stats:
            - ogd_projection_mean: Mean reduction ratio across all layers
            - ogd_projection_max: Maximum reduction ratio
            - ogd_projection_min: Minimum reduction ratio
            - ogd_projection_count: Total number of projections performed
            - ogd_projection_layers: Number of layers with projections
        """
        if not self._projection_stats or self._projection_count == 0:
            return {
                'ogd_projection_mean': 0.0,
                'ogd_projection_max': 0.0,
                'ogd_projection_min': 0.0,
                'ogd_projection_count': 0,
                'ogd_projection_layers': 0
            }
        
        # Aggregate across all layers
        all_reductions = []
        for layer_stats in self._projection_stats.values():
            all_reductions.extend(layer_stats)
        
        if not all_reductions:
            return {
                'ogd_projection_mean': 0.0,
                'ogd_projection_max': 0.0,
                'ogd_projection_min': 0.0,
                'ogd_projection_count': 0,
                'ogd_projection_layers': 0
            }
        
        # numpy is already imported at module level
        return {
            'ogd_projection_mean': float(np.mean(all_reductions)),
            'ogd_projection_max': float(np.max(all_reductions)),
            'ogd_projection_min': float(np.min(all_reductions)),
            'ogd_projection_count': self._projection_count,
            'ogd_projection_layers': len(self._projection_stats)
        }
    
    def reset_projection_stats(self):
        """Reset accumulated projection stats (call at start of each logging interval)."""
        self._projection_stats.clear()
        self._projection_count = 0
    
    def set_output_dir(self, output_dir: str):
        """Set the output directory for saving projection stats."""
        self._output_dir = output_dir
        logger.info(f"[OGD] Output directory set to: {output_dir}")
    
    def save_projection_stats(self, global_step: int, force_save: bool = False):
        """
        Save projection stats to files in the output directory.
        
        Args:
            global_step: Current training step
            force_save: If True, save even if no new stats (for final save)
        """
        if self._output_dir is None:
            logger.warning(f"[OGD] Cannot save stats: output_dir is None (step={global_step})")
            return
        
        # Get current aggregated stats
        stats = self.get_projection_stats()
        
        # Only save if we have stats or if forced
        if not force_save and stats.get('ogd_projection_count', 0) == 0:
            logger.debug(f"[OGD] Skipping save at step {global_step}: no projection stats (count=0)")
            return
        
        logger.info(f"[OGD] Saving projection stats at step {global_step} to {self._output_dir}")
        
        try:
            # Create output directory if it doesn't exist
            os.makedirs(self._output_dir, exist_ok=True)
        except Exception as e:
            logger.error(f"[OGD] ERROR: Cannot create output directory {self._output_dir}: {e}")
            return
        
        # Save detailed stats per layer
        detailed_stats = {
            'global_step': global_step,
            'timestamp': None,  # Could add timestamp if needed
            'summary': stats,
            'per_layer': {}
        }
        
        # Add per-layer stats
        for layer_name, reductions in self._projection_stats.items():
            if reductions:
                detailed_stats['per_layer'][layer_name] = {
                    'count': len(reductions),
                    'mean': float(np.mean(reductions)),
                    'std': float(np.std(reductions)),
                    'min': float(np.min(reductions)),
                    'max': float(np.max(reductions)),
                    'all_values': [float(r) for r in reductions]  # Keep all values for analysis
                }
        
        # Append to history
        self._stats_history.append(detailed_stats)
        
        # Save to JSON file (append mode, one JSON object per line)
        try:
            jsonl_path = os.path.join(self._output_dir, 'ogd_projection_stats.jsonl')
            with open(jsonl_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(detailed_stats) + '\n')
            logger.debug(f"[OGD] Wrote to {jsonl_path}")
        except Exception as e:
            logger.error(f"[OGD] ERROR writing JSONL file: {e}", exc_info=True)
        
        # Save summary CSV (one row per logging step)
        try:
            csv_path = os.path.join(self._output_dir, 'ogd_projection_summary.csv')
            file_exists = os.path.exists(csv_path)
            
            with open(csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if not file_exists:
                    # Write header
                    writer.writerow([
                        'global_step',
                        'projection_count',
                        'projection_layers',
                        'mean_reduction',
                        'min_reduction',
                        'max_reduction'
                    ])
                
                # Write summary row
                writer.writerow([
                    global_step,
                    stats.get('ogd_projection_count', 0),
                    stats.get('ogd_projection_layers', 0),
                    stats.get('ogd_projection_mean', 0.0),
                    stats.get('ogd_projection_min', 0.0),
                    stats.get('ogd_projection_max', 0.0)
                ])
            logger.debug(f"[OGD] Wrote to {csv_path}")
        except Exception as e:
            logger.error(f"[OGD] ERROR writing summary CSV: {e}", exc_info=True)
        
        # Save per-layer detailed CSV (one row per layer per step)
        try:
            layer_csv_path = os.path.join(self._output_dir, 'ogd_projection_per_layer.csv')
            layer_file_exists = os.path.exists(layer_csv_path)
            
            with open(layer_csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if not layer_file_exists:
                    # Write header
                    writer.writerow([
                        'global_step',
                        'layer_name',
                        'projection_count',
                        'mean_reduction',
                        'std_reduction',
                        'min_reduction',
                        'max_reduction'
                    ])
                
                # Write one row per layer
                for layer_name, layer_data in detailed_stats['per_layer'].items():
                    writer.writerow([
                        global_step,
                        layer_name,
                        layer_data['count'],
                        layer_data['mean'],
                        layer_data['std'],
                        layer_data['min'],
                        layer_data['max']
                    ])
            logger.debug(f"[OGD] Wrote to {layer_csv_path}")
        except Exception as e:
            logger.error(f"[OGD] ERROR writing per-layer CSV: {e}", exc_info=True)
        
        logger.info(
            f"[OGD] ✅ Saved projection stats to {self._output_dir} "
            f"(step={global_step}, count={stats.get('ogd_projection_count', 0)}, "
            f"layers={stats.get('ogd_projection_layers', 0)})"
        )
    
    def print_protected_layers_weights_metrics(self, sample_size: int = 5):
        """
        Print partial weights metrics of protected layers to debug Zero3 weight gathering.
        
        This method helps verify that weights are being gathered correctly in Zero3 settings
        by printing statistics and sample values for each protected layer.
        
        Args:
            sample_size: Number of sample values to print per parameter (default: 5)
        """
        # Check if using DeepSpeed ZeRO-3
        try:
            from transformers.integrations import is_deepspeed_zero3_enabled
            is_zero3 = is_deepspeed_zero3_enabled()
        except (ImportError, AttributeError):
            # Fallback: check model structure
            try:
                import deepspeed
                is_zero3 = (
                    hasattr(self.model, 'module') and 
                    hasattr(self.model, 'is_zero3_enabled') and 
                    self.model.is_zero3_enabled()
                )
            except (ImportError, AttributeError):
                is_zero3 = False
        
        import torch.distributed as dist
        is_dist = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if is_dist else 0
        
        print(f"\n{'='*80}", flush=True)
        print(f"[OGD] Protected Layers Weights Metrics (Zero3: {is_zero3}, Rank: {rank})", flush=True)
        print(f"{'='*80}", flush=True)
        logger.info(f"[OGD] Printing protected layers weights metrics (Zero3: {is_zero3}, Rank: {rank})")
        
        for layer_name in self.protected_layers:
            module = self._get_module_by_name(layer_name)
            if module is None:
                print(f"[OGD] ⚠️  Layer '{layer_name}': NOT FOUND", flush=True)
                logger.warning(f"[OGD] Layer '{layer_name}' not found for weight metrics")
                continue
            
            print(f"\n[OGD] Layer: {layer_name}", flush=True)
            logger.info(f"[OGD] Analyzing weights for layer: {layer_name}")
            
            # Get all parameters in this module
            params = list(module.named_parameters(recurse=False))
            if not params:
                print(f"  No parameters found in this module", flush=True)
                logger.warning(f"[OGD] No parameters found in layer '{layer_name}'")
                continue
            
            for param_name, param in params:
                if not param.requires_grad:
                    continue
                
                full_param_name = f"{layer_name}.{param_name}" if param_name else layer_name
                print(f"  Parameter: {param_name}", flush=True)
                
                try:
                    # Gather parameter if Zero3, otherwise use directly
                    if is_zero3:
                        try:
                            import deepspeed
                            # In Zero3, all ranks must enter the context, but only rank 0 gets full data
                            with deepspeed.zero.GatheredParameters(param, modifier_rank=0):
                                if rank == 0:
                                    # Rank 0 has the full gathered parameter
                                    weight_data = param.data.cpu().clone()
                                    gathered = True
                                else:
                                    # Non-rank-0: parameter may be partitioned, but we can still check shape
                                    # Note: In Zero3, non-rank-0 may have empty or partial tensors
                                    if param.numel() > 0:
                                        weight_data = param.data.cpu().clone()
                                        gathered = True
                                    else:
                                        # Empty tensor on this rank (partitioned away)
                                        weight_data = None
                                        gathered = True  # Gathering happened, but this rank doesn't have data
                        except Exception as e:
                            print(f"    ⚠️  Failed to gather parameter: {e}", flush=True)
                            logger.warning(f"[OGD] Failed to gather parameter {full_param_name}: {e}")
                            weight_data = param.data.cpu().clone() if param.numel() > 0 else None
                            gathered = False
                    else:
                        weight_data = param.data.cpu().clone()
                        gathered = False
                    
                    if weight_data is None or weight_data.numel() == 0:
                        # In Zero3, non-rank-0 may have empty tensors (partitioned away)
                        if is_zero3 and rank != 0:
                            print(f"    Shape: {tuple(param.shape)} | Partitioned (not on this rank)", flush=True)
                            logger.info(f"[OGD] Parameter {full_param_name} is partitioned away on rank {rank}")
                        else:
                            print(f"    Shape: {tuple(param.shape)} | Empty parameter", flush=True)
                            logger.warning(f"[OGD] Parameter {full_param_name} is empty")
                        continue
                    
                    # Calculate statistics
                    weight_flat = weight_data.flatten()
                    mean_val = weight_flat.mean().item()
                    std_val = weight_flat.std().item()
                    min_val = weight_flat.min().item()
                    max_val = weight_flat.max().item()
                    norm_val = weight_flat.norm().item()
                    
                    # Print statistics
                    print(f"    Shape: {tuple(weight_data.shape)} | Total elements: {weight_data.numel()}", flush=True)
                    print(f"    Gathered (Zero3): {gathered} | Device: {param.device}", flush=True)
                    print(f"    Statistics:", flush=True)
                    print(f"      Mean:  {mean_val:.6f}", flush=True)
                    print(f"      Std:   {std_val:.6f}", flush=True)
                    print(f"      Min:   {min_val:.6f}", flush=True)
                    print(f"      Max:   {max_val:.6f}", flush=True)
                    print(f"      Norm:  {norm_val:.6f}", flush=True)
                    
                    # Print sample values
                    num_samples = min(sample_size, weight_flat.numel())
                    if num_samples > 0:
                        sample_values = weight_flat[:num_samples].tolist()
                        print(f"    Sample values (first {num_samples}): {[f'{v:.6f}' for v in sample_values]}", flush=True)
                    
                    # Log to logger as well
                    logger.info(
                        f"[OGD] {full_param_name} | shape={tuple(weight_data.shape)} | "
                        f"gathered={gathered} | mean={mean_val:.6f} | std={std_val:.6f} | "
                        f"norm={norm_val:.6f}"
                    )
                    
                except Exception as e:
                    print(f"    ❌ Error analyzing parameter: {e}", flush=True)
                    logger.error(f"[OGD] Error analyzing parameter {full_param_name}: {e}", exc_info=True)
        
        print(f"\n{'='*80}\n", flush=True)
        logger.info("[OGD] Finished printing protected layers weights metrics")
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        self._remove_hooks()
    
    def _remove_hooks(self):
        """Internal method to remove hooks."""
        for handle in self.hooks.values():
            handle.remove()
        self.hooks.clear()


class OGDOptimizer(torch.optim.Optimizer):
    """
    Wrapper around PyTorch optimizer that applies OGD gradient projection.
    
    This class inherits from torch.optim.Optimizer to be compatible with PyTorch schedulers.
    
    Usage:
        base_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        ogd_optimizer = OGDOptimizer(base_optimizer, ogd_trainer)
        
        # Use ogd_optimizer instead of base_optimizer
        loss.backward()
        ogd_optimizer.step()
    """
    
    def __init__(self, optimizer: torch.optim.Optimizer, ogd_trainer: OGDTrainer):
        """
        Initialize OGD Optimizer.
        
        Args:
            optimizer: Base PyTorch optimizer (with different LRs for different param groups)
            ogd_trainer: OGDTrainer instance
        
        Note: This wrapper preserves the base optimizer's parameter groups and their learning rates.
        For example, if the base optimizer has different LRs for graph_encoder vs image/text encoders,
        those will be preserved. The OGD gradient projection is applied during backward pass via hooks,
        but the learning rates remain unchanged.
        """
        # Initialize parent Optimizer with the same parameter groups as base optimizer
        # This is required for PyTorch schedulers to recognize it as a valid optimizer
        # IMPORTANT: We pass a COPY of param_groups to parent, but our property getter
        # delegates to the base optimizer's param_groups, so schedulers will modify the
        # base optimizer's param_groups directly (which is what we want)
        # 
        # Note: torch.optim.Optimizer.__init__ tries to set self.defaults = defaults.
        # If we define defaults as a property, it will conflict. So we need to:
        # 1. Store the base optimizer first
        # 2. Initialize parent with copies
        # 3. Then override defaults property to delegate to base optimizer
        self.optimizer = optimizer
        defaults_copy = dict(optimizer.defaults) if optimizer.defaults else {}
        
        # Initialize parent - this will set self.defaults internally
        # We can't define defaults as a property because parent __init__ sets it directly
        # The parent class stores a copy of defaults, which is fine for scheduler compatibility
        super().__init__(optimizer.param_groups, defaults_copy)
        
        self.ogd_trainer = ogd_trainer
        self._grad_hooks: Dict[str, torch.utils.hooks.RemovableHandle] = {}
        
        # Log that we're preserving the base optimizer's learning rate configuration
        if hasattr(optimizer, 'param_groups') and len(optimizer.param_groups) > 0:
            lrs = [pg.get('lr', 'N/A') for pg in optimizer.param_groups]
            unique_lrs = sorted(set(lrs))
            if len(unique_lrs) > 1:
                import logging
                logger = logging.getLogger(__name__)
                logger.info(f"[OGD] OGDOptimizer preserving base optimizer's learning rates: {unique_lrs}")
        
        # Register backward hooks on protected layers
        self._register_grad_hooks()
    
    def _register_grad_hooks(self):
        """Register backward hooks to project gradients."""
        for layer_name in self.ogd_trainer.protected_layers:
            module = self.ogd_trainer._get_module_by_name(layer_name)
            if module is None:
                continue
            
            # Register hook on parameters
            for param_name, param in module.named_parameters():
                if not param.requires_grad:
                    continue
                
                full_name = f"{layer_name}.{param_name}" if param_name else layer_name
                
                def make_hook(name):
                    def grad_hook(grad):
                        if grad is None:
                            return None
                        return self.ogd_trainer.project_gradient(name, grad)
                    return grad_hook
                
                handle = param.register_hook(make_hook(layer_name))
                self._grad_hooks[full_name] = handle
    
    @property
    def param_groups(self):
        """
        Delegate param_groups to base optimizer (required for scheduler compatibility).
        
        This ensures that:
        1. Schedulers can access and modify learning rates in param_groups
        2. Different learning rates for different parameter groups (e.g., graph_encoder vs image/text)
           are preserved and can be scheduled independently
        3. The base optimizer's param_groups are the source of truth
        """
        return self.optimizer.param_groups
    
    @param_groups.setter
    def param_groups(self, value):
        """
        Set param_groups on base optimizer.
        
        This allows schedulers to modify learning rates, and the changes will be
        applied to the base optimizer, preserving the different LRs for different groups.
        """
        self.optimizer.param_groups = value
    
    # Note: We don't define defaults as a property because torch.optim.Optimizer.__init__
    # sets self.defaults directly, and defining it as a property would conflict.
    # The parent class stores a copy of defaults, which is fine for scheduler compatibility.
    # Schedulers primarily use param_groups (which we delegate via property) to access
    # learning rates, so they don't need direct access to defaults.
    
    def zero_grad(self, set_to_none: bool = False):
        """Zero gradients (delegate to base optimizer)."""
        return self.optimizer.zero_grad(set_to_none=set_to_none)
    
    def step(self, closure=None):
        """Perform optimization step (gradients are already projected by hooks)."""
        return self.optimizer.step(closure)
    
    def state_dict(self):
        """Get optimizer state dict."""
        return self.optimizer.state_dict()
    
    def load_state_dict(self, state_dict):
        """Load optimizer state dict."""
        return self.optimizer.load_state_dict(state_dict)
    
    def __getattr__(self, name):
        """Delegate other attributes to base optimizer."""
        return getattr(self.optimizer, name)
    
    def remove_hooks(self):
        """Remove gradient hooks."""
        for handle in self._grad_hooks.values():
            handle.remove()
        self._grad_hooks.clear()


def identify_qwen2vl_layers(model: nn.Module, num_visual_blocks: int = 24, num_llm_layers: int = 32) -> List[str]:
    """
    Identify important layers in Qwen2VL model for OGD protection.
    
    Args:
        model: Qwen2VL model
        num_visual_blocks: Number of visual transformer blocks (default: 24 for 7B)
        num_llm_layers: Number of LLM layers (default: 32 for 7B)
    
    Returns:
        List of layer names to protect
    """
    protected = []
    
    # Find visual encoder layers (last 2 blocks)
    for name, module in model.named_modules():
        # Visual encoder: last 2 transformer blocks
        if 'visual' in name and 'transformer' in name and 'resblocks' in name:
            try:
                # Extract block number from name like 'visual.transformer.resblocks.23'
                parts = name.split('.')
                if len(parts) >= 4:
                    block_num = int(parts[-1])
                    if block_num >= num_visual_blocks - 2:  # Last 2 blocks
                        protected.append(name)
            except (ValueError, IndexError):
                pass
        
        # Text encoder: last 2 layers
        if 'model.layers' in name:
            try:
                parts = name.split('.')
                if len(parts) >= 3:
                    layer_num = int(parts[2])
                    if layer_num >= num_llm_layers - 2:  # Last 2 layers
                        protected.append(name)
            except (ValueError, IndexError):
                pass
        
        # Projection heads (VERY IMPORTANT!)
        if 'projection' in name.lower() or '_proj' in name.lower():
            protected.append(name)
        
        # Aligner/merger layers (connect vision to language)
        if 'merger' in name.lower() or 'aligner' in name.lower():
            protected.append(name)
    
    # Remove duplicates and sort
    protected = sorted(list(set(protected)))
    
    logger.info(f"[OGD] Identified {len(protected)} layers to protect")
    for layer in protected[:10]:  # Show first 10
        logger.info(f"  - {layer}")
    if len(protected) > 10:
        logger.info(f"  ... and {len(protected) - 10} more")
    
    return protected


def identify_phi3_vision_layers(
    model: nn.Module,
    num_vision_blocks: int = 24,  # Adjust based on actual vision encoder size
    num_llm_layers: int = 32
) -> List[str]:
    """
    Identify important layers in Phi-3.5-vision model for OGD protection.
    
    Phi-3.5-vision has:
    - Vision encoder: model.vision_embed_tokens.img_processor (image processor)
    - Aligner: model.vision_embed_tokens.img_projection (projects vision to text space)
    - Language model: model.layers (text decoder)
    
    Architecture paths:
    - Vision: model.vision_embed_tokens.img_processor.*
    - Aligner: model.vision_embed_tokens.img_projection.*
    - Language: model.layers.{}.self_attn.qkv_proj, model.layers.{}.self_attn.o_proj
    
    Args:
        model: Phi-3.5-vision model
        num_vision_blocks: Number of vision encoder blocks (default: 24, adjust if needed)
        num_llm_layers: Number of LLM layers (default: 32)
    
    Returns:
        List of layer names to protect
    
    Recommendations:
        - Protect last 2 vision blocks (if accessible)
        - Protect aligner/projection layers (CRITICAL for vision-text alignment)
        - Protect last 2-3 LLM layers' attention projections
    """
    protected = []
    
    for name, module in model.named_modules():
        # Vision encoder: last 2 blocks (if we can identify them)
        # Phi-3.5-vision uses img_processor which may have internal blocks
        if 'vision_embed_tokens' in name and 'img_processor' in name:
            # Protect vision processor layers (these encode images)
            # The exact structure depends on the vision encoder implementation
            if 'attn' in name or 'mlp' in name or 'norm' in name:
                # Try to extract block number if possible
                try:
                    parts = name.split('.')
                    for i, part in enumerate(parts):
                        if part.isdigit():
                            block_num = int(part)
                            # Protect last 2 blocks if we can identify them
                            if block_num >= num_vision_blocks - 2:
                                protected.append(name)
                            break
                except (ValueError, IndexError):
                    # If we can't identify block numbers, protect all vision layers
                    # (This is conservative but safe)
                    protected.append(name)
        
        # Aligner/Projection (VERY CRITICAL for vision-text alignment!)
        if 'vision_embed_tokens' in name and 'img_projection' in name:
            protected.append(name)
        
        # Language model: last 2-3 layers (attention projections)
        if 'model.layers' in name:
            try:
                parts = name.split('.')
                if len(parts) >= 3:
                    layer_num = int(parts[2])
                    # Protect last 3 layers' attention projections
                    if layer_num >= num_llm_layers - 3:
                        if 'self_attn' in name and ('qkv_proj' in name or 'o_proj' in name):
                            protected.append(name)
            except (ValueError, IndexError):
                pass
    
    # Remove duplicates and sort
    protected = sorted(list(set(protected)))
    
    logger.info(f"[OGD] Identified {len(protected)} layers to protect for Phi-3.5-vision")
    logger.info(f"[OGD] Protection strategy: vision blocks (last 2), aligner, LLM layers (last 3)")
    for layer in protected[:15]:  # Show first 15 (may have more vision layers)
        logger.info(f"  - {layer}")
    if len(protected) > 15:
        logger.info(f"  ... and {len(protected) - 15} more")
    
    return protected
