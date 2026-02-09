# Copyright (c) Alibaba, Inc. and its affiliates.
# Part of the implementation is borrowed from huggingface/transformers.
import inspect
import json
import os
from contextlib import contextmanager, nullcontext
from functools import partial, wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from peft import PeftModel
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformers import EvalPrediction
from transformers import Seq2SeqTrainer as HfSeq2SeqTrainer
from transformers import Trainer as HfTrainer
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.utils import is_peft_available

from swift.utils import JsonlWriter, Serializer, gc_collect, get_logger, unwrap_model_for_generation
from .arguments import Seq2SeqTrainingArguments, TrainingArguments
from .mixin import DataLoaderMixin, SwiftMixin
from .utils import per_token_loss_func, per_token_loss_func_sp

logger = get_logger()


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


_GRAPH_DEBUG_ENABLED = _env_flag("SWIFT_DEBUG_GRAPHS")


def _graph_param_should_be_saved(name: str) -> bool:
    """
    Include any parameter that belongs to graph_encoder EXCEPT:
    - Shared backbone weights under 'qwen_model' (works for Qwen2VL, Phi-3.5-vision, etc.)
      Note: 'qwen_model' is just a variable name in the graph encoder - it references
      the shared language model (could be any model type)
    - LoRA adapter tensors (already saved in adapter bundle)
    This avoids brittle hardcoded prefixes and captures future modules.
    """
    if not name:
        return False
    lowered = name.lower()
    # Filter out shared language model weights (referenced as 'qwen_model' in graph encoder)
    # This works for all models (Qwen2VL, Phi-3.5-vision, etc.) since they all share the language model
    if 'qwen_model' in lowered:
        return False
    # Filter out LoRA adapters (saved separately by PEFT)
    if 'lora_a' in lowered or 'lora_b' in lowered or 'lora_embedding' in lowered:
        return False
    return True


def _log_graph_state(logger, graph_state: Dict[str, torch.Tensor], context: str) -> None:
    if not _GRAPH_DEBUG_ENABLED:
        return
    if not graph_state:
        logger.info("No %s tensors to save.", context)
        return
    logger.info("%s tensors to save: %d", context, len(graph_state))
    preview = sorted(graph_state.items())
    max_preview = 20
    for name, tensor in preview[:max_preview]:
        shape = tuple(tensor.shape)
        logger.info("  • %s | shape=%s | dtype=%s", name, shape, tensor.dtype)
    if len(preview) > max_preview:
        logger.info("  … (%d more tensors)", len(preview) - max_preview)


def _ensure_full_graph_state(
    graph_encoder: nn.Module,
    graph_state: Dict[str, torch.Tensor],
    is_zero3: bool,
) -> None:
    """
    Ensure all graph encoder parameters are in graph_state.
    This is a safety net for any parameters that weren't gathered properly.
    """
    params = dict(graph_encoder.named_parameters(recurse=True))
    buffers = dict(graph_encoder.named_buffers(recurse=True))

    def _clone_param(param: torch.nn.Parameter, name: str) -> Optional[torch.Tensor]:
        # CRITICAL: In ZeRO-3, don't check numel() before gathering
        # Parameters are partitioned, so numel() will be 0 on non-owning ranks
        if param is None:
            return None
        if is_zero3:
            try:
                import deepspeed

                # Gather the parameter first, then check size
                with deepspeed.zero.GatheredParameters(param, modifier_rank=0):
                    # After gathering, clone the data
                    cloned = param.detach().cpu().clone()
                    if cloned.numel() > 0:
                        return cloned
                    else:
                        logger = get_logger()
                        logger.error(
                            "[graph-save] Parameter %s has zero elements after gathering during completeness check! "
                            "This indicates a serious issue with parameter gathering.",
                            name,
                        )
                    return None
            except Exception as exc:
                logger = get_logger()
                logger.warning(
                    "[graph-save] Failed to gather parameter %s during completeness check: %s",
                    name,
                    exc,
                )
                return None
        # For non-ZeRO-3, check numel before cloning
        if param.numel() == 0:
                return None
        return param.detach().cpu().clone()

    added: List[str] = []
    for name in list(params.keys()) + list(buffers.keys()):
        if not _graph_param_should_be_saved(name):
            continue
        if name in graph_state:
            continue

        tensor = None
        if name in params:
            tensor = _clone_param(params[name], name)
        elif name in buffers:
            if buffers[name].numel() > 0:
                tensor = buffers[name].detach().cpu().clone()

        if tensor is not None:
            graph_state[name] = tensor
            added.append(name)

    if added:
        logger = get_logger()
        logger.info(
            "[graph-save] Added %d missing parameters: %s%s",
            len(added),
            added[:5],
            "..." if len(added) > 5 else "",
        )



class Trainer(SwiftMixin, DataLoaderMixin, HfTrainer):
    args: TrainingArguments

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._zero3_state_dict = None
        self._zero3_graph_encoder_state = None

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        import torch.distributed as dist
        is_dist = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if is_dist else 0
        
        logger.info(f"Rank {rank}: Trainer._save() ENTRY POINT - called")
        
        # Check if we've already saved (to prevent duplicate saves from super()._save_checkpoint())
        # This can happen in two cases:
        # 1. _swift_already_saved: Set when _save_checkpoint() calls _save() directly
        # 2. _swift_pre_saved: Set when _maybe_log_save_evaluate() pre-emptively calls _save()
        if (hasattr(self.args, '_swift_already_saved') and getattr(self.args, '_swift_already_saved', False)) or \
           (hasattr(self.args, '_swift_pre_saved') and getattr(self.args, '_swift_pre_saved', False)):
            logger.info(f"Rank {rank}: _save() already called, skipping duplicate save but still participating in barriers")
            # Still need to participate in barriers for synchronization
            if is_dist:
                dist.barrier()
                logger.info(f"Rank {rank}: Completed barrier sync (duplicate save skipped)")
            return
        
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        if rank == 0 or not is_dist:
            os.makedirs(output_dir, exist_ok=True)

        is_zero3 = False
        try:
            import deepspeed
            is_zero3 = deepspeed.is_deepspeed_zero3_enabled()
        except Exception:
            pass

        # Use pre-gathered ZeRO-3 state dict (gathered by ALL ranks in save_model)
        # Following the tested approach from swift2/trainers/trainers.py
        pre_state_dict = getattr(self, '_zero3_state_dict', None)
        
        # Store full state dict for graph encoder extraction
        full_state_dict_for_graph = None
        
        if pre_state_dict is not None:
            # Use pre-gathered full state dict
            logger.info(f"[ZeRO-3] Using pre-gathered full state dict with {len(pre_state_dict)} parameters")
            full_state_dict_for_graph = pre_state_dict  # Keep full state dict for graph encoder extraction
            state_dict = pre_state_dict
            # Clear it after use (but we'll use full_state_dict_for_graph for graph encoder)
            self._zero3_state_dict = None
        elif state_dict is None:
            # Fallback: gather state dict if not pre-gathered
            if is_zero3 and hasattr(self, 'model_wrapped') and hasattr(
                    self.model_wrapped, '_zero3_consolidated_16bit_state_dict'):
                logger.info("Gathering full model state dict for ZeRO-3 (this may take a while...)")
                state_dict = self.model_wrapped._zero3_consolidated_16bit_state_dict()
                full_state_dict_for_graph = state_dict
            else:
                state_dict = self.model.state_dict()
                full_state_dict_for_graph = state_dict

        # Ensure we have rank info for save operations
        if 'is_dist' not in locals() or 'rank' not in locals():
            import torch.distributed as dist
            is_dist = dist.is_available() and dist.is_initialized()
            rank = dist.get_rank() if is_dist else 0
        
        # For PEFT models, use the FULL state dict directly for saving adapters
        # The user confirmed that saving 80,740,352 elements (154.00 MB) directly works correctly
        # in downstream tasks, so we should save the full state dict instead of extracting PEFT parameters
        # IMPORTANT: The full state dict from _zero3_consolidated_16bit_state_dict() contains adapter parameters
        # which is what we want to save (154MB). We should NOT extract PEFT parameters from it.
        if is_peft_available() and isinstance(self.model, PeftModel) and state_dict is not None:
            # Calculate sizes for diagnostics
            full_size = sum(t.numel() * t.element_size() for t in state_dict.values()) / (1024 * 1024)
            full_elements = sum(t.numel() for t in state_dict.values())
            
            logger.info(f"[ZeRO-3] Using FULL state dict for adapters (confirmed correct in downstream tasks)")
            logger.info(f"[ZeRO-3] Full state dict: {len(state_dict)} parameters, {full_elements:,} elements ({full_size:.2f} MB)")
            
            # Log sample keys to verify
            if len(state_dict) > 0:
                sample_keys = list(state_dict.keys())[:10]
                logger.info(f"[ZeRO-3] Sample parameter keys (first 10): {sample_keys}")
            
            # Check if graph encoder trainable parameters (like node_projection.weight) are in the full state dict
            # For PeftModel, the state dict from _zero3_consolidated_16bit_state_dict() typically only contains
            # adapter parameters (LoRA weights), not the base model trainable parameters like graph_encoder's
            # node_projection, convs, etc. These need to be gathered separately.
            graph_encoder_prefixes = [
                'base_model.model.graph_encoder.',  # Correct path for PeftModel
                'graph_encoder.',  # Fallback for non-PeftModel
                'model.graph_encoder.',  # Another fallback
                'base_model.model.model.graph_encoder.',  # Another possible path
            ]
            graph_keys_in_state_dict = [k for k in state_dict.keys() 
                                       if any(k.startswith(prefix) for prefix in graph_encoder_prefixes)]
            
            if graph_keys_in_state_dict:
                logger.info(f"[ZeRO-3] Found {len(graph_keys_in_state_dict)} graph encoder keys in full state dict")
                logger.info(f"[ZeRO-3] Sample graph encoder keys: {graph_keys_in_state_dict[:5]}")
                # Check specifically for node_projection.weight (the trainable parameter we need)
                node_proj_keys = [k for k in graph_keys_in_state_dict if 'node_projection.weight' in k and 'qwen_model' not in k]
                if node_proj_keys:
                    logger.info(f"[ZeRO-3] Found node_projection.weight in full state dict: {node_proj_keys[0]}")
                    node_proj_tensor = state_dict[node_proj_keys[0]]
                    logger.info(f"[ZeRO-3]   Shape: {tuple(node_proj_tensor.shape)}, numel: {node_proj_tensor.numel()}")
                else:
                    logger.warning(f"[ZeRO-3] node_projection.weight NOT found in full state dict (only LoRA adapters from graph_encoder's qwen_model are present)")
                    logger.warning(f"[ZeRO-3] Graph encoder trainable parameters will be gathered separately from base model")
            else:
                logger.warning(f"[ZeRO-3] No graph encoder keys found in full state dict from PeftModel")
                logger.warning(f"[ZeRO-3] Graph encoder parameters will be gathered from base model separately")
            
            # Use the full state dict directly (don't extract PEFT parameters)
            # This is the 154MB state dict with 80,740,352 elements that the user confirmed works correctly
            # state_dict is already the full state dict from _zero3_consolidated_16bit_state_dict(), so we keep it as is
        
        # Store full state dict for graph encoder extraction
        # This will be used in _save_graph_encoder_for_trainer to extract graph encoder params
        # Note: For PeftModel, this might not include graph encoder, so we'll also use pre-gathered state
        if full_state_dict_for_graph is not None:
            self._zero3_full_state_dict_for_graph = full_state_dict_for_graph
        
        # Critical: Barrier before saving to ensure all ranks are synchronized
        # This prevents deadlocks when rank 0 tries to save while other ranks are still gathering
        if is_dist:
            logger.info(f"Rank {rank}: Waiting for all ranks before save operation...")
            dist.barrier()
            logger.info(f"Rank {rank}: All ranks synchronized, proceeding with save")
        
        # Only rank 0 should save to avoid file conflicts and deadlocks
        if rank == 0 or not is_dist:
            logger.info(f"Rank {rank}: Starting save process...")
            if state_dict is not None:
                state_dict_size = sum(t.numel() * t.element_size() for t in state_dict.values()) / (1024 * 1024)
                state_dict_elements = sum(t.numel() for t in state_dict.values())
                logger.info(f"Rank {rank}: About to call super()._save() with state_dict containing {len(state_dict)} parameters, {state_dict_elements:,} elements ({state_dict_size:.2f} MB)")
            else:
                logger.info(f"Rank {rank}: About to call super()._save() with state_dict=None")
            
            # For PeftModel, we need to save the full state dict (154MB) directly
            # because model.save_pretrained() will filter it internally
            # We'll override _save_model() to handle this
            try:
                super()._save(output_dir, state_dict=state_dict)
                logger.info(f"Rank {rank}: ✅ Completed super()._save() - adapters saved")
            except Exception as e:
                logger.error(f"Rank {rank}: ❌ Error in super()._save(): {e}")
                import traceback
                logger.error(traceback.format_exc())
                raise
            
            logger.info(f"Rank {rank}: About to save graph encoder...")
            try:
                # Pass the full state dict to graph encoder saving function
                full_state_dict = getattr(self, '_zero3_full_state_dict_for_graph', None)
                _save_graph_encoder_for_trainer(self, output_dir, is_zero3, full_state_dict=full_state_dict)
                # Clear the stored full state dict after use
                if hasattr(self, '_zero3_full_state_dict_for_graph'):
                    delattr(self, '_zero3_full_state_dict_for_graph')
                logger.info(f"Rank {rank}: ✅ Completed graph encoder save")
            except Exception as e:
                logger.error(f"Rank {rank}: ❌ Error in graph encoder save: {e}")
                import traceback
                logger.error(traceback.format_exc())
                raise
        else:
            logger.info(f"Rank {rank}: Skipping save (rank 0 only) - but still need to participate in barriers")
            # Still need to call _save_graph_encoder_for_trainer for barrier synchronization
            # but it will skip saving on non-zero ranks
            logger.info(f"Rank {rank}: Calling _save_graph_encoder_for_trainer for barrier sync...")
            full_state_dict = getattr(self, '_zero3_full_state_dict_for_graph', None)
            _save_graph_encoder_for_trainer(self, output_dir, is_zero3, full_state_dict=full_state_dict)
            if hasattr(self, '_zero3_full_state_dict_for_graph'):
                delattr(self, '_zero3_full_state_dict_for_graph')
            logger.info(f"Rank {rank}: ✅ Completed _save_graph_encoder_for_trainer (barrier sync)")
        
        # Critical: Final barrier to ensure all ranks complete _save before returning
        # HuggingFace Trainer may wait for all ranks to complete _save, so we need to synchronize
        if is_dist:
            logger.info(f"Rank {rank}: Waiting for all ranks to complete _save()...")

            # CRITICAL: After checkpoint save, ensure DeepSpeed ZeRO-3 parameters are properly released
            # This prevents hangs when DeepSpeed tries to gather parameters for the next training step
            # The AllGather operations after checkpoint save are likely from DeepSpeed trying to gather
            # parameters for the next forward/backward pass, but previous gathering hasn't been fully released

            # Clear any cached state dicts that might interfere with next training step
            if hasattr(self, '_zero3_state_dict') and self._zero3_state_dict is not None:
                logger.info(f"Rank {rank}: Clearing cached _zero3_state_dict after _save()")
                self._zero3_state_dict = None
            if hasattr(self, '_zero3_graph_encoder_state') and self._zero3_graph_encoder_state is not None:
                logger.info(f"Rank {rank}: Clearing cached _zero3_graph_encoder_state after _save()")
                self._zero3_graph_encoder_state = None

            # Synchronize CUDA operations to ensure all parameter gathering is complete
            if torch.cuda.is_available():
                torch.cuda.synchronize()  # Ensure all CUDA operations complete

            # Barrier to synchronize all ranks and ensure parameters are released
            dist.barrier()
            logger.info(f"Rank {rank}: All ranks completed _save(), ZeRO-3 parameters released, returning")

    def _save_model(self, output_dir: Optional[str] = None, state_dict=None):
        """
        Override _save_model to save the full state dict (154MB) directly for PeftModel,
        bypassing model.save_pretrained() which filters the state dict.
        """
        from swift.utils import get_logger
        logger = get_logger()
        
        # For PeftModel with full state dict, save it directly
        if is_peft_available() and isinstance(self.model, PeftModel) and state_dict is not None:
            import os
            import safetensors.torch
            
            # Filter out graph_encoder's qwen_model LoRA parameters to avoid shared memory issues
            # These are duplicates of the main model's LoRA parameters (they share memory)
            # We only want to save the main model's LoRA parameters, not the graph_encoder's qwen_model ones
            filtered_state_dict = {}
            graph_encoder_qwen_prefix = 'base_model.model.graph_encoder.node_encoder.qwen_model.'
            filtered_count = 0
            
            for key, tensor in state_dict.items():
                # Skip graph_encoder's qwen_model LoRA parameters (they're duplicates/shared with main model)
                if key.startswith(graph_encoder_qwen_prefix) and ('lora_A' in key or 'lora_B' in key):
                    filtered_count += 1
                    continue
                # Clone the tensor to avoid shared memory issues with safetensors
                # This ensures each tensor has its own memory
                if isinstance(tensor, torch.Tensor):
                    filtered_state_dict[key] = tensor.clone().detach()
                else:
                    filtered_state_dict[key] = tensor
            
            if filtered_count > 0:
                logger.info(f"[ZeRO-3] _save_model: Filtered out {filtered_count} duplicate graph_encoder qwen_model LoRA parameters (shared memory)")
            
            # Calculate size for logging
            state_dict_size = sum(t.numel() * t.element_size() for t in filtered_state_dict.values()) / (1024 * 1024)
            state_dict_elements = sum(t.numel() for t in filtered_state_dict.values())
            
            logger.info(f"[ZeRO-3] _save_model: Saving FULL state dict directly for PeftModel")
            logger.info(f"[ZeRO-3] _save_model: Full state dict: {len(filtered_state_dict)} params, {state_dict_elements:,} elements ({state_dict_size:.2f} MB)")
            
            # Save the filtered state dict directly to adapter_model.safetensors
            adapter_model_path = os.path.join(output_dir, 'adapter_model.safetensors')
            logger.info(f"[ZeRO-3] _save_model: Saving to {adapter_model_path}")
            safetensors.torch.save_file(filtered_state_dict, adapter_model_path)
            logger.info(f"[ZeRO-3] _save_model: ✅ Saved full state dict ({len(filtered_state_dict)} params, {state_dict_elements:,} elements, {state_dict_size:.2f} MB)")
            
            # Save adapter_config.json if it exists
            try:
                if hasattr(self.model, 'peft_config'):
                    import json
                    adapter_config_path = os.path.join(output_dir, 'adapter_config.json')
                    peft_config = self.model.peft_config
                    config_dict = {}
                    for adapter_name, config in peft_config.items():
                        config_dict[adapter_name] = config.to_dict()
                    with open(adapter_config_path, 'w') as f:
                        json.dump(config_dict, f, indent=2)
                    logger.info(f"[ZeRO-3] _save_model: ✅ Saved adapter_config.json")
            except Exception as e:
                logger.warning(f"[ZeRO-3] _save_model: Could not save adapter_config.json: {e}")
            
            # CRITICAL: Don't call super()._save_model() for PeftModel, as it will call model.save_pretrained()
            # which will filter the state dict and overwrite our 77MB file with a smaller one
            # We've already saved the full state dict directly, so we're done
            logger.info(f"[ZeRO-3] _save_model: Skipping super()._save_model() to avoid overwriting our saved file")
            return
        else:
            # For non-PeftModel or when state_dict is None, use the parent implementation
            super()._save_model(output_dir, state_dict=state_dict)

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """
        Override save_model to gather ZeRO-3 state dict with ALL ranks participating.
        This is called BEFORE _save, and all ranks participate.
        Following the tested approach from swift2/trainers/trainers.py
        """
        from swift.utils import get_logger
        logger = get_logger()
        
        is_zero3 = False
        try:
            import deepspeed
            is_zero3 = deepspeed.is_deepspeed_zero3_enabled()
        except Exception:
            pass

        if is_zero3 and hasattr(self, 'model_wrapped'):
            import torch.distributed as dist
            rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
            
            logger.info(f"[Rank {rank}] 🔄 save_model called (ALL ranks participating)")
            logger.info(f"[Rank {rank}] ⏳ Gathering ZeRO-3 state dict...")
            
            # ALL ranks must call this
            if dist.is_available() and dist.is_initialized():
                dist.barrier()

            try:
                # Collective operation 1: Gather FULL state dict (includes all parameters)
                # This is the key difference - we gather the FULL state dict, not just adapters
                logger.info(f"[Rank {rank}] Calling _zero3_consolidated_16bit_state_dict...")
                state_dict = self.model_wrapped._zero3_consolidated_16bit_state_dict()
                logger.info(f"[Rank {rank}] ✅ Got full state dict with {len(state_dict)} parameters")
                
                # Store for rank 0 to use in _save
                # In _save, we'll extract adapter parameters from this full state dict
                self._zero3_state_dict = state_dict

                # Collective operation 2: Gather graph_encoder params (if exists)
                import deepspeed
                
                # Find graph_encoder in model
                model = self.model
                logger.info(f"[Rank {rank}] save_model: Checking for graph_encoder, initial model type: {type(model).__name__}")
                if isinstance(model, PeftModel):
                    base_model = getattr(model, 'base_model', None)
                    if base_model is not None and hasattr(base_model, 'model'):
                        model = base_model.model
                        logger.info(f"[Rank {rank}] save_model: Unwrapped to base_model.model: {type(model).__name__}")
                    else:
                        model = getattr(model, 'model', model)
                        logger.info(f"[Rank {rank}] save_model: Unwrapped to model.model: {type(model).__name__}")
                
                logger.info(f"[Rank {rank}] save_model: Final model type: {type(model).__name__}, has graph_encoder: {hasattr(model, 'graph_encoder')}")
                if hasattr(model, 'graph_encoder'):
                    logger.info(f"[Rank {rank}] save_model: graph_encoder found! Type: {type(model.graph_encoder).__name__}")
                else:
                    logger.warning(f"[Rank {rank}] save_model: graph_encoder NOT found in model! Model attributes: {[attr for attr in dir(model) if not attr.startswith('_')][:20]}")

                self._zero3_graph_encoder_state = None
                if hasattr(model, 'graph_encoder'):
                    logger.info(f"[Rank {rank}] Gathering graph_encoder trainable params...")
                    
                    # Try a different approach: gather all graph encoder parameters at once
                    # by getting the state_dict of the graph_encoder module within a gathering context
                    graph_encoder = model.graph_encoder
                    
                    # First, collect all parameters that need to be gathered
                    all_graph_params = dict(graph_encoder.named_parameters(recurse=True))
                    all_graph_buffers = dict(graph_encoder.named_buffers(recurse=True))
                    
                    # DIAGNOSTIC: Check node_projection.weight specifically
                    if 'node_projection.weight' in all_graph_params:
                        node_proj_param = all_graph_params['node_projection.weight']
                        logger.info(
                            f"[save_model] DIAGNOSTIC: Found node_projection.weight: "
                            f"requires_grad={node_proj_param.requires_grad}, "
                            f"shape={tuple(node_proj_param.shape)}, "
                            f"numel={node_proj_param.numel()}, "
                            f"should_save={_graph_param_should_be_saved('node_projection.weight')}"
                        )
                    else:
                        logger.error(f"[save_model] DIAGNOSTIC: node_projection.weight NOT FOUND in graph_encoder.named_parameters()!")
                        logger.error(f"[save_model] Available parameters: {list(all_graph_params.keys())[:30]}")
                    
                    graph_encoder_state = {}
                    lora_skipped = 0
                    skipped_not_trainable = []
                    skipped_filtered = []
                    
                    # Collect parameters that should be saved
                    # Following swift2 approach: only save trainable parameters that pass the filter
                    for name, param in all_graph_params.items():
                        # Track why parameters are skipped
                        if not param.requires_grad:
                            skipped_not_trainable.append(name)
                        if not _graph_param_should_be_saved(name):
                            skipped_filtered.append(name)
                        
                        # Skip non-trainable parameters or parameters that should be filtered out
                        if not param.requires_grad or not _graph_param_should_be_saved(name):
                            if param.requires_grad and 'lora' in name.lower():
                                lora_skipped += 1
                            continue

                        # Gather base trainable params only (node_head, convs, layer_norms, output_proj, etc.)
                        # ALL ranks must enter this context for ZeRO-3 to work correctly
                        # The GatheredParameters context manager requires all ranks to participate
                        try:
                            # DIAGNOSTIC: Log before gathering for critical parameters
                            if name == 'node_projection.weight' and rank == 0:
                                logger.info(
                                    f"[save_model] DIAGNOSTIC: About to gather {name}: "
                                    f"original shape={tuple(param.shape)}, original numel={param.numel()}, "
                                    f"original device={param.device}"
                                )
                            
                            with deepspeed.zero.GatheredParameters(param, modifier_rank=0):
                                # Only rank 0 saves the data
                                # Inside the context, param.data contains the full gathered parameter on rank 0
                                if rank == 0:
                                    # CRITICAL: Access param.data INSIDE the context to get the gathered value
                                    gathered_data = param.data
                                    
                                    # DIAGNOSTIC: Log after gathering for critical parameters
                                    if name == 'node_projection.weight':
                                        logger.info(
                                            f"[save_model] DIAGNOSTIC: After gathering {name}: "
                                            f"gathered shape={tuple(gathered_data.shape)}, "
                                            f"gathered numel={gathered_data.numel()}, "
                                            f"gathered device={gathered_data.device}"
                                        )
                                    
                                    # Verify the gathered parameter is valid
                                    if gathered_data.numel() > 0:
                                        graph_encoder_state[name] = gathered_data.cpu().clone()
                                    else:
                                        logger.error(
                                            f"[save_model] CRITICAL: Parameter {name} has zero elements after gathering! "
                                            f"Shape: {tuple(gathered_data.shape)}, dtype: {gathered_data.dtype}, "
                                            f"device: {gathered_data.device}, "
                                            f"original param shape: {tuple(param.shape)}, "
                                            f"original param numel: {param.numel()}"
                                        )
                                        # Still save it to detect the issue
                                        graph_encoder_state[name] = gathered_data.cpu().clone()
                        except Exception as e:
                            if rank == 0:
                                logger.error(f"[save_model] Failed to gather graph encoder parameter {name}: {e}")
                                import traceback
                                logger.error(traceback.format_exc())
                    
                    if rank == 0:
                        self._zero3_graph_encoder_state = graph_encoder_state
                        logger.info(f"[Rank {rank}] ✅ Gathered {len(graph_encoder_state)} graph_encoder params (skipped {lora_skipped} LoRA params)")
                        if skipped_not_trainable:
                            logger.info(f"[save_model] Skipped {len(skipped_not_trainable)} non-trainable params (first 10): {skipped_not_trainable[:10]}")
                        if skipped_filtered:
                            logger.info(f"[save_model] Skipped {len(skipped_filtered)} filtered params (first 10): {skipped_filtered[:10]}")
                        # Check if node_projection.weight was skipped
                        if 'node_projection.weight' in skipped_not_trainable:
                            logger.error(f"[save_model] ❌ node_projection.weight was SKIPPED because requires_grad=False!")
                        if 'node_projection.weight' in skipped_filtered:
                            logger.error(f"[save_model] ❌ node_projection.weight was SKIPPED by filter!")
                        
                        # Verify critical parameters - DETAILED CHECK
                        critical_params = ['node_projection.weight', 'node_projection.bias']
                        for crit_param in critical_params:
                            if crit_param in graph_encoder_state:
                                param = graph_encoder_state[crit_param]
                                numel = param.numel()
                                shape = tuple(param.shape)
                                dtype = param.dtype
                                device = param.device
                                
                                # Calculate size in MB
                                size_mb = (numel * param.element_size()) / (1024 * 1024)
                                
                                if numel == 0:
                                    logger.error(
                                        f"[save_model] ❌ CRITICAL: {crit_param} is EMPTY after gathering! "
                                        f"Shape: {shape}, dtype: {dtype}, device: {device}, numel: {numel}"
                                    )
                                else:
                                    logger.info(
                                        f"[save_model] ✅ {crit_param} gathered successfully: "
                                        f"shape={shape}, numel={numel:,}, dtype={dtype}, device={device}, size={size_mb:.4f} MB"
                                    )
                                    # Print a sample of the actual values to verify it's not all zeros
                                    if numel > 0:
                                        sample_values = param.flatten()[:min(10, numel)]
                                        logger.info(
                                            f"[save_model]   Sample values (first 10): {sample_values.tolist()}"
                                        )
                                        # Check if all zeros
                                        if torch.allclose(param, torch.zeros_like(param)):
                                            logger.error(f"[save_model] ⚠️ WARNING: {crit_param} appears to be all zeros!")
                                        else:
                                            logger.info(f"[save_model]   Values are non-zero (min={param.min().item():.6f}, max={param.max().item():.6f})")
                            else:
                                logger.error(f"[save_model] ❌ CRITICAL: {crit_param} is MISSING from gathered state!")
                                logger.error(f"[save_model] Available keys: {sorted(graph_encoder_state.keys())[:20]}")
                else:
                    if rank == 0:
                        self._zero3_graph_encoder_state = None
                
            except Exception as e:
                logger.error(f"[Rank {rank}] ❌ Failed to gather: {e}")
                import traceback
                logger.error(traceback.format_exc())
                self._zero3_state_dict = None
                self._zero3_graph_encoder_state = None

            if dist.is_available() and dist.is_initialized():
                dist.barrier()
                logger.info(f"[Rank {rank}] 🔄 Synchronized after gathering")

        # Call parent save_model (will call _save on rank 0 only)
        super().save_model(output_dir, _internal_call)

    @contextmanager
    def _patch_loss_function(self):
        model = self.model
        if isinstance(model, PeftModel):
            model = model.model
        model_cls = model.__class__
        if not hasattr(model_cls, 'loss_function'):
            yield
            return

        loss_function = model.loss_function
        _old_loss_function = model_cls.loss_function

        @staticmethod
        @wraps(loss_function)
        def new_loss_function(logits, labels, **kwargs):
            labels = labels.to(logits.device)  # fix device_map
            return loss_function(logits=logits, labels=labels, **kwargs)

        model_cls.loss_function = new_loss_function
        try:
            yield
        finally:
            model_cls.loss_function = _old_loss_function

    def train(self, *args, **kwargs):
        with self._patch_loss_function():
            return super().train(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
        if inputs.get('labels') is not None:
            self._compute_acc(outputs, inputs['labels'])
        if num_items_in_batch is not None and self.model_accepts_loss_kwargs:
            loss = loss / self.args.gradient_accumulation_steps
        return (loss, outputs) if return_outputs else loss


def gather_for_unpadded_tensors(input_data, use_gather_object=False):
    from accelerate.utils import gather_object
    input_data = gather_object(input_data)
    output = []
    for _data in input_data:
        if len(_data.shape) == 0:
            _data = _data.unsqueeze(0)
        _data = _data.cpu()
        output.append(_data)
    if len(output[0].shape) == 1 and output[0].shape[0] > 1:
        data = torch.stack(output, dim=0)
    else:
        data = torch.concat(output, dim=0)
    return data


def _save_graph_encoder_for_trainer(trainer, output_dir: str, is_zero3: bool, full_state_dict: Optional[Dict[str, torch.Tensor]] = None):
    """
    Save graph encoder weights with proper DeepSpeed ZeRO-3 handling.

    Key fixes:
    1. Gather ALL parameters (params + buffers) under ZeRO-3
    2. Use barrier synchronization to prevent race conditions
    3. Handle empty/partitioned tensors correctly
    4. Ensure rank-0 only saves after all gathering is complete
    """
    from swift.utils import get_logger

    logger = get_logger()

    model = trainer.model
    model_type_name = type(model).__name__
    
    if isinstance(model, PeftModel):
        logger.debug(
            f"🔍 Unwrapping PeftModel to access base model (model type: {model_type_name})"
        )
        base_model = getattr(model, "base_model", None)
        if base_model is not None and hasattr(base_model, "model"):
            model = base_model.model
            logger.debug(f"🔍 Unwrapped to: {type(model).__name__} (via base_model.model)")
        else:
            model = getattr(model, "model", model)
            logger.debug(f"🔍 Unwrapped to: {type(model).__name__} (via model.model fallback)")

    if not hasattr(model, "graph_encoder"):
        logger.debug(
            f"ℹ️ No graph_encoder found in {type(model).__name__} - skipping save"
        )
        return
    
    logger.info(
        f"💾 Saving graph_encoder from {type(model).__name__} (original: {model_type_name})"
    )

    graph_encoder = model.graph_encoder

    import torch.distributed as dist

    is_dist = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if is_dist else 0

    if is_dist:
        dist.barrier()

    graph_state: Dict[str, torch.Tensor] = {}
    
    # PRIORITY 1: Extract from full state dict (if it contains graph encoder parameters)
    # Note: For PeftModel, the full state dict from _zero3_consolidated_16bit_state_dict()
    # might only contain adapter parameters, not graph encoder parameters.
    # Graph encoder is in the base model, so we need to check if it's included.
    if full_state_dict is not None and rank == 0:
        logger.info("[ZeRO-3] Attempting to extract graph encoder parameters from full state dict...")
        logger.info(f"[ZeRO-3] Full state dict contains {len(full_state_dict)} parameters")
        
        # Extract graph encoder parameters from full state dict
        # Based on model architecture: PeftModel.base_model.model.graph_encoder
        # So keys should be: base_model.model.graph_encoder.*
        graph_encoder_prefixes = [
            'base_model.model.graph_encoder.',  # Correct path for PeftModel
            'graph_encoder.',  # Fallback for non-PeftModel
            'model.graph_encoder.',  # Another fallback
            'base_model.model.model.graph_encoder.',  # Another possible path
        ]
        
        # First, find all graph encoder keys in the full state dict
        graph_encoder_keys = []
        for key in full_state_dict.keys():
            for prefix in graph_encoder_prefixes:
                if key.startswith(prefix):
                    graph_encoder_keys.append((key, prefix))
                    break
        
        if graph_encoder_keys:
            logger.info(f"[ZeRO-3] Found {len(graph_encoder_keys)} graph encoder keys in full state dict")
            
            # Extract graph encoder parameters
            for full_key, prefix in graph_encoder_keys:
                # Remove prefix to get the parameter name (e.g., "node_projection.weight")
                graph_param_name = full_key[len(prefix):]
                
                # Filter out shared backbone and LoRA parameters
                if not _graph_param_should_be_saved(graph_param_name):
                    continue
                
                tensor = full_state_dict[full_key]
                
                # Extract the parameter
                if tensor.numel() > 0:
                    graph_state[graph_param_name] = tensor.cpu().clone()
                else:
                    logger.error(
                        f"[ZeRO-3] CRITICAL: Parameter {graph_param_name} (from key {full_key}) has zero elements in full state dict! "
                        f"Shape: {tuple(tensor.shape)}, dtype: {tensor.dtype}"
                    )
                    # Still include it to detect the issue
                    graph_state[graph_param_name] = tensor.cpu().clone()
            
            logger.info(f"[ZeRO-3] Extracted {len(graph_state)} graph encoder parameters from full state dict")
            
            # Verify critical parameters
            if 'node_projection.weight' in graph_state:
                node_proj = graph_state['node_projection.weight']
                logger.info(
                    f"[ZeRO-3] ✅ node_projection.weight extracted from full state dict: "
                    f"shape={tuple(node_proj.shape)}, numel={node_proj.numel()}, "
                    f"dtype={node_proj.dtype}"
                )
                if node_proj.numel() > 0:
                    sample_values = node_proj.flatten()[:min(10, node_proj.numel())]
                    logger.info(f"[ZeRO-3]   Sample values (first 10): {sample_values.tolist()}")
            else:
                logger.warning(f"[ZeRO-3] node_projection.weight NOT found in extracted graph state, will try pre-gathered state")
                # Try to find it with different prefixes in full state dict
                for key in full_state_dict.keys():
                    if 'node_projection.weight' in key:
                        logger.info(f"[ZeRO-3] Found similar key in full state dict: {key} (shape={tuple(full_state_dict[key].shape)}, numel={full_state_dict[key].numel()})")
        else:
            logger.warning(f"[ZeRO-3] No graph encoder keys found in full state dict (likely PeftModel with only adapters)")
            logger.warning(f"[ZeRO-3] Will use pre-gathered graph encoder state from save_model() instead")
    
    # PRIORITY 2: Fallback to pre-gathered state from save_model()
    # Use this if full state dict doesn't contain graph encoder parameters (e.g., PeftModel with only adapters)
    if len(graph_state) == 0 and getattr(trainer, "_zero3_graph_encoder_state", None) is not None and rank == 0:
        pre_gathered_state = trainer._zero3_graph_encoder_state
        logger.info("[ZeRO-3] Using pre-gathered graph encoder state from save_model()")
        logger.info(f"[ZeRO-3] Pre-gathered state contains {len(pre_gathered_state)} parameters")
        
        # DIAGNOSTIC: Check if node_projection.weight is in pre-gathered state
        if 'node_projection.weight' in pre_gathered_state:
            node_proj = pre_gathered_state['node_projection.weight']
            logger.info(
                f"[ZeRO-3] DIAGNOSTIC: node_projection.weight found in pre-gathered state: "
                f"shape={tuple(node_proj.shape)}, numel={node_proj.numel()}, "
                f"dtype={node_proj.dtype}, device={node_proj.device}"
            )
        else:
            logger.error(f"[ZeRO-3] ❌ CRITICAL: node_projection.weight NOT in pre-gathered state!")
            logger.error(f"[ZeRO-3] Available keys: {sorted(pre_gathered_state.keys())[:30]}")
        
        graph_state = {}
        zero_sized_params = []
        for k, v in pre_gathered_state.items():
            # Ensure tensors are on CPU and have valid data
            numel = v.numel()
            shape = tuple(v.shape) if hasattr(v, 'shape') else 'unknown'
            dtype = v.dtype if hasattr(v, 'dtype') else 'unknown'
            
            if numel > 0:
                graph_state[k] = v.cpu() if v.device.type != 'cpu' else v
            else:
                zero_sized_params.append((k, shape, dtype))
                logger.error(
                    f"[ZeRO-3] CRITICAL: Pre-gathered parameter {k} has zero elements! "
                    f"Shape: {shape}, dtype: {dtype}, device: {v.device if hasattr(v, 'device') else 'unknown'}"
                )
                # Still include it so we can detect the issue
                graph_state[k] = v.cpu() if v.device.type != 'cpu' else v
        
        if zero_sized_params:
            logger.error(f"[ZeRO-3] Found {len(zero_sized_params)} zero-sized parameters in pre-gathered state:")
            for k, shape, dtype in zero_sized_params[:10]:  # Show first 10
                logger.error(f"  - {k}: shape={shape}, dtype={dtype}")
            if len(zero_sized_params) > 10:
                logger.error(f"  ... and {len(zero_sized_params) - 10} more")
        else:
            logger.info("[ZeRO-3] All pre-gathered parameters have valid sizes")
        
        # Verify critical parameters - DETAILED CHECK BEFORE SAVING
        critical_params = ['node_projection.weight', 'node_projection.bias']
        for crit_param in critical_params:
            if crit_param in graph_state:
                param = graph_state[crit_param]
                numel = param.numel()
                shape = tuple(param.shape)
                dtype = param.dtype
                device = param.device
                
                # Calculate size in MB
                size_mb = (numel * param.element_size()) / (1024 * 1024)
                
                if numel == 0:
                    logger.error(
                        f"[ZeRO-3] ❌ CRITICAL: {crit_param} is EMPTY before saving! "
                        f"Shape: {shape}, dtype: {dtype}, device: {device}, numel: {numel}"
                    )
                else:
                    logger.info(
                        f"[ZeRO-3] ✅ {crit_param} is valid before saving: "
                        f"shape={shape}, numel={numel:,}, dtype={dtype}, device={device}, size={size_mb:.4f} MB"
                    )
                    # Print a sample of the actual values to verify it's not all zeros
                    if numel > 0:
                        sample_values = param.flatten()[:min(10, numel)]
                        logger.info(
                            f"[ZeRO-3]   Sample values (first 10): {sample_values.tolist()}"
                        )
                        # Check if all zeros
                        if torch.allclose(param, torch.zeros_like(param)):
                            logger.error(f"[ZeRO-3] ⚠️ WARNING: {crit_param} appears to be all zeros before saving!")
                        else:
                            logger.info(f"[ZeRO-3]   Values are non-zero (min={param.min().item():.6f}, max={param.max().item():.6f})")
            else:
                logger.error(f"[ZeRO-3] ❌ CRITICAL: {crit_param} is MISSING from pre-gathered state!")
                logger.error(f"[ZeRO-3] Available keys: {sorted(graph_state.keys())[:20]}")
        
        logger.info(f"[ZeRO-3] Loaded {len(graph_state)} parameters from pre-gathered state")
        trainer._zero3_graph_encoder_state = None
    elif is_zero3:
        try:
            import deepspeed
        except ImportError:
            logger.warning(
                "DeepSpeed not available – cannot gather graph encoder parameters for ZeRO-3."
            )
            return

        # Gather parameters: rank 0 collects, all ranks participate in gathering
        logger.info(f"[ZeRO-3] Gathering graph encoder parameters (rank {rank})...")
        all_tensors: Dict[str, Tuple[str, torch.Tensor]] = {}

        # First, collect all parameters and buffers for diagnostics
        all_params_dict = dict(graph_encoder.named_parameters(recurse=True))
        all_buffers_dict = dict(graph_encoder.named_buffers(recurse=True))
        
        if rank == 0:
            logger.info(f"[ZeRO-3] Total parameters in graph_encoder: {len(all_params_dict)}")
            logger.info(f"[ZeRO-3] Total buffers in graph_encoder: {len(all_buffers_dict)}")
            
            # Count parameters by category
            qwen_params = [n for n in all_params_dict.keys() if 'qwen_model' in n.lower()]
            lora_params = [n for n in all_params_dict.keys() if any(x in n.lower() for x in ['lora_a', 'lora_b', 'lora_embedding'])]
            other_params = [n for n in all_params_dict.keys() if n not in qwen_params and n not in lora_params]
            
            logger.info(f"[ZeRO-3] Parameter breakdown:")
            logger.info(f"  - qwen_model parameters: {len(qwen_params)}")
            logger.info(f"  - LoRA parameters: {len(lora_params)}")
            logger.info(f"  - Other parameters (should be saved): {len(other_params)}")
            
            # Calculate sizes
            qwen_size = sum(all_params_dict[n].numel() * all_params_dict[n].element_size() for n in qwen_params) / (1024 * 1024)
            lora_size = sum(all_params_dict[n].numel() * all_params_dict[n].element_size() for n in lora_params) / (1024 * 1024)
            other_size = sum(all_params_dict[n].numel() * all_params_dict[n].element_size() for n in other_params) / (1024 * 1024)
            logger.info(f"[ZeRO-3] Size breakdown:")
            logger.info(f"  - qwen_model: {qwen_size:.2f} MB")
            logger.info(f"  - LoRA: {lora_size:.2f} MB")
            logger.info(f"  - Other (should be saved): {other_size:.2f} MB")

        # Collect all parameters and buffers that should be saved
        for name, param in all_params_dict.items():
                if _graph_param_should_be_saved(name):
                    all_tensors[name] = ("param", param)

        for name, buffer in all_buffers_dict.items():
                if _graph_param_should_be_saved(name):
                    all_tensors[name] = ("buffer", buffer)

        logger.info(f"[ZeRO-3] Found {len(all_tensors)} graph encoder tensors to gather (after filtering)")

        # Gather all tensors (all ranks participate, but only rank 0 collects)
        # CRITICAL: ALL ranks must enter GatheredParameters context for ZeRO-3 to work
        gathered_count = 0
        for name, (tensor_type, tensor) in all_tensors.items():
            if tensor is None:
                continue

            if tensor_type == "param":
                try:
                    # CRITICAL: In ZeRO-3, parameters are partitioned across ranks
                    # ALL ranks must enter the GatheredParameters context for gathering to work
                    # Only rank 0 will have the full parameter after gathering
                    # IMPORTANT: We must access the parameter INSIDE the context to get the gathered value
                    with deepspeed.zero.GatheredParameters(tensor, modifier_rank=0):
                        # Inside the context, tensor.data contains the full gathered parameter on rank 0
                        # On other ranks, it may still be partitioned
                        if rank == 0:
                            # CRITICAL: Access tensor.data INSIDE the context to get gathered value
                            # The gathered parameter is available as tensor.data on rank 0
                            gathered_data = tensor.data  # This is the gathered full parameter
                            
                            # Verify we have data before cloning
                            if gathered_data.numel() > 0:
                                # Clone to CPU to ensure we have a copy
                                gathered_tensor = gathered_data.cpu().clone()
                                graph_state[name] = gathered_tensor
                                gathered_count += 1
                                if _GRAPH_DEBUG_ENABLED:
                                    logger.debug(
                                        f"[ZeRO-3] Gathered param: {name}, shape={tuple(gathered_tensor.shape)}, "
                                        f"dtype={gathered_tensor.dtype}, numel={gathered_tensor.numel()}"
                                    )
                            else:
                                logger.error(
                                    f"[ZeRO-3] CRITICAL: Parameter {name} has zero elements after gathering! "
                                    f"This should not happen for a valid parameter. "
                                    f"Gathered data shape: {tuple(gathered_data.shape)}, "
                                    f"Gathered data numel: {gathered_data.numel()}, "
                                    f"Gathered data device: {gathered_data.device}, "
                                    f"Original tensor shape: {tuple(tensor.shape)}, "
                                    f"Original tensor numel: {tensor.numel()}"
                                )
                                # Still save it (even if empty) so we can detect the issue
                                graph_state[name] = gathered_data.cpu().clone()
                except Exception as exc:
                    if rank == 0:
                        logger.error(f"[ZeRO-3] Failed to gather parameter {name}: {exc}")
                        import traceback
                        logger.error(traceback.format_exc())
            elif tensor_type == "buffer":
                # Buffers are not partitioned in ZeRO-3, so we can copy directly on rank 0
                if rank == 0:
                    if tensor.numel() > 0:
                        graph_state[name] = tensor.cpu().clone()
                        gathered_count += 1
                        if _GRAPH_DEBUG_ENABLED:
                            logger.debug(
                                f"[ZeRO-3] Copied buffer: {name}, shape={tuple(tensor.shape)}"
                            )
        
        if rank == 0:
            logger.info(f"[ZeRO-3] Successfully gathered {gathered_count} graph encoder tensors")
        else:
            logger.info(f"Rank {rank}: Participated in graph encoder parameter gathering")

        # Barrier to ensure all ranks complete gathering before rank 0 proceeds to save
        if is_dist:
            logger.info(f"Rank {rank}: Waiting for all ranks to complete graph encoder gathering...")
            dist.barrier()
            logger.info(f"Rank {rank}: All ranks completed graph encoder gathering")
    else:
        state_dict = graph_encoder.state_dict()
        for name, tensor in state_dict.items():
            if _graph_param_should_be_saved(name):
                graph_state[name] = tensor.detach().cpu()

    if rank != 0:
        return

    logger.info(f"[graph-save] Collected {len(graph_state)} tensors before completeness check")
    
    # Log total parameter count for diagnostics
    all_params = dict(graph_encoder.named_parameters(recurse=True))
    all_buffers = dict(graph_encoder.named_buffers(recurse=True))
    total_params_in_model = len(all_params)
    total_buffers_in_model = len(all_buffers)
    logger.info(f"[graph-save] Graph encoder has {total_params_in_model} parameters and {total_buffers_in_model} buffers total")
    
    # Diagnose which parameters are being filtered out
    if _GRAPH_DEBUG_ENABLED or rank == 0:
        filtered_out = []
        filtered_qwen = []
        filtered_lora = []
        saved_params = []
        
        for name in all_params.keys():
            if _graph_param_should_be_saved(name):
                saved_params.append(name)
            else:
                filtered_out.append(name)
                lowered = name.lower()
                if 'qwen_model' in lowered:
                    filtered_qwen.append(name)
                elif 'lora_a' in lowered or 'lora_b' in lowered or 'lora_embedding' in lowered:
                    filtered_lora.append(name)
        
        logger.info(f"[graph-save] Parameter breakdown:")
        logger.info(f"  - Saved: {len(saved_params)} parameters")
        logger.info(f"  - Filtered (qwen_model): {len(filtered_qwen)} parameters")
        logger.info(f"  - Filtered (lora): {len(filtered_lora)} parameters")
        logger.info(f"  - Filtered (other): {len(filtered_out) - len(filtered_qwen) - len(filtered_lora)} parameters")
        
        # Log sample of saved parameters
        if saved_params:
            logger.info(f"[graph-save] Sample saved parameters (first 10): {saved_params[:10]}")
        
        # Log sample of filtered parameters (non-qwen, non-lora) to catch unexpected filters
        other_filtered = [n for n in filtered_out if n not in filtered_qwen and n not in filtered_lora]
        if other_filtered:
            logger.warning(f"[graph-save] Unexpected filtered parameters (not qwen_model or lora): {other_filtered[:10]}")
        
        # Calculate size of saved vs filtered
        saved_size = sum(all_params[n].numel() * all_params[n].element_size() for n in saved_params if n in all_params) / (1024 * 1024)
        qwen_size = sum(all_params[n].numel() * all_params[n].element_size() for n in filtered_qwen if n in all_params) / (1024 * 1024)
        logger.info(f"[graph-save] Size breakdown:")
        logger.info(f"  - Saved parameters: {saved_size:.2f} MB")
        logger.info(f"  - Filtered qwen_model: {qwen_size:.2f} MB")

    _ensure_full_graph_state(graph_encoder, graph_state, is_zero3)

    logger.info(f"[graph-save] Final tensor count: {len(graph_state)}")
    
    # Calculate total elements saved for size verification
    total_elements = sum(t.numel() for t in graph_state.values())
    total_size_mb = sum(t.numel() * t.element_size() for t in graph_state.values()) / (1024 * 1024)
    logger.info(f"[graph-save] Total elements: {total_elements:,}, Total size: {total_size_mb:.2f} MB")

    if not graph_state:
        logger.warning("No graph encoder parameters to save after gathering!")
        return

    # Check for expected critical parameters
    expected_keys = ["node_projection.weight"]
    missing_critical = [k for k in expected_keys if k not in graph_state]
    if missing_critical:
        logger.error(f"[graph-save] CRITICAL: Missing expected parameters: {missing_critical}")
        logger.error(f"[graph-save] Available keys: {sorted(graph_state.keys())}")
        
        # Try to find similar keys
        for missing_key in missing_critical:
            base_name = missing_key.split('.')[0]
            similar_keys = [k for k in graph_state.keys() if base_name in k]
            if similar_keys:
                logger.error(f"[graph-save] Found similar keys for '{missing_key}': {similar_keys}")

    _log_graph_state(logger, graph_state, "graph_encoder")

    logger.info("[graph-save-debug] final graph_state keys: %s", sorted(graph_state.keys()))

    # FINAL VERIFICATION: Check critical parameters right before saving
    critical_params = ['node_projection.weight', 'node_projection.bias']
    for crit_param in critical_params:
        if crit_param in graph_state:
            param = graph_state[crit_param]
            numel = param.numel()
            shape = tuple(param.shape)
            dtype = param.dtype
            
            if numel == 0:
                logger.error(
                    f"[graph-save] ❌ CRITICAL: {crit_param} is EMPTY right before saving to disk! "
                    f"Shape: {shape}, dtype: {dtype}, numel: {numel}"
                )
            else:
                size_mb = (numel * param.element_size()) / (1024 * 1024)
                logger.info(
                    f"[graph-save] ✅ {crit_param} verified before disk write: "
                    f"shape={shape}, numel={numel:,}, dtype={dtype}, size={size_mb:.4f} MB"
                )
                # Print sample values
                sample_values = param.flatten()[:min(10, numel)]
                logger.info(f"[graph-save]   Final sample values: {sample_values.tolist()}")
                if not torch.allclose(param, torch.zeros_like(param)):
                    logger.info(f"[graph-save]   Final value range: min={param.min().item():.6f}, max={param.max().item():.6f}")
        else:
            logger.error(f"[graph-save] ❌ CRITICAL: {crit_param} is MISSING right before saving!")

    graph_path = os.path.join(output_dir, "graph_encoder.bin")
    torch.save(graph_state, graph_path)
    logger.info(
        f"✅ Saved {len(graph_state)} graph encoder tensors to {graph_path}"
    )
    
    # VERIFY AFTER SAVING: Load back and check
    try:
        loaded_state = torch.load(graph_path, map_location='cpu')
        if 'node_projection.weight' in loaded_state:
            loaded_param = loaded_state['node_projection.weight']
            if loaded_param.numel() == 0:
                logger.error(
                    f"[graph-save] ❌ CRITICAL: node_projection.weight is EMPTY after loading from disk! "
                    f"Shape: {tuple(loaded_param.shape)}, dtype: {loaded_param.dtype}"
                )
            else:
                logger.info(
                    f"[graph-save] ✅ Verified after disk write: node_projection.weight has {loaded_param.numel():,} elements, "
                    f"shape={tuple(loaded_param.shape)}, dtype={loaded_param.dtype}"
                )
                sample_values = loaded_param.flatten()[:min(10, loaded_param.numel())]
                logger.info(f"[graph-save]   Loaded sample values: {sample_values.tolist()}")
        else:
            logger.error(f"[graph-save] ❌ CRITICAL: node_projection.weight is MISSING from saved file!")
    except Exception as e:
        logger.warning(f"[graph-save] Could not verify saved file: {e}")

    config = {
        "hidden_dim": getattr(graph_encoder, "hidden_dim", None),
        "output_dim": getattr(graph_encoder, "output_dim", None),
        "num_layers": getattr(graph_encoder, "num_layers", None),
        "edge_dim": getattr(graph_encoder, "edge_dim", None),
        "use_spatial_encoding": getattr(graph_encoder, "use_spatial_encoding", None),
        "use_edge_features": getattr(graph_encoder, "use_edge_features", None),
        "use_gat": getattr(graph_encoder, "use_gat", None),
        "use_spatial_auxiliary": getattr(
            graph_encoder, "use_spatial_auxiliary", None
        ),
        "spatial_embed_dim": getattr(graph_encoder, "spatial_embed_dim", None),
        "spatial_frequency_num": getattr(
            graph_encoder, "spatial_frequency_num", None
        ),
    }
    config_path = os.path.join(output_dir, "graph_encoder_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    logger.info(f"✅ Saved graph encoder config to {config_path}")


class EmbeddingTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.compute_metrics = self.calculate_metric
        self.preprocess_logits_for_metrics = None
        self.label_names = ['labels']
        self.gather_function = gather_for_unpadded_tensors

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute loss with optional spatial auxiliary loss for graph-based training.
        
        Uses S2 hierarchical token prediction as auxiliary task (based on GeoToken).
        
        Note: Spatial auxiliary loss is OPTIONAL.
        Training will proceed normally even if it is disabled (default behavior).
        """
        # 1. Compute main loss (InfoNCE or other embedding loss)
        from swift.utils import get_logger
        logger = get_logger()
        # logger.info(f"[TRACE] EmbeddingTrainer.compute_loss enter: step={self.state.global_step}, "
        #             f"batch_size={len(inputs.get('input_ids', [])) if isinstance(inputs.get('input_ids'), torch.Tensor) else 'NA'}")
        main_loss, outputs = super().compute_loss(model, inputs, return_outputs=True, 
                                                    num_items_in_batch=num_items_in_batch)
        # logger.info(f"[TRACE] EmbeddingTrainer.compute_loss main loss ok: step={self.state.global_step}, loss={main_loss.item() if torch.is_tensor(main_loss) else main_loss}")
        
        total_loss = main_loss
        # logger.info(f"[DEBUG] EmbeddingTrainer: total_loss: {total_loss}")
        
        # 2. Compute spatial auxiliary loss if enabled (OPTIONAL)
        # If use_spatial_auxiliary=False (default), this step is skipped entirely.
        spatial_loss = None
        if getattr(self.args, 'use_spatial_auxiliary', False):
            spatial_loss = self._compute_spatial_auxiliary_loss(model, inputs)
            
            # logger.info(f"[DEBUG] EmbeddingTrainer: spatial_loss: {spatial_loss}")

            # Only add spatial loss if it was successfully computed (not None)
        # DEBUG: Check if spatial_loss is None and log why
        # if spatial_loss is None:
        #     if self.state.global_step % self.args.logging_steps == 0:
        #         # from swift.utils import get_logger
        #         # logger = get_logger()
        #         # Try to unwrap model for checking
        #         check_model = model
        #         if hasattr(self, 'accelerator'):
        #             check_model = self.accelerator.unwrap_model(model)
        #
        #         logger.warning(f"[DEBUG] spatial_loss is None at step {self.state.global_step}. "
        #                      f"use_spatial_auxiliary={getattr(self.args, 'use_spatial_auxiliary', False)}, "
        #                      f"has_graph_encoder={hasattr(check_model, 'graph_encoder')}")
        #         if hasattr(check_model, 'graph_encoder'):
        #             logger.warning(f"[DEBUG] graph_encoder.s2_token_head exists: "
        #                          f"{hasattr(check_model.graph_encoder, 's2_token_head')}, "
        #                          f"is None: {getattr(check_model.graph_encoder, 's2_token_head', None) is None}")
            # Skip adding spatial loss if it's None
            # spatial_loss = torch.tensor(0.0, device=main_loss.device, requires_grad=False)

            # Combine losses with weighting (PE-GNN style)
            spatial_weight = getattr(self.args, 'spatial_loss_weight', 0.1)
            if spatial_loss is not None:
                logger.info(f"[TRACE] EmbeddingTrainer.compute_loss spatial_loss ok: step={self.state.global_step}, loss={spatial_loss.item() if torch.is_tensor(spatial_loss) else spatial_loss}")
                # logger.info(f"[DEBUG] Spatial loss {spatial_loss} at step {self.state.global_step}. "
                #             f"use_spatial_auxiliary={getattr(self.args, 'use_spatial_auxiliary', False)}, "
                #             f"has_graph_encoder={hasattr(model, 'graph_encoder')}")
                total_loss = total_loss + spatial_weight * spatial_loss
            else:
                total_loss = main_loss
                # logger.warning(f"[DEBUG] Spatial loss is None at step {self.state.global_step}. "
                #                f"use_spatial_auxiliary={getattr(self.args, 'use_spatial_auxiliary', False)}, "
                #                f"has_graph_encoder={hasattr(model, 'graph_encoder')}")

            # Log losses for monitoring
            if self.state.global_step % self.args.logging_steps == 0:
                log_payload = {
                    'train/main_loss': main_loss.item(),
                    'train/total_loss': total_loss.item(),
                    'train/spatial_weight': spatial_weight,
                }
                if spatial_loss is not None:
                    log_payload['train/spatial_loss'] = spatial_loss.item()
                self.log(log_payload)
            # If spatial_loss is None, training continues normally
        
        # Return total loss (which equals main_loss if optional losses are disabled)
        return (total_loss, outputs) if return_outputs else total_loss
    
    def _compute_spatial_auxiliary_loss(self, model, inputs):
        """
        Compute spatial auxiliary loss for graphs in the batch.
        
        Uses S2 hierarchical token prediction (based on GeoToken paper).
        Predicts S2 tokens from node embeddings to learn spatial representations.
        
        Returns:
            spatial_loss: Scalar tensor or None if no graphs with coordinates
        """
        from swift.utils import get_logger
        logger = get_logger()
        
        # Unwrap model if it's wrapped (DeepSpeed, PEFT, etc.)
        # Try accelerator unwrap first (for DeepSpeed)
        if hasattr(self, 'accelerator'):
            try:
                unwrapped_model = self.accelerator.unwrap_model(model)
                if unwrapped_model is not model:
                    model = unwrapped_model
                    logger.debug(f"[DEBUG] Unwrapped model from {type(model).__name__} via accelerator")
            except Exception:
                pass

        # Also try unwrapping PeftModel if needed
        if isinstance(model, PeftModel):
            base_model = getattr(model, 'base_model', None)
            if base_model is not None and hasattr(base_model, 'model'):
                model = base_model.model
                logger.debug(f"[DEBUG] Unwrapped PeftModel to {type(model).__name__}")
            else:
                model = getattr(model, 'model', model)

        # Check if model has graph encoder
        if not hasattr(model, 'graph_encoder'):
            logger.warning_once(f"[DEBUG] _compute_spatial_auxiliary_loss: Model {type(model).__name__} has no graph_encoder attribute")
            return None
        
        # Check if S2 token head exists
        if not hasattr(model.graph_encoder, 's2_token_head') or model.graph_encoder.s2_token_head is None:
            logger.warning_once(f"[DEBUG] _compute_spatial_auxiliary_loss: Graph encoder exists but s2_token_head not initialized. "
                                f"Set use_spatial_auxiliary=true during model initialization. "
                                f"hasattr: {hasattr(model.graph_encoder, 's2_token_head')}, "
                                f"value: {getattr(model.graph_encoder, 's2_token_head', 'NOT_FOUND')}")
            return None
        
        # Get graphs from inputs
        graphs = inputs.get('graphs', None)
        if graphs is None or len(graphs) == 0:
            logger.warning_once(f"[DEBUG] _compute_spatial_auxiliary_loss: No graphs in inputs. "
                              f"inputs keys: {list(inputs.keys())}, graphs: {inputs.get('graphs')}")
            return None
        
        # Import S2 loss computation function
        # try:
        from swift.llm.model.model.spatial_encoders import compute_s2_auxiliary_loss
        # except ImportError:
        #     logger.warning_once("Could not import compute_s2_auxiliary_loss from spatial_encoders")
        #     return None
        
        spatial_losses = []
        all_metrics = []
        graphs_with_coords = 0
        graphs_processed = 0
        
        for graph in graphs:

            # Reuse cached node embeddings from template if available
            node_embeddings = getattr(graph, 'cached_node_embeddings', None)

            if node_embeddings is None:
                # Fallback: run graph encoder again (only happens when caching disabled or missing)
                node_embeddings_list = model.graph_encoder([graph])
                node_embeddings = node_embeddings_list[0]  # [N, D]
            logger.debug(f"[DEBUG] _compute_spatial_auxiliary_loss: node_embeddings: {node_embeddings.shape}")

            # Check if graph has coordinates for edge weight computation
            # Coordinates should be stored by graph encoder during forward pass
            if not hasattr(graph, 'coords') or graph.coords is None:
                # Try to extract coordinates from node_text if not already stored
                if hasattr(model.graph_encoder, 'embed_node_text'):
                    # try:
                        _, coordinates_list = model.graph_encoder.embed_node_text(graph)
                        if coordinates_list and any(c is not None for c in coordinates_list):
                            # Convert to tensor
                            valid_coords = []
                            for coords in coordinates_list:
                                if coords is not None:
                                    valid_coords.append([coords[0], coords[1]])  # [lon, lat]
                                else:
                                    valid_coords.append([0.0, 0.0])
                            if valid_coords:
                                device = node_embeddings.device
                                coords_tensor = torch.tensor(valid_coords, dtype=torch.float32, device=device)  # [lon, lat]
                                # Convert to [lat, lon] format expected by spatial loss
                                graph.coords = torch.stack([coords_tensor[:, 1], coords_tensor[:, 0]], dim=1)
                    # except Exception as e:
                    #     logger.debug(f"Could not extract coordinates from node_text: {e}")

            # if not hasattr(graph, 'coords') or graph.coords is None:
                #     logger.warning_once("Graph missing 'coords' attribute needed for spatial loss. "
                #                        "Ensure node_text contains 'Coordinates: (lon, lat)' format.")
            #     continue
            
            # Check if graph has coords before processing
            if not hasattr(graph, 'coords') or graph.coords is None:
                logger.warning_once(f"[DEBUG] Graph missing 'coords' after extraction attempt. "
                                  f"hasattr: {hasattr(graph, 'coords')}, "
                                  f"value: {getattr(graph, 'coords', 'NOT_FOUND')}")
                continue

            graphs_with_coords += 1

            # Compute S2 token prediction loss
            # try:
            logger.info(f"[DEBUG] _compute_spatial_auxiliary_loss: graph.coords: {graph.coords.shape}")
            s2_level = getattr(model.graph_encoder, 's2_level', 4)
            s2_group_size = getattr(model.graph_encoder, 's2_group_size', 2)
            loss, metrics = compute_s2_auxiliary_loss(
                node_embeddings=node_embeddings,
                coords=graph.coords,
                s2_token_head=model.graph_encoder.s2_token_head,
                s2_level=s2_level,
                group_size=s2_group_size
            )
            
            spatial_losses.append(loss)
            all_metrics.append(metrics)
            graphs_processed += 1
            # except Exception as e:
            #     logger.warning_once(f"Failed to compute spatial loss for graph: {e}")
            #     continue
        
        # Average spatial losses across batch
        if len(spatial_losses) > 0:
            avg_spatial_loss = sum(spatial_losses) / len(spatial_losses)
            
            # Log metrics (average across all graphs in batch)
            if self.state.global_step % self.args.logging_steps == 0 and len(all_metrics) > 0:
                avg_accuracy = sum(m.get('s2_accuracy', 0) for m in all_metrics) / len(all_metrics)
                avg_loss = sum(m.get('s2_loss', 0) for m in all_metrics) / len(all_metrics)
                
                self.log({
                    'train/s2_loss': avg_loss,
                    'train/s2_accuracy': avg_accuracy,
                    'train/spatial_loss_graphs_processed': graphs_processed,
                    'train/spatial_loss_graphs_with_coords': graphs_with_coords,
                })

            # Log diagnostic info periodically
            if self.state.global_step % (self.args.logging_steps * 10) == 0:
                logger.info(f"Spatial auxiliary loss (S2 tokens): {graphs_processed}/{len(graphs)} graphs processed "
                          f"({graphs_with_coords} with coords)")
            
            return avg_spatial_loss
        else:
            # Log why no spatial loss was computed (with more detail for debugging)
            logger.warning_once(f"[DEBUG] _compute_spatial_auxiliary_loss: No graphs processed. "
                              f"Total graphs: {len(graphs)}, "
                              f"Graphs with coords: {graphs_with_coords}, "
                              f"Graphs processed: {graphs_processed}, "
                              f"spatial_losses length: {len(spatial_losses)}")
            # Log details about each graph
            for i, graph in enumerate(graphs):
                has_coords = hasattr(graph, 'coords') and graph.coords is not None
                logger.warning_once(f"[DEBUG] Graph {i}: has_coords={has_coords}, "
                                  f"coords shape: {graph.coords.shape if has_coords else 'N/A'}")
            return None

    def evaluation_loop(self, *args, **kwargs):
        output = super().evaluation_loop(*args, **kwargs)
        self.gather_function = gather_for_unpadded_tensors
        return output

    def calculate_metric(self, eval_prediction: EvalPrediction) -> Dict[str, float]:
        from swift.plugin.loss import calculate_paired_metrics, calculate_infonce_metrics
        args = self.args
        if args.loss_type == 'infonce':
            return calculate_infonce_metrics(eval_prediction.predictions, eval_prediction.label_ids)
        else:
            return calculate_paired_metrics(eval_prediction.predictions, eval_prediction.label_ids)


class RerankerTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.compute_metrics = self.calculate_metric
        self.label_names = ['labels']

        # Set up preprocess_logits_for_metrics to reduce memory usage for generative reranker
        if self.args.loss_type in {'generative_reranker', 'listwise_generative_reranker'}:
            self.preprocess_logits_for_metrics = self._preprocess_generative_reranker_logits
        else:
            self.preprocess_logits_for_metrics = None
        self.gather_function = gather_for_unpadded_tensors

    def _preprocess_generative_reranker_logits(self, logits, labels):
        """
        Preprocess logits for generative reranker to reduce memory usage.
        Extract only the yes/no token logits at the last valid (non -100) timestep
        for each sample, avoiding padded timesteps created by multi-GPU gather.
        """
        import torch
        import os

        # Get token IDs for positive and negative tokens
        positive_token = os.environ.get('GENERATIVE_RERANKER_POSITIVE_TOKEN', 'yes')
        negative_token = os.environ.get('GENERATIVE_RERANKER_NEGATIVE_TOKEN', 'no')

        tokenizer = getattr(self, 'processing_class', None)
        if tokenizer is None:
            # Fallback: return full logits if tokenizer not available
            return logits

        try:
            positive_token_id = tokenizer.convert_tokens_to_ids(positive_token)
            negative_token_id = tokenizer.convert_tokens_to_ids(negative_token)
        except Exception:
            # Fallback: return full logits if token conversion fails
            return logits

        # Extract only the yes/no token logits from the last non -100 position per sample
        # Shapes: logits [batch, seq_len, vocab]
        if len(logits.shape) == 3:
            batch_size, _, vocab_size = logits.shape

            # Identify padded rows whose entire vocab logits are -100
            row_is_pad = (logits == -100).all(dim=-1)  # [batch, seq_len]
            valid_mask = ~row_is_pad
            lengths = valid_mask.long().sum(dim=1) - 1
            lengths = torch.clamp(lengths, min=0)
            last_indices = lengths.to(device=logits.device)

            # Gather the logits at the last valid index for each sample: [batch, vocab]
            gather_index = last_indices.view(batch_size, 1, 1).expand(batch_size, 1, vocab_size)
            last_step_logits = torch.gather(logits, dim=1, index=gather_index).squeeze(1)

            positive_logits = last_step_logits[:, positive_token_id]
            negative_logits = last_step_logits[:, negative_token_id]
            logits = positive_logits - negative_logits
            return logits
        else:
            # Unexpected shape, return as-is
            return logits

    def evaluation_loop(self, *args, **kwargs):
        output = super().evaluation_loop(*args, **kwargs)
        self.gather_function = gather_for_unpadded_tensors
        return output

    def calculate_metric(self, eval_prediction: EvalPrediction) -> Dict[str, float]:
        from swift.plugin.loss import calculate_reranker_metrics
        return calculate_reranker_metrics(eval_prediction.predictions, eval_prediction.label_ids)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Check if we have a custom loss function
        if self.compute_loss_func is not None:
            # Get labels and compute outputs
            labels = inputs.get('labels')
            if labels is not None:
                labels = inputs.pop('labels')

            outputs = model(**inputs)

            if labels is not None:
                # Call custom loss function
                loss = self.compute_loss_func(outputs, labels, num_items_in_batch=num_items_in_batch, trainer=self)
            else:
                # Fallback to model's loss
                loss = outputs.loss

            if num_items_in_batch is not None and self.model_accepts_loss_kwargs:
                loss = loss / self.args.gradient_accumulation_steps

            if labels is not None:
                self._compute_acc(outputs, labels)

            return (loss, outputs) if return_outputs else loss
        else:
            return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)


class Seq2SeqTrainer(SwiftMixin, DataLoaderMixin, HfSeq2SeqTrainer):
    args: Seq2SeqTrainingArguments

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        is_zero3 = False
        try:
            import deepspeed
            is_zero3 = deepspeed.is_deepspeed_zero3_enabled()
        except Exception:
            pass

        pre_state_dict = getattr(self, '_zero3_state_dict', None)
        if pre_state_dict is not None:
            state_dict = pre_state_dict
            self._zero3_state_dict = None

        if state_dict is None:
            if is_zero3 and hasattr(self, 'model_wrapped') and hasattr(
                    self.model_wrapped, '_zero3_consolidated_16bit_state_dict'):
                state_dict = self.model_wrapped._zero3_consolidated_16bit_state_dict()
            else:
                state_dict = self.model.state_dict()

        super()._save(output_dir, state_dict=state_dict)
        _save_graph_encoder_for_trainer(self, output_dir, is_zero3)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_accepts_loss_kwargs = True  # fix transformers>=4.46.2
        if self.args.predict_with_generate:
            from swift.llm import PtEngine
            self.infer_engine = PtEngine.from_model_template(
                self.model, self.template, max_batch_size=self.args.per_device_eval_batch_size)
        self.jsonl_writer = JsonlWriter(os.path.join(self.args.output_dir, 'predict.jsonl'))

    @staticmethod
    def _predict_data_collator(batch):
        return {'_data': batch}

    @contextmanager
    def _patch_predict_with_generate(self):
        origin_data_collator = self.data_collator
        self.data_collator = self._predict_data_collator
        packing = self.template.packing
        padding_free = self.template.padding_free
        self.template.packing = False
        self.template.padding_free = False
        try:
            yield
        finally:
            self.template.packing = packing
            self.template.padding_free = padding_free
            self.data_collator = origin_data_collator

    def evaluate(self, *args, **kwargs):
        context = self._patch_predict_with_generate() if self.args.predict_with_generate else nullcontext()
        with context:
            res = super().evaluate(*args, **kwargs)
            gc_collect()
            return res

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
        **gen_kwargs,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        # Safety check: ensure inputs is not None
        # This can happen with empty batches in multi-GPU evaluation
        if inputs is None:
            logger.warning(
                f"prediction_step received None inputs. "
                f"This may indicate an empty batch in multi-GPU evaluation. "
                f"Process: {getattr(self.accelerator, 'process_index', 'unknown')}, "
                f"Returning None values."
            )
            inputs = {}
        if not self.args.predict_with_generate or prediction_loss_only:
            # If inputs is empty, return None values to skip this step
            if not inputs:
                return None, None, None
            with self.template.forward_context(self.model, inputs):
                return super().prediction_step(
                    model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys)
        from swift.llm import RequestConfig, InferRequest
        # Safety check: ensure _data exists
        if '_data' not in inputs:
            logger.warning(
                f"prediction_step: '_data' key missing from inputs. "
                f"This may indicate an empty batch or data collation issue."
            )
            return None, None, None
        data_list = inputs['_data']
        labels_list = [InferRequest.remove_response(data['messages']) for data in data_list]
        with unwrap_model_for_generation(
                self.model_wrapped, self.accelerator,
                gather_deepspeed3_params=self.args.ds3_gather_for_generation), self.template.generate_context():
            resp_list = self.infer_engine.infer(
                data_list,
                RequestConfig(max_tokens=self.model.generation_config.max_new_tokens),
                use_tqdm=False,
                template=self.template)

        response_list = []
        jsonl_cache = []
        device = self.args.device
        for data, resp, labels in zip(data_list, resp_list, labels_list):
            response = resp.choices[0].message.content
            jsonl_cache.append({'response': response, 'labels': labels, **data})
            response_list.append(Serializer.to_tensor(resp.choices[0].message.content).to(device=device))
        self.jsonl_writer.append(jsonl_cache, gather_obj=True)
        labels_list = [Serializer.to_tensor(labels).to(device=device) for labels in labels_list]
        response_list = pad_sequence(response_list, batch_first=True, padding_value=0)
        labels_list = pad_sequence(labels_list, batch_first=True, padding_value=0)
        return None, response_list, labels_list

    def _prepare_inputs(self, inputs):
        from swift.llm import HfConfigFactory
        args = self.args
        inputs = super()._prepare_inputs(inputs)
        if self.template.sequence_parallel_size > 1:
            from swift.trainers.sequence_parallel import sequence_parallel
            sequence_parallel.prepare_inputs(inputs)

        use_logits_to_keep = self.get_use_logits_to_keep(self.template.sequence_parallel_size == 1)
        if use_logits_to_keep:
            self.prepare_logits_to_keep(inputs)
            if args.tuner_backend == 'unsloth' and isinstance(inputs['logits_to_keep'], torch.Tensor):
                inputs['logits_to_keep'] = int(inputs['logits_to_keep'].sum())

        base_model = self.template.get_base_model(self.model)
        if self.model.model_info.is_moe_model and 'output_router_logits' in inspect.signature(
                base_model.forward).parameters:
            HfConfigFactory.set_config_attr(base_model.config, 'router_aux_loss_coef', args.router_aux_loss_coef)
            base_model.router_aux_loss_coef = args.router_aux_loss_coef
            logger.info_once(f'router_aux_loss_coef: {args.router_aux_loss_coef}')
            if args.router_aux_loss_coef > 0:
                inputs['output_router_logits'] = True
        inputs['compute_loss_func'] = self.compute_loss_func
        return inputs

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = None
        compute_loss_func: Callable = inputs.pop('compute_loss_func', None)
        loss_scale = inputs.pop('loss_scale', None)
        text_position_ids = inputs.pop('text_position_ids', None)
        if text_position_ids is None:
            text_position_ids = inputs.get('position_ids')
        channels = inputs.pop('channel', None)

        if (self.label_smoother is not None or compute_loss_func is not None or loss_scale is not None
                or self.args.enable_dft_loss or self.args.enable_channel_loss
                or self.template.sequence_parallel_size > 1) and 'labels' in inputs:
            if self.args.use_liger_kernel:
                logger.warning_once('The cross_entropy loss function defined in Liger Kernel will not '
                                    'take effect, potentially leading to increased GPU memory consumption.')
            labels = inputs.pop('labels')
        outputs = model(**inputs)
        if getattr(outputs, 'aux_loss', None) is not None:
            mode = 'train' if self.model.training else 'eval'
            self.custom_metrics[mode]['aux_loss'].update(outputs.aux_loss)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is None:
            labels = inputs['labels']
            outputs.loss = outputs.loss.to(labels.device)
            # fix https://github.com/huggingface/transformers/issues/34263
            if num_items_in_batch is not None:
                outputs.loss = outputs.loss * ((labels[:, 1:] != -100).sum() / num_items_in_batch)

            if isinstance(outputs, dict) and 'loss' not in outputs:
                raise ValueError(
                    'The model did not return a loss from the inputs, only the following keys: '
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}.")
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs['loss'] if isinstance(outputs, dict) else outputs[0]
        else:
            outputs.loss = None
            if (self.args.enable_dft_loss or loss_scale is not None or self.args.enable_channel_loss
                    or self.template.sequence_parallel_size > 1):
                if self.template.sequence_parallel_size > 1:
                    outputs.loss = per_token_loss_func_sp(outputs, labels, enable_dft_loss=self.args.enable_dft_loss)
                else:
                    outputs.loss = per_token_loss_func(outputs, labels, enable_dft_loss=self.args.enable_dft_loss)

                if loss_scale is not None:
                    loss_scale = torch.roll(loss_scale, shifts=-1, dims=-1).view(-1)
                    outputs.loss = outputs.loss * loss_scale

                if self.args.enable_channel_loss and channels is not None:
                    mode = 'train' if self.model.training else 'eval'
                    metrics = self.custom_metrics[mode]
                    masks = torch.roll(labels, shifts=-1, dims=-1).view(-1) != -100
                    if self.template.padding_free:
                        cu_seqlens = self.get_cu_seqlens(text_position_ids, inputs.get('logits_to_keep'))
                    else:
                        cu_seqlens = torch.arange(0, labels.shape[0] + 1) * labels.shape[1]
                    for i in range(cu_seqlens.shape[0] - 1):
                        channel = channels[i]
                        slice_ = slice(cu_seqlens[i], cu_seqlens[i + 1])
                        metrics[f'loss_{channel}'].update(outputs.loss[slice_][masks[slice_]])

            unwrapped_model = self.accelerator.unwrap_model(model)
            if is_peft_available() and isinstance(unwrapped_model, PeftModel):
                model_name = unwrapped_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            # User-defined compute_loss function
            if compute_loss_func is not None:
                loss = compute_loss_func(outputs, labels, num_items_in_batch=num_items_in_batch, trainer=self)
            elif self.label_smoother is None:
                # Handle the outputs.loss generated by loss_scale.
                if num_items_in_batch is None:
                    num_items_in_batch = (labels[:, 1:] != -100).sum()
                loss = outputs.loss.sum() / num_items_in_batch
            else:
                if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                    loss = self.label_smoother(outputs, labels, shift_labels=True)
                else:
                    loss = self.label_smoother(outputs, labels)

            if self.model.model_info.is_moe_model and self.args.router_aux_loss_coef is not None:
                aux_loss = outputs.get('aux_loss')
                if aux_loss is not None:
                    if num_items_in_batch is not None:
                        aux_loss = aux_loss * ((labels[:, 1:] != -100).sum() / num_items_in_batch)
                    loss = loss + self.args.router_aux_loss_coef * aux_loss.to(loss.device)

        if getattr(self.args, 'average_tokens_across_devices',
                   False) and self.model_accepts_loss_kwargs and num_items_in_batch is not None:
            loss *= self.accelerator.num_processes

        if (outputs.logits is not None and labels is not None and self.args.tuner_backend != 'unsloth'):
            cu_seqlens = None
            if self.template.padding_free and self.args.acc_strategy == 'seq':
                cu_seqlens = self.get_cu_seqlens(text_position_ids, inputs.get('logits_to_keep'))
            # Liger does not have logits
            # Unsloth has a bug with output logits
            self._compute_acc(outputs, labels, cu_seqlens=cu_seqlens)
        return (loss, outputs) if return_outputs else loss

    def training_step(self, model, inputs, *args, **kwargs):
        with self.template.forward_context(self.model, inputs):
            return super().training_step(model, inputs, *args, **kwargs)
