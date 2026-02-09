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


def _graph_param_should_be_saved(name: str) -> bool:
    """
    Include any parameter that belongs to graph_encoder EXCEPT:
    - Shared backbone weights under 'qwen_model'
    - LoRA adapter tensors (already saved in adapter bundle)
    This avoids brittle hardcoded prefixes and captures future modules.
    """
    if not name:
        return False
    lowered = name.lower()
    if 'qwen_model' in lowered:
        return False
    if 'lora_a' in lowered or 'lora_b' in lowered or 'lora_embedding' in lowered:
        return False
    return True


class Trainer(SwiftMixin, DataLoaderMixin, HfTrainer):
    args: TrainingArguments

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._zero3_state_dict = None
        self._zero3_graph_encoder_state = None

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

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        is_zero3 = False
        try:
            import deepspeed
            is_zero3 = deepspeed.is_deepspeed_zero3_enabled()
        except Exception:
            pass

        if is_zero3 and hasattr(self, 'model_wrapped'):
            import torch.distributed as dist
            rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
            if dist.is_available() and dist.is_initialized():
                dist.barrier()

            try:
                state_dict = self.model_wrapped._zero3_consolidated_16bit_state_dict()
                self._zero3_state_dict = state_dict
            except Exception:
                self._zero3_state_dict = None

            try:
                import deepspeed
                model = self.model
                if isinstance(model, PeftModel):
                    base_model = getattr(model, 'base_model', None)
                    if base_model is not None and hasattr(base_model, 'model'):
                        model = base_model.model
                    else:
                        model = getattr(model, 'model', model)

                self._zero3_graph_encoder_state = None
                if hasattr(model, 'graph_encoder'):
                    graph_state = {}
                    graph_encoder = model.graph_encoder
                    # Collect all parameters using named_parameters() to ensure we get everything
                    # This includes both weight and bias for Linear layers
                    all_params = dict(graph_encoder.named_parameters(recurse=True))
                    filtered_out = []
                    for name, param in all_params.items():
                        if not _graph_param_should_be_saved(name):
                            filtered_out.append(name)
                            continue
                        with deepspeed.zero.GatheredParameters(param, modifier_rank=0):
                            if rank == 0:
                                graph_state[name] = param.data.cpu().clone()
                    if rank == 0:
                        self._zero3_graph_encoder_state = graph_state
                        logger.debug(f"Gathered {len(graph_state)} graph encoder parameters for ZeRO-3. "
                                   f"Sample keys: {list(graph_state.keys())[:10]}")
                        if filtered_out:
                            logger.debug(f"Filtered out {len(filtered_out)} parameters: {filtered_out[:10]}...")
                else:
                    if rank == 0:
                        self._zero3_graph_encoder_state = None
            except Exception:
                if rank == 0:
                    self._zero3_graph_encoder_state = None

            if dist.is_available() and dist.is_initialized():
                dist.barrier()

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
        # Remove graph-related keys from inputs before calling model.forward()
        # These are processed in template._post_encode and embeddings are injected into inputs_embeds
        # We keep them in a separate dict for potential use in spatial auxiliary loss
        graph_keys = {}
        for key in ['graphs', 'has_graphs', 'central_node_ids']:
            if key in inputs:
                graph_keys[key] = inputs.pop(key)
        
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
        
        # Restore graph keys to inputs for potential use in spatial auxiliary loss
        # inputs.update(graph_keys)
        
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


def _save_graph_encoder_for_trainer(trainer, output_dir: str, is_zero3: bool):
    from swift.utils import get_logger
    logger = get_logger()

    model = trainer.model
    if isinstance(model, PeftModel):
        base_model = getattr(model, 'base_model', None)
        if base_model is not None and hasattr(base_model, 'model'):
            model = base_model.model
        else:
            model = getattr(model, 'model', model)

    if not hasattr(model, 'graph_encoder'):
        return

    graph_encoder = model.graph_encoder
    import torch.distributed as dist
    is_dist = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if is_dist else 0

    graph_state: Dict[str, torch.Tensor] = {}
    pre_gathered_state = getattr(trainer, '_zero3_graph_encoder_state', None)

    if pre_gathered_state is not None and rank == 0:
        graph_state = {k: v.cpu() for k, v in pre_gathered_state.items()}
        trainer._zero3_graph_encoder_state = None
    elif is_zero3:
        try:
            import deepspeed
        except ImportError:
            logger.warning("DeepSpeed not available – cannot gather graph encoder parameters for ZeRO-3.")
            return

        for name, param in graph_encoder.named_parameters(recurse=True):
            if not _graph_param_should_be_saved(name):
                continue
            with deepspeed.zero.GatheredParameters(param, modifier_rank=0):
                if rank == 0:
                    graph_state[name] = param.data.cpu().clone()

    else:
        # Non-ZeRO-3: Use named_parameters() to get all parameters (weight, bias, etc.)
        # This ensures we get both weight and bias for Linear layers
        all_params = dict(graph_encoder.named_parameters(recurse=True))
        logger.debug(f"Graph encoder has {len(all_params)} parameters. Sample keys: {list(all_params.keys())[:10]}")
        filtered_out = []
        for name, param in all_params.items():
            if not _graph_param_should_be_saved(name):
                filtered_out.append(name)
                continue
            graph_state[name] = param.data.cpu().clone()
        if filtered_out:
            logger.debug(f"Filtered out {len(filtered_out)} parameters: {filtered_out[:10]}...")

    if rank != 0:
        return

    # Filter out empty tensors (shouldn't happen but keeps files clean)
    graph_state = {k: v for k, v in graph_state.items() if v.numel() > 0}
    if not graph_state:
        logger.info("No graph encoder parameters to save.")
        return

    # Debug: Log what we're saving
    logger.info(f"Saving {len(graph_state)} graph encoder parameters:")
    for name in sorted(graph_state.keys()):
        shape = graph_state[name].shape
        numel = graph_state[name].numel()
        logger.info(f"  {name} | shape={shape} | numel={numel}")

    graph_path = os.path.join(output_dir, 'graph_encoder.bin')
    torch.save(graph_state, graph_path)

    config = {
        'hidden_dim': getattr(graph_encoder, 'hidden_dim', None),
        'output_dim': getattr(graph_encoder, 'output_dim', None),
        'num_layers': getattr(graph_encoder, 'num_layers', None),
        'edge_dim': getattr(graph_encoder, 'edge_dim', None),
        'use_spatial_encoding': getattr(graph_encoder, 'use_spatial_encoding', None),
        'use_edge_features': getattr(graph_encoder, 'use_edge_features', None),
        'use_gat': getattr(graph_encoder, 'use_gat', None),
        'use_spatial_auxiliary': getattr(graph_encoder, 'use_spatial_auxiliary', None),
        'spatial_embed_dim': getattr(graph_encoder, 'spatial_embed_dim', None),
        'spatial_frequency_num': getattr(graph_encoder, 'spatial_frequency_num', None),
    }
    config_path = os.path.join(output_dir, 'graph_encoder_config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)

    logger.info(f"Saved graph encoder weights to {graph_path}")


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
        
        Integrates PE-GNN style local Moran's I prediction as auxiliary task.
        
        Note: Spatial auxiliary loss is OPTIONAL.
        Training will proceed normally even if it is disabled (default behavior).
        """
        # 1. Compute main loss (InfoNCE or other embedding loss)
        main_loss, outputs = super().compute_loss(model, inputs, return_outputs=True, 
                                                    num_items_in_batch=num_items_in_batch)
        
        total_loss = main_loss
        
        # 2. Compute spatial auxiliary loss if enabled (OPTIONAL)
        # If use_spatial_auxiliary=False (default), this step is skipped entirely.
        if getattr(self.args, 'use_spatial_auxiliary', False):
            spatial_loss = self._compute_spatial_auxiliary_loss(model, inputs)
            
            # Only add spatial loss if it was successfully computed (not None)
            if spatial_loss is not None:
                # Combine losses with weighting (PE-GNN style)
                spatial_weight = getattr(self.args, 'spatial_loss_weight', 0.1)
                total_loss = total_loss + spatial_weight * spatial_loss
                
                # Log losses for monitoring
                if self.state.global_step % self.args.logging_steps == 0:
                    self.log({
                        'train/main_loss': main_loss.item(),
                        'train/spatial_loss': spatial_loss.item(),
                        'train/total_loss': total_loss.item(),
                        'train/spatial_weight': spatial_weight,
                    })
            # If spatial_loss is None, training continues normally
        
        # Return total loss (which equals main_loss if optional losses are disabled)
        return (total_loss, outputs) if return_outputs else total_loss
    
    def _compute_spatial_auxiliary_loss(self, model, inputs):
        """
        Compute spatial auxiliary loss for graphs in the batch.
        
        Uses PE-GNN's local Moran's I formula:
        mi_i = (n-1) * z_i * (Σ_j w_ij * z_j) / Σ_k(z_k^2)
        
        Returns:
            spatial_loss: Scalar tensor or None if no graphs with targets
        """
        from swift.utils import get_logger
        logger = get_logger()
        
        # Check if model has graph encoder
        if not hasattr(model, 'graph_encoder'):
            return None
        
        # Check if morans_head exists
        if not hasattr(model.graph_encoder, 'morans_head'):
            logger.warning_once("Graph encoder exists but morans_head not initialized. "
                                "Set use_spatial_auxiliary=true during model initialization.")
            return None
        
        # Get graphs from inputs
        graphs = inputs.get('graphs', None)
        if graphs is None or len(graphs) == 0:
            return None
        
        # Import loss computation function
        try:
            from swift.llm.model.model.spatial_encoders import compute_spatial_auxiliary_loss
        except ImportError:
            logger.warning_once("Could not import compute_spatial_auxiliary_loss from spatial_encoders")
            return None
        
        spatial_losses = []
        all_metrics = []
        
        for graph in graphs:
            # Check if graph has ground truth values
            if not hasattr(graph, 'y') or graph.y is None:
                continue
            
            # Check if graph has coordinates for edge weight computation
            if not hasattr(graph, 'coords') or graph.coords is None:
                logger.warning_once("Graph missing 'coords' attribute needed for spatial loss")
                continue
            
            # Get node embeddings from graph encoder
            try:
                node_embeddings_list = model.graph_encoder([graph])
                node_embeddings = node_embeddings_list[0]  # [N, 1536]
                
                # Compute spatial loss for this graph using PE-GNN formula
                loss, metrics = compute_spatial_auxiliary_loss(
                    node_embeddings=node_embeddings,
                    target_values=graph.y,
                    edge_index=graph.edge_index,
                    coords=graph.coords,
                    morans_head=model.graph_encoder.morans_head,
                    distance_units='meters'
                )
                
                spatial_losses.append(loss)
                all_metrics.append(metrics)
            
            except Exception as e:
                logger.warning_once(f"Failed to compute spatial loss for graph: {e}")
                continue
        
        # Average spatial losses across batch
        if len(spatial_losses) > 0:
            avg_spatial_loss = sum(spatial_losses) / len(spatial_losses)
            
            # Log metrics (average across all graphs in batch)
            if self.state.global_step % self.args.logging_steps == 0 and len(all_metrics) > 0:
                avg_correlation = sum(m['morans_correlation'] for m in all_metrics) / len(all_metrics)
                avg_mae = sum(m['morans_mae'] for m in all_metrics) / len(all_metrics)
                
                self.log({
                    'train/morans_correlation': avg_correlation,
                    'train/morans_mae': avg_mae,
                })
            
            return avg_spatial_loss
        else:
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
        
        # Remove graph-related keys from inputs before calling model.forward()
        # These are processed in template._post_encode and embeddings are injected into inputs_embeds
        # We keep them in a separate dict for potential use in spatial auxiliary loss
        graph_keys = {}
        for key in ['graphs', 'has_graphs', 'central_node_ids']:
            if key in inputs:
                graph_keys[key] = inputs.pop(key)
        
        outputs = model(**inputs)
        
        # Restore graph keys to inputs for potential use in spatial auxiliary loss
        # inputs.update(graph_keys)
        # if getattr(outputs, 'aux_loss', None) is not None:
        #     mode = 'train' if self.model.training else 'eval'
        #     self.custom_metrics[mode]['aux_loss'].update(outputs.aux_loss)
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
