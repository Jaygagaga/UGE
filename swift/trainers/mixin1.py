# Copyright (c) Alibaba, Inc. and its affiliates.
# Part of the implementation is borrowed from huggingface/transformers.
import collections
import inspect
import logging
import os
import random
import re
import shutil
import time
import types
import warnings
from contextlib import contextmanager
from copy import copy
from functools import partial, wraps
from types import MethodType
from typing import Callable, Dict, List, Optional, Tuple, Union

import datasets
import numpy as np
import safetensors
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.utils.checkpoint
import transformers
from datasets import Dataset as HfDataset
from modelscope import check_local_model_is_latest
from packaging import version
from peft import PeftModel
from torch.nn import Module
from torch.utils.data import DataLoader
from transformers import PreTrainedModel
from transformers.data.data_collator import DataCollator
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers.modeling_utils import unwrap_model
from transformers.trainer import (OPTIMIZER_NAME, PREFIX_CHECKPOINT_DIR, SCHEDULER_NAME, TRAINER_STATE_NAME,
                                  ParallelMode, Trainer, TrainerCallback, reissue_pt_warnings)
from transformers.trainer_utils import IntervalStrategy

from swift.hub import get_hub
from swift.llm import BatchSamplerShard, DataLoaderDispatcher, DataLoaderShard, Template, get_llm_model
from swift.llm.utils import update_generation_config_eos_token
from swift.plugin import MeanMetric, compute_acc, extra_tuners, get_loss_func, get_metric
from swift.tuners import SwiftModel
from swift.utils import get_current_device, get_logger, is_dist, is_mp, is_mp_ddp, ms_logger_context, seed_worker
from ..llm.model.patcher import get_lm_head_model, revert_padding_free, transformers_seq_cls_forward
from .arguments import TrainingArguments
from .utils import can_return_loss, find_labels, get_function, is_instance_of_ms_model

try:
    from trl import AutoModelForCausalLMWithValueHead
except (ImportError, RuntimeError):
    AutoModelForCausalLMWithValueHead = None

logger = get_logger()


class SwiftMixin:
    FLASH_CKPT_WAIT_TIMEOUT = 1800

    def __init__(self,
                 model: Union[PreTrainedModel, Module] = None,
                 args: TrainingArguments = None,
                 data_collator: Optional[DataCollator] = None,
                 train_dataset: Optional[HfDataset] = None,
                 eval_dataset: Optional[Union[HfDataset, Dict[str, HfDataset]]] = None,
                 template: Optional[Template] = None,
                 model_init: Optional[Callable[[], PreTrainedModel]] = None,
                 callbacks: Optional[List[TrainerCallback]] = None,
                 optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
                 **kwargs) -> None:
        if not hasattr(train_dataset, '__len__') and args.dataloader_num_workers > 1:
            args.dataloader_num_workers = 1
            logger.warning('Using IterableDataset, setting args.dataloader_num_workers to 1.')
        self.compute_loss_func = None  # Compatible with the older version of transformers

        if args.check_model and hasattr(model, 'model_dir'):
            with ms_logger_context(logging.CRITICAL), self._patch_timeout():
                config_info = self._collect_config_info()
                config_info.update({
                    'invoked_by': 'local_trainer',
                    'third_party': 'swift',
                    'trainer_class': self.__class__.__name__,
                })
                check_local_model_is_latest(model.model_dir, user_agent=config_info)
        if eval_dataset is None and args:
            if getattr(args, 'eval_dataset', None):
                # Avoid trainer throwing errors.
                eval_dataset = []
            else:
                args.evaluation_strategy = IntervalStrategy.NO
                args.eval_strategy = IntervalStrategy.NO

        def _get_mean_metric():
            return MeanMetric(nan_value=None, device=args.device)

        self.custom_metrics = {
            'train': collections.defaultdict(_get_mean_metric),
            'eval': collections.defaultdict(_get_mean_metric)
        }
        self.template = template
        self.hub = get_hub()

        self.model_meta = model.model_meta

        kwargs.update(self.create_loss_and_metric(args))
        trainer_parameters = inspect.signature(Trainer.__init__).parameters
        tokenizer_key = 'processing_class' if 'processing_class' in trainer_parameters else 'tokenizer'
        kwargs[tokenizer_key] = template.tokenizer
        with self.hub.patch_hub():
            super().__init__(
                model=model,
                args=args,
                data_collator=data_collator,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                model_init=model_init,
                callbacks=callbacks,
                optimizers=optimizers,
                **kwargs)

        if get_function(model.__class__.forward) is not get_function(model.forward):
            self.label_names = find_labels(model)
            self.can_return_loss = can_return_loss(model)
        self.label_names = self.label_names or ['labels']
        self.start_time = time.time()
        self._fix_gradient_checkpointing()
        self._patch_tasks()
        update_generation_config_eos_token(self.model.generation_config, self.template)
        if getattr(self.model, 'origin_generation_config', None):
            self.model.origin_generation_config.eos_token_id = self.model.generation_config.eos_token_id
        if self.args.resume_only_model and self.args.ignore_data_skip:
            # The weights have already been loaded outside the trainer,
            # so reading train_state is skipped here.
            self.args.resume_from_checkpoint = None

    @contextmanager
    def _patch_timeout(self):
        from modelscope.hub.api import HubApi
        __init__ = HubApi.__init__

        def __new_init__(self, *args, **kwargs):
            timeout = kwargs.get('timeout')
            if timeout is not None and timeout > 5:
                kwargs['timeout'] = 5
            __init__(self, *args, **kwargs)

        HubApi.__init__ = __new_init__

        try:
            yield
        finally:
            HubApi.__init__ = __init__

    def _collect_config_info(self) -> Dict[str, str]:
        """
        Collects trainer-specific configuration details.

        Subclasses can override this method to provide additional configuration
        information for model compatibility verification.

        Returns:
            Dict[str, str]: Configuration parameters as key-value pairs.
        """
        return {}

    @property
    def tokenizer(self):
        # compat transformers5.0
        return self.processing_class

    @contextmanager
    def _patch_deepspeed_load_checkpoint(self):
        from transformers import trainer
        if not self.args.resume_from_checkpoint or not self.args.resume_only_model or not hasattr(
                trainer, 'deepspeed_load_checkpoint'):
            yield
            return
        origin_deepspeed_load_checkpoint = trainer.deepspeed_load_checkpoint

        def deepspeed_load_checkpoint(*args, **kwargs):
            try:
                return origin_deepspeed_load_checkpoint(*args, **kwargs)
            except Exception as e:
                logger.warning('Failed to call deepspeed_load_checkpoint function. '
                               f'If `--resume_only_model true` is set, this warning can be ignored. {e}.')

        trainer.deepspeed_load_checkpoint = deepspeed_load_checkpoint

        try:
            yield
        finally:
            trainer.deepspeed_load_checkpoint = origin_deepspeed_load_checkpoint

    def get_use_logits_to_keep(self, default_value: bool = True):
        use_logits_to_keep = self.args.use_logits_to_keep
        if use_logits_to_keep is None:
            base_model = self.template.get_base_model(self.model)
            use_logits_to_keep = (not self.model.model_meta.is_multimodal
                                  and 'logits_to_keep' in inspect.signature(base_model.forward).parameters
                                  and default_value)
        logger.info_once(f'use_logits_to_keep: {use_logits_to_keep}')
        return use_logits_to_keep

    def _save_initial_model(self, output_dir):
        # pissa/olora/lora-ga
        model = unwrap_model(self.model)
        if isinstance(model, PeftModel):
            config = model.peft_config.get('default')
            init_lora_weights = getattr(config, 'init_lora_weights', None)
            if (isinstance(init_lora_weights, str)
                    and any(s in init_lora_weights for s in ('pissa', 'olora', 'lora-ga'))):
                config.init_lora_weights = True
                model.save_pretrained(os.path.join(output_dir, 'initial_model'))
                config.init_lora_weights = init_lora_weights

    def _save_converted_model(self, output_dir):
        # pissa/olora/lora-ga
        model = unwrap_model(self.model)
        if isinstance(model, PeftModel):
            config = model.peft_config.get('default')
            init_lora_weights = getattr(config, 'init_lora_weights', None)
            if isinstance(init_lora_weights, str):
                config = copy(config)
                os.makedirs(os.path.join(output_dir, 'converted'), exist_ok=True)
                if 'lora-ga' in init_lora_weights:
                    try:
                        from lora_ga.entrypoint import LoraGAContext
                        with LoraGAContext(model):
                            model.save_pretrained(
                                os.path.join(output_dir, 'converted', 'default'),
                                path_initial_model_for_weight_conversion=os.path.join(
                                    os.path.dirname(output_dir), 'initial_model'),
                            )
                            model.peft_config['default'] = config
                    except ImportError as e:
                        error_message = """
                        Since 'LoRA-GA' is not implemented by PEFT, you will need to install it directly from GitHub.
                        Command: 'pip install git+https://github.com/lxline/LoRA-GA.git'.
                        """
                        logger.info(error_message)
                        raise RuntimeError(error_message) from e
                elif 'pissa' in init_lora_weights or 'olora' in init_lora_weights:
                    model.save_pretrained(
                        os.path.join(output_dir, 'converted', 'default'),
                        path_initial_model_for_weight_conversion=os.path.join(
                            os.path.dirname(output_dir), 'initial_model'),
                    )
                    model.peft_config['default'] = config

    def _load_rng_state(self, *args, **kwargs):
        if self.args.resume_only_model:
            return
        return super()._load_rng_state(*args, **kwargs)

    def _load_optimizer_and_scheduler(self, *args, **kwargs):
        if self.args.resume_only_model:
            return
        try:
            super()._load_optimizer_and_scheduler(*args, **kwargs)
        except (IndexError, KeyError, AttributeError) as e:
            # Scheduler config mismatch (e.g., different parameter groups when adding graph encoder)
            # This is common in continual learning when introducing new modalities or changing datasets
            from swift.utils import get_logger
            import os
            logger = get_logger()
            logger.warning(
                f"⚠️ Scheduler loading failed: {e}. "
                "This is expected when adding new modalities (e.g., graph encoder) or changing datasets. "
                "Starting scheduler from step 0 for the new task."
            )
            
            # Load optimizer (usually works even with new parameter groups)
            try:
                if hasattr(super(), '_load_optimizer'):
                    super()._load_optimizer(*args, **kwargs)
            except Exception:
                pass
            
            # IMPORTANT: Do NOT restore scheduler step when:
            # 1. Scheduler config mismatch (different parameter groups = different task/dataset)
            # 2. Adding new modalities (graph encoder) = new task
            # 3. Different dataset = new task
            # 
            # The scheduler step should only be restored when continuing the SAME task/dataset.
            # When starting a new task (e.g., image-text → image-text+graph), the scheduler
            # should start from step 0, not from the previous task's step count.
            #
            # If you want to continue the scheduler from a previous checkpoint for the SAME task,
            # ensure the optimizer structure matches (same parameter groups).
            
            logger.info(
                "ℹ️ Scheduler will start from step 0. "
                "This is correct when training on a new dataset or adding new modalities."
            )
        
        if is_mp_ddp():
            # fix mp+ddp adamw
            for v in self.optimizer.state.values():
                if 'step' in v:
                    # not on the same device
                    device_set = set([t.device for t in v.values()]) - {v['step'].device, torch.device('cpu')}
                    if len(device_set) >= 1:
                        v['step'] = v['step'].to('cpu')

    def _save_model(self, output_dir: Optional[str] = None, state_dict=None):
        # model
        supported_classes = (SwiftModel, PreTrainedModel, PeftModel)
        supported_names = ('SentenceTransformer', )
        if AutoModelForCausalLMWithValueHead is not None:
            supported_classes = supported_classes + (AutoModelForCausalLMWithValueHead, )
        save_safetensors = self.args.save_safetensors
        use_flash_ckpt = self.args.use_flash_ckpt

        if not isinstance(self.model, supported_classes) and self.model.__class__.__name__ not in supported_names:
            if state_dict is None:
                state_dict = self.model.state_dict()

            _unwrap_model = unwrap_model(self.model)
            if isinstance(_unwrap_model, supported_classes):
                if use_flash_ckpt:
                    _unwrap_model.save_pretrained(
                        output_dir,
                        state_dict=state_dict,
                        safe_serialization=False,
                        save_function=self.flash_checkpointer.ckpt_agent.save)
                else:
                    _unwrap_model.save_pretrained(
                        output_dir, state_dict=state_dict, safe_serialization=save_safetensors)
            else:
                logger.info('Trainer.model is not a `PreTrainedModel`, only saving its state dict.')
                if use_flash_ckpt:
                    self.flash_checkpointer.ckpt_agent.save(state_dict, os.path.join(output_dir, 'pytorch_model.bin'))
                else:
                    if save_safetensors:
                        safetensors.torch.save_file(state_dict, os.path.join(output_dir, 'model.safetensors'))
                    else:
                        torch.save(state_dict, os.path.join(output_dir, 'pytorch_model.bin'))
        elif AutoModelForCausalLMWithValueHead and isinstance(self.model, AutoModelForCausalLMWithValueHead):
            # save reward model
            state_dict = self.model.state_dict()
            decoder_state_dict, v_head_state_dict = {}, {}
            for name, param in state_dict.items():
                if name.startswith('v_head.'):
                    v_head_state_dict[name] = param
                else:
                    decoder_state_dict[name.replace('pretrained_model.', '', 1)] = param
            self.model.pretrained_model.save_pretrained(
                output_dir, state_dict=decoder_state_dict or None, safe_serialization=save_safetensors)
            if save_safetensors:
                from safetensors.torch import save_file
                save_file(
                    v_head_state_dict, os.path.join(output_dir, 'value_head.safetensors'), metadata={'format': 'pt'})
            else:
                torch.save(v_head_state_dict, os.path.join(output_dir, 'value_head.bin'))
        elif is_instance_of_ms_model(self.model):
            if use_flash_ckpt:
                PreTrainedModel.save_pretrained(
                    self.model,
                    output_dir,
                    state_dict=state_dict,
                    safe_serialization=False,
                    save_function=self.flash_checkpointer.ckpt_agent.save)
            else:
                # modelscope save_pretrained does not support safe_serialization
                PreTrainedModel.save_pretrained(
                    self.model, output_dir, state_dict=state_dict, safe_serialization=save_safetensors)
        elif self.args.train_type in extra_tuners:
            extra_tuners[self.args.train_type].save_pretrained(
                self.model, output_dir, state_dict=state_dict, safe_serialization=save_safetensors)
        else:
            if self.model.__class__.__name__ != 'SentenceTransformer':
                logger.info(f"About to call model.save_pretrained() - model type: {type(self.model).__name__}, state_dict size: {len(state_dict) if state_dict else 0}")
                if use_flash_ckpt:
                    logger.info("Using flash checkpoint save function...")
                    self.model.save_pretrained(
                        output_dir,
                        state_dict=state_dict,
                        safe_serialization=False,
                        save_function=self.flash_checkpointer.ckpt_agent.save)
                    logger.info("✅ Completed flash checkpoint save")
                else:
                    logger.info(f"Calling model.save_pretrained() with safe_serialization={save_safetensors}...")
                    self.model.save_pretrained(output_dir, state_dict=state_dict, safe_serialization=save_safetensors)
                    logger.info("✅ Completed model.save_pretrained()")
            else:

                @contextmanager
                def save_context():
                    save_pretrained = self.model[0].auto_model.save_pretrained
                    _state_dict = {
                        key[len('0.auto_model.'):] if 'auto_model' in key else key: value
                        for key, value in state_dict.items()
                    }
                    self.model[0].auto_model.save_pretrained = partial(
                        self.model[0].auto_model.save_pretrained, state_dict=_state_dict)
                    yield
                    self.model[0].auto_model.save_pretrained = save_pretrained

                with save_context():
                    if use_flash_ckpt:
                        self.model.save_pretrained(
                            output_dir,
                            state_dict=state_dict,
                            safe_serialization=False,
                            save_function=self.flash_checkpointer.ckpt_agent.save)
                    else:
                        self.model.save_pretrained(output_dir, safe_serialization=save_safetensors)
                        # copy sentencetransformers files
                    from swift.utils import copy_files_by_pattern
                    copy_files_by_pattern(
                        self.model.model_dir, output_dir, '*.py', exclude_patterns=['model.safetensors.index.json'])
                    copy_files_by_pattern(
                        self.model.model_dir, output_dir, '*.json', exclude_patterns=['model.safetensors.index.json'])

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        """Compatible with swift and peft"""
        import torch.distributed as dist
        is_dist = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if is_dist else 0
        
        logger.info(f"Rank {rank}: SwiftMixin._save() called")
        
        # Only rank 0 should save files to avoid conflicts
        # Other ranks should skip file I/O but still participate in barriers
        if rank == 0 or not is_dist:
            # If we are executing this function, we are the process zero, so we don't check for that.
            output_dir = output_dir if output_dir is not None else self.args.output_dir
            os.makedirs(output_dir, exist_ok=True)
            
            logger.info(f"Rank {rank}: Calling _save_model()...")
            self._save_model(output_dir, state_dict)
            logger.info(f"Rank {rank}: ✅ Completed _save_model()")
            
            # training_args.bin
            logger.info(f"Rank {rank}: Saving training_args.bin...")
            try:
                # Try to save args directly
                torch.save(self.args, os.path.join(output_dir, 'training_args.bin'))
                logger.info(f"Rank {rank}: ✅ Saved training_args.bin")
            except (AttributeError, TypeError) as e:
                # If saving fails due to unpicklable attributes (e.g., DeepSpeed hooks),
                # create a clean copy without problematic attributes
                logger.warning(f"Rank {rank}: Failed to save args directly: {e}. Creating clean copy...")
                import copy
                import pickle
                import dataclasses
                
                # Create a copy and try to clean it
                try:
                    # If args is a dataclass, convert to dict and back
                    if dataclasses.is_dataclass(self.args):
                        args_dict = dataclasses.asdict(self.args)
                        # Remove any problematic values (functions, hooks, etc.)
                        cleaned_dict = {}
                        for k, v in args_dict.items():
                            # Skip non-serializable types
                            if isinstance(v, (type, types.FunctionType, types.MethodType, types.LambdaType)):
                                continue
                            try:
                                # Try to pickle the value to check if it's serializable
                                pickle.dumps(v)
                                cleaned_dict[k] = v
                            except (TypeError, pickle.PickleError):
                                logger.debug(f"Rank {rank}: Skipping unpicklable key in args: {k}")
                                continue
                        # Save as dict
                        torch.save(cleaned_dict, os.path.join(output_dir, 'training_args.bin'))
                        logger.info(f"Rank {rank}: ✅ Saved training_args.bin as dict (cleaned)")
                    else:
                        # For non-dataclass, try to remove problematic attributes
                        args_copy = copy.copy(self.args)
                        # Remove common problematic attributes
                        problematic_attrs = ['deepspeed', 'accelerator', '_deepspeed_engine']
                        for attr in problematic_attrs:
                            if hasattr(args_copy, attr):
                                try:
                                    delattr(args_copy, attr)
                                except AttributeError:
                                    pass
                        torch.save(args_copy, os.path.join(output_dir, 'training_args.bin'))
                        logger.info(f"Rank {rank}: ✅ Saved training_args.bin (cleaned copy)")
                except Exception as e2:
                    logger.error(f"Rank {rank}: ❌ Failed to save training_args.bin even after cleaning: {e2}")
                    # Continue without saving args - checkpoint can still be resumed without it
            
            logger.info(f"Rank {rank}: Calling _save_converted_model()...")
            self._save_converted_model(output_dir)
            logger.info(f"Rank {rank}: ✅ Completed _save_converted_model()")
        else:
            logger.info(f"Rank {rank}: Skipping SwiftMixin._save() file operations (rank 0 only)")
        
        # Barrier to ensure all ranks complete before returning
        if is_dist:
            logger.info(f"Rank {rank}: Waiting for all ranks to complete SwiftMixin._save()...")
            dist.barrier()
            logger.info(f"Rank {rank}: All ranks completed SwiftMixin._save()")
        # args.json
        args_path = os.path.join(os.path.dirname(output_dir), 'args.json')
        if os.path.exists(args_path):
            shutil.copy(args_path, os.path.join(output_dir, 'args.json'))
        # predict.jsonl
        predict_jsonl = os.path.join(os.path.dirname(output_dir), 'predict.jsonl')
        if os.path.exists(predict_jsonl):
            shutil.move(predict_jsonl, os.path.join(output_dir, 'predict.jsonl'))

        is_adapter = isinstance(self.model, (SwiftModel, PeftModel))
        # tokenizer - always save for graph encoder (contains special tokens)
        # For adapters, also save tokenizer if graph encoder is enabled
        should_save_tokenizer = (not is_adapter) or getattr(self.args, 'use_graph_encoder', False)
        if should_save_tokenizer:
            from swift.llm import save_checkpoint
            additional_saved_files = self.model_meta.additional_saved_files
            save_checkpoint(
                None,
                self.template.processor,
                output_dir,
                model_dirs=[self.model.model_dir] if hasattr(self.model, 'model_dir') else [],
                additional_saved_files=additional_saved_files)
            if getattr(self.model, 'origin_generation_config', None):
                self.model.origin_generation_config.save_pretrained(output_dir)

    def _rotate_flash_checkpoints(self, use_mtime=False, output_dir=None) -> None:
        if (self.args.save_total_limit is None or self.args.save_total_limit <= 0):
            return

        last_step = self._get_last_checkpoint_step()

        # Check if we should delete older checkpoint(s)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=use_mtime, output_dir=output_dir)

        valid_checkpoints = []
        for path in checkpoints_sorted:
            regex_match = re.match(f'.*{PREFIX_CHECKPOINT_DIR}-([0-9]+)', path)
            if regex_match is not None and regex_match.groups() is not None:
                step = int(regex_match.groups()[0])
                if step <= last_step:
                    valid_checkpoints.append(path)

        if len(valid_checkpoints) <= self.args.save_total_limit:
            return

        # If save_total_limit=1 with load_best_model_at_end=True,
        # we could end up deleting the last checkpoint, which
        # should be avoided and allow resuming
        save_total_limit = self.args.save_total_limit
        if (self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1
                and valid_checkpoints[-1] != self.state.best_model_checkpoint):
            save_total_limit = 2

        number_of_checkpoints_to_delete = max(0, len(valid_checkpoints) - save_total_limit)
        checkpoints_to_be_deleted = valid_checkpoints[:number_of_checkpoints_to_delete]
        for checkpoint in checkpoints_to_be_deleted:
            logger.info(f'Deleting older checkpoint [{checkpoint}] '
                        f'due to save_total_limit = {self.args.save_total_limit}.')
            shutil.rmtree(checkpoint, ignore_errors=True)

    def get_last_checkpoint(self):
        """
        Get the path of the last complete checkpoint. Some latter directories
        may not have the complete checkpoint because the asynchronous
        persistence may not finish. The step in the `dlrover_latest.txt` is
        the last step of complete checkpoint. We can get the path by the step.
        """
        step = self._get_last_checkpoint_step()
        if step == 0:
            return False
        checkpoint_folder = f'{PREFIX_CHECKPOINT_DIR}-{step}'
        ckpt_dir = os.path.join(self.args.output_dir, checkpoint_folder)
        return ckpt_dir

    def _get_last_checkpoint_step(self):
        tracer_file = os.path.join(self.args.output_dir, 'dlrover_latest.txt')
        if not os.path.exists(tracer_file):
            return 0
        with open(tracer_file, 'r') as f:
            step = int(f.read())
        return step

    def wait_latest_checkpoint(self, timeout=FLASH_CKPT_WAIT_TIMEOUT):
        """
        Wait for the latest checkpoint.
        Args:
            timeout (second): The timeout to wait.
        """
        self.flash_checkpointer.async_save_engine.wait_latest_checkpoint(timeout)

    def _fix_zero3_gather_all_parameters(self) -> None:
        if is_deepspeed_zero3_enabled() and not hasattr(self.deepspeed, '_zero3_consolidated_16bit_state_dict_origin'):
            parameters = inspect.signature(self.deepspeed._zero3_consolidated_16bit_state_dict).parameters
            if 'exclude_frozen_parameters' in parameters:

                def _zero3_consolidated_16bit_state_dict(model, exclude_frozen_parameters=False):
                    unwrapped = unwrap_model(model)
                    exclude_frozen_parameters = False
                    if isinstance(unwrapped, SwiftModel) and unwrapped.has_additional_modules:
                        exclude_frozen_parameters = True
                    if isinstance(unwrapped, PeftModel):
                        exclude_frozen_parameters = True
                    return model._zero3_consolidated_16bit_state_dict_origin(exclude_frozen_parameters)

                self.deepspeed._zero3_consolidated_16bit_state_dict_origin = (
                    self.deepspeed._zero3_consolidated_16bit_state_dict)
                self.deepspeed._zero3_consolidated_16bit_state_dict = MethodType(_zero3_consolidated_16bit_state_dict,
                                                                                 self.deepspeed)

    def _save_checkpoint(self, *args, **kwargs):
        import torch.distributed as dist
        is_dist = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if is_dist else 0
        
        logger.info(f"Rank {rank}: _save_checkpoint called (step {getattr(self.state, 'global_step', 'unknown')})")
        print(f"[RANK {rank}] _save_checkpoint() ENTRY - should_save={getattr(self.control, 'should_save', False)}", flush=True)
        logger.info(f"Rank {rank}: should_save={getattr(self.control, 'should_save', False)}")
        
        # Set checkpoint path on all ranks (needed for state tracking)
        self.state.last_model_checkpoint = os.path.join(self.args.output_dir, f'checkpoint-{self.state.global_step}')
        self._fix_zero3_gather_all_parameters()

        # CRITICAL: Ensure ALL ranks participate in _save() to avoid deadlocks.
        # HuggingFace Trainer only calls _save_checkpoint() on rank 0, but we need all ranks
        # to participate in the save barriers. We check if _swift_should_save_synced is set
        # (which means all ranks know they should save) or if this is rank 0 with should_save=True.
        should_save = getattr(self.control, 'should_save', False)
        should_save_synced = getattr(self.args, '_swift_should_save_synced', False)
        
        # All ranks should participate if:
        # 1. This is rank 0 and should_save is True (normal case)
        # 2. OR should_save_synced is True (all ranks know they should save)
        if should_save or should_save_synced:
            output_dir = self.state.last_model_checkpoint
            
            # CRITICAL: Call save_model() FIRST to gather ZeRO-3 state dict (ALL ranks participate)
            # This follows the normal HuggingFace Trainer order: save_model() → _save()
            # save_model() gathers the full state dict and graph encoder parameters
            # _save() then uses the pre-gathered state to save
            print(f"[RANK {rank}] Calling save_model() to gather ZeRO-3 state (ALL ranks participate)...", flush=True)
            logger.info(f"Rank {rank}: Calling save_model() to gather ZeRO-3 state (ALL ranks participate)...")
            self.save_model(output_dir, _internal_call=True)
            print(f"[RANK {rank}] ✅ Completed save_model() - ZeRO-3 state gathered", flush=True)
            logger.info(f"Rank {rank}: ✅ Completed save_model() - ZeRO-3 state gathered")
            
            # Now call _save() to save model + graph encoder using pre-gathered state
            print(f"[RANK {rank}] Calling _save() on all ranks to ensure participation...", flush=True)
            logger.info(f"Rank {rank}: Calling _save() on all ranks to ensure participation...")
            self._save(output_dir)
            # Set flag AFTER saving so that super()._save_checkpoint() won't call _save() again
            # (super()._save_checkpoint() calls _save() internally, but we've already saved)
            object.__setattr__(self.args, '_swift_already_saved', True)
            print(f"[RANK {rank}] ✅ Completed _save() call in _save_checkpoint()", flush=True)
            logger.info(f"Rank {rank}: ✅ Completed _save() call in _save_checkpoint()")
        else:
            print(f"[RANK {rank}] Skipping _save() in _save_checkpoint() (should_save=False, should_save_synced=False)", flush=True)
            logger.info(f"Rank {rank}: Skipping _save() in _save_checkpoint() (should_save=False, should_save_synced=False)")
        
        # OPTIONAL: Skip optimizer/scheduler saving for final stage training
        # This is useful when you don't need to resume training and want to avoid
        # the expensive optimizer state gathering in DeepSpeed ZeRO-3
        # self.args is the TrainingArguments object, so we check it directly
        skip_optimizer_save = getattr(self.args, 'skip_optimizer_state_save', False)
        
        # Also check if it's set on the original TrainArguments (via training_args attribute)
        if not skip_optimizer_save and hasattr(self.args, 'training_args'):
            original_args = self.args.training_args
            if hasattr(original_args, 'skip_optimizer_state_save'):
                skip_optimizer_save = original_args.skip_optimizer_state_save
                print(f"[RANK {rank}] Found skip_optimizer_state_save={skip_optimizer_save} from original TrainArguments", flush=True)
                logger.info(f"Rank {rank}: Found skip_optimizer_state_save={skip_optimizer_save} from original TrainArguments")
        
        if skip_optimizer_save:
            print(f"[RANK {rank}] ⏭️ Skipping optimizer/scheduler save (skip_optimizer_state_save=True)", flush=True)
            logger.info(f"Rank {rank}: ⏭️ Skipping optimizer/scheduler save (skip_optimizer_state_save=True)")
            result = None
        else:
            # Now call super()._save_checkpoint() which will handle optimizer, scheduler, etc.
            # It will call _save() again, but our _save() will skip the actual save and just participate in barriers
            # 
            # CRITICAL: This is where rank 0 might hang if saving takes too long or encounters issues.
            # Other ranks are waiting at the barrier in _maybe_log_save_evaluate() for rank 0 to complete.
            print(f"[RANK {rank}] About to call super()._save_checkpoint() for optimizer/scheduler save...", flush=True)
            logger.info(f"Rank {rank}: About to call super()._save_checkpoint() for optimizer/scheduler save...")
            try:
                if self.args.use_flash_ckpt:
                    print(f"[RANK {rank}] Using flash checkpoint...", flush=True)
                    logger.info(f"Rank {rank}: Using flash checkpoint...")
                    result = self._save_flash_checkpoint(*args, **kwargs)
                    print(f"[RANK {rank}] ✅ Completed _save_flash_checkpoint()", flush=True)
                    logger.info(f"Rank {rank}: ✅ Completed _save_flash_checkpoint()")
                else:
                    print(f"[RANK {rank}] Calling super()._save_checkpoint()...", flush=True)
                    logger.info(f"Rank {rank}: Calling super()._save_checkpoint() for optimizer/scheduler save...")
                    result = super()._save_checkpoint(*args, **kwargs)
                    print(f"[RANK {rank}] ✅ Completed super()._save_checkpoint()", flush=True)
                    logger.info(f"Rank {rank}: ✅ Completed super()._save_checkpoint()")
            except Exception as e:
                print(f"[RANK {rank}] ❌ Error in _save_checkpoint(): {e}", flush=True)
                logger.error(f"Rank {rank}: ❌ Error in _save_checkpoint(): {e}")
                import traceback
                logger.error(traceback.format_exc())
                raise
            finally:
                # Clean up flags
                if hasattr(self.args, '_swift_already_saved'):
                    object.__delattr__(self.args, '_swift_already_saved')
        
        print(f"[RANK {rank}] Checkpoint save completed: {self.state.last_model_checkpoint}", flush=True)
        logger.info(f'Rank {rank}: Checkpoint save completed: {self.state.last_model_checkpoint}')
        
        # CRITICAL: After checkpoint save, ensure DeepSpeed ZeRO-3 parameters are properly released
        # This prevents hangs when DeepSpeed tries to gather parameters for the next training step
        if is_dist:
            # Clear any cached state dicts that might interfere with next training step
            if hasattr(self, '_zero3_state_dict') and self._zero3_state_dict is not None:
                logger.info(f"Rank {rank}: Clearing cached _zero3_state_dict after checkpoint save")
                self._zero3_state_dict = None
            if hasattr(self, '_zero3_graph_encoder_state') and self._zero3_graph_encoder_state is not None:
                logger.info(f"Rank {rank}: Clearing cached _zero3_graph_encoder_state after checkpoint save")
                self._zero3_graph_encoder_state = None
            
            # Synchronize all ranks and ensure CUDA operations are complete
            # This ensures all parameter gathering operations from checkpoint save are fully released
            logger.info(f"Rank {rank}: Synchronizing all ranks after checkpoint save to release ZeRO-3 parameters...")
            if torch.cuda.is_available():
                torch.cuda.synchronize()  # Ensure all CUDA operations complete
            dist.barrier()  # Synchronize all ranks
            logger.info(f"Rank {rank}: All ranks synchronized, ZeRO-3 parameters released")
        
        # CRITICAL: Even though _save_checkpoint() is only called on rank 0, we add a barrier here
        # to ensure rank 0 signals completion. However, this barrier won't work because other ranks
        # never enter this function. The real synchronization happens in _maybe_log_save_evaluate()
        # after super() returns. But we add logging here to help debug if rank 0 hangs.
        # 
        # The actual barrier that synchronizes all ranks is in _maybe_log_save_evaluate() after
        # super()._maybe_log_save_evaluate() returns. If rank 0 hangs here, other ranks will be
        # stuck at that barrier, which is the expected behavior.
        
        return result

    def _save_flash_checkpoint(self, model, trial, metrics=None):
        from transformers.trainer import DeepSpeedSchedulerWrapper
        from transformers.trainer_utils import SaveStrategy
        from dlrover.trainer.torch.flash_checkpoint.hf_trainer import HfDdpCheckpointer, HfDeepSpeedCheckpointer
        run_dir = self._get_output_dir(trial=trial)

        torch_native_save = torch.save

        # Save model checkpoint
        checkpoint_folder = f'{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}'
        output_dir = os.path.join(run_dir, checkpoint_folder)

        if not hasattr(self, 'flash_checkpointer'):
            if self.is_deepspeed_enabled:
                self.flash_checkpointer = HfDeepSpeedCheckpointer(self.model_wrapped, run_dir)
            elif not self.is_deepspeed_enabled and not self.is_fsdp_enabled:
                self.flash_checkpointer = HfDdpCheckpointer(run_dir)
            else:
                raise ValueError('Flash Checkpoint only supports DeepSpeed or DDP.')

        if self.hp_search_backend is None and trial is None:
            self.store_flos()

        torch.save = self.flash_checkpointer.ckpt_agent.save
        self.save_model(output_dir, _internal_call=True)
        if self.is_deepspeed_enabled:
            self.model_wrapped.save_checkpoint(output_dir)

        elif (self.args.should_save and not self.is_deepspeed_enabled and not self.is_fsdp_enabled):
            # deepspeed.save_checkpoint above saves model/optim/sched
            torch.save(
                self.optimizer.state_dict(),
                os.path.join(output_dir, OPTIMIZER_NAME),
            )

        # Save SCHEDULER & SCALER
        is_deepspeed_custom_scheduler = (
            self.is_deepspeed_enabled and not isinstance(self.lr_scheduler, DeepSpeedSchedulerWrapper))
        if self.args.should_save and (not self.is_deepspeed_enabled or is_deepspeed_custom_scheduler):
            with warnings.catch_warnings(record=True) as caught_warnings:
                torch.save(
                    self.lr_scheduler.state_dict(),
                    os.path.join(output_dir, SCHEDULER_NAME),
                )
            reissue_pt_warnings(caught_warnings)
        if self.args.save_strategy in [SaveStrategy.STEPS, SaveStrategy.EPOCH] and self.state.best_global_step:
            best_checkpoint_folder = f'{PREFIX_CHECKPOINT_DIR}-{self.state.best_global_step}'
            best_checkpoint_dir = os.path.join(run_dir, best_checkpoint_folder)

            if os.path.exists(best_checkpoint_dir):
                self.state.best_model_checkpoint = best_checkpoint_dir

        # Save the Trainer state
        if self.args.should_save:
            # Update `ExportableState` callbacks and `TrainerControl` state to where we are currently
            from transformers.trainer_callback import ExportableState
            for cb in [
                    cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
            ]:
                cb_name = cb.__class__.__name__
                cb_state = cb.state()
                if isinstance(self.state.stateful_callbacks[cb_name], list):
                    self.state.stateful_callbacks[cb_name].append(cb_state)
                else:
                    self.state.stateful_callbacks[cb_name] = cb_state
            self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))
        # Save RNG state in non-distributed training
        rng_states = {
            'python': random.getstate(),
            'numpy': np.random.get_state(),
            'cpu': torch.random.get_rng_state(),
        }
        if torch.cuda.is_available():
            if self.args.parallel_mode == ParallelMode.DISTRIBUTED:
                # In non distributed, we save the global
                # CUDA RNG state (will take care of DataParallel)
                rng_states['cuda'] = torch.cuda.random.get_rng_state_all()
            else:
                rng_states['cuda'] = torch.cuda.random.get_rng_state()

        # A process can arrive here before the process 0 has a chance to
        # save the model, in which case output_dir may not yet exist.
        os.makedirs(output_dir, exist_ok=True)

        if self.args.world_size <= 1:
            torch.save(rng_states, os.path.join(output_dir, 'rng_state.pth'))
        else:
            torch.save(
                rng_states,
                os.path.join(output_dir, f'rng_state_{self.args.process_index}.pth'),
            )

        torch.save = torch_native_save
        success = self.flash_checkpointer.save_checkpoint_to_storage(self.state.global_step)
        if not success:
            logger.info(f'Skip saving the checkpoint of step {self.state.global_step} '
                        'because the latest checkpoint is not finished.')
            shutil.rmtree(output_dir, ignore_errors=True)

        if self.args.push_to_hub:
            self._push_from_checkpoint(output_dir)

        # Maybe delete some older checkpoints.
        if self.args.should_save:
            self._rotate_flash_checkpoints(use_mtime=True, output_dir=run_dir)

    @staticmethod
    @contextmanager
    def _fix_grad_norm_nan():
        from accelerate import Accelerator
        origin_clip_grad_norm_ = Accelerator.clip_grad_norm_

        def clip_grad_norm_(self, parameters, *args, **kwargs):
            # If NaN occurs, ignore weight updates.
            parameters = list(parameters)
            grad_norm = origin_clip_grad_norm_(self, parameters, *args, **kwargs)
            if isinstance(grad_norm, torch.Tensor) and grad_norm.isnan().item():
                for p in parameters:
                    p.grad = None
            return grad_norm

        Accelerator.clip_grad_norm_ = clip_grad_norm_
        try:
            yield
        finally:
            Accelerator.clip_grad_norm_ = origin_clip_grad_norm_

    def _patch_tasks(self):
        if isinstance(self.model, PeftModel):
            model = self.model.model
        else:
            model = self.model
        if 'SentenceTransformer' in model.__class__.__name__:

            def forward_transformer(transformer, features: Dict[str, torch.Tensor],
                                    **kwargs) -> Dict[str, torch.Tensor]:
                trans_features = {
                    key: value
                    for key, value in features.items()
                    if key in ['input_ids', 'attention_mask', 'token_type_ids', 'inputs_embeds', 'position_ids']
                }

                outputs = transformer.auto_model(**trans_features, **kwargs, return_dict=True)
                token_embeddings = outputs[0]
                features['token_embeddings'] = token_embeddings

                if transformer.auto_model.config.output_hidden_states and 'hidden_states' in outputs:
                    features['all_layer_embeddings'] = outputs['hidden_states']

                return features

            from sentence_transformers.models import Transformer
            if isinstance(model[0], Transformer):
                model[0].forward = MethodType(forward_transformer, model[0])

            def forward_sentence_transformer(sentence_transformer, **kwargs) -> Dict[str, torch.Tensor]:
                input = kwargs
                kwargs = {}
                for idx, (module_name, module) in enumerate(sentence_transformer.named_children()):
                    from sentence_transformers.models import Router
                    if isinstance(module, Router):
                        module_kwargs = kwargs
                    else:
                        module_kwarg_keys = []
                        if sentence_transformer.module_kwargs is not None:
                            module_kwarg_keys = sentence_transformer.module_kwargs.get(module_name, [])
                        module_kwargs = {
                            key: value
                            for key, value in kwargs.items() if key in module_kwarg_keys or (
                                hasattr(module, 'forward_kwargs') and key in module.forward_kwargs)
                        }
                    output = module(input, **module_kwargs)
                    if idx == 0 and self.args.padding_free:
                        output = revert_padding_free(output, input, self.args.padding_side)
                    input = output
                return {'last_hidden_state': input['sentence_embedding']}

            model.forward = MethodType(forward_sentence_transformer, model)
        elif self.args.padding_free:
            if self.args.task_type == 'embedding':
                llm_model = get_lm_head_model(self.model, model_meta=self.model.model_meta)

                def revert_padding_free_hook(module, args, input, output):
                    return revert_padding_free(output, input, self.args.padding_side)

                llm_model.register_forward_hook(revert_padding_free_hook, with_kwargs=True, prepend=True)
            elif self.args.task_type == 'seq_cls':
                llm_model = get_llm_model(self.model, model_meta=self.model.model_meta)

                @wraps(model.forward.__func__)
                def seq_cls_forward(model, *args, **kwargs):

                    def inner_forward(*args, **kwargs):
                        output = llm_model.forward(*args, **kwargs)
                        return revert_padding_free(output, kwargs, self.args.padding_side)

                    return transformers_seq_cls_forward(
                        model, *args, origin_forward=inner_forward, padding_side=self.args.padding_side, **kwargs)

                model.forward = MethodType(seq_cls_forward, model)
            elif self.args.task_type == 'reranker':
                llm_model = get_llm_model(self.model, model_meta=self.model.model_meta)

                @wraps(model.forward.__func__)
                def reranker_forward(model, *args, **kwargs):

                    def inner_forward(*args, **kwargs):
                        output = llm_model.forward(*args, **kwargs)
                        return revert_padding_free(output, kwargs, self.args.padding_side)

                    padding_free_fn = getattr(model, 'padding_free_fn', None)
                    if callable(padding_free_fn):
                        output = inner_forward(*args, **kwargs)
                        return padding_free_fn(output, kwargs, self.args.padding_side)

                    return transformers_seq_cls_forward(
                        model, *args, origin_forward=inner_forward, padding_side=self.args.padding_side, **kwargs)

                model.forward = MethodType(reranker_forward, model)
            elif self.args.task_type == 'generative_reranker':
                llm_model = get_llm_model(self.model, model_meta=self.model.model_meta)

                def revert_padding_free_hook(module, args, input, output):
                    return revert_padding_free(output, input, self.args.padding_side)

                llm_model.register_forward_hook(revert_padding_free_hook, with_kwargs=True, prepend=True)

    def _fix_gradient_checkpointing(self):
        # fix use_reentrant
        if hasattr(torch.utils.checkpoint, '_old_checkpoint'):  # avoid double patching
            return
        args = self.args
        if args.gradient_checkpointing_kwargs:
            use_reentrant_ = args.gradient_checkpointing_kwargs.get('use_reentrant')
        else:
            use_reentrant_ = None
        if use_reentrant_ is None:
            if is_dist() and not self.is_deepspeed_enabled and not self.is_fsdp_enabled:
                use_reentrant_ = False
            else:
                use_reentrant_ = True
        logger.info(f'use_reentrant: {use_reentrant_}')
        _old_checkpoint = torch.utils.checkpoint.checkpoint

        @wraps(_old_checkpoint)
        def _new_checkpoint(*args, use_reentrant=None, **kwargs):
            return _old_checkpoint(*args, use_reentrant=use_reentrant_, **kwargs)

        torch.utils.checkpoint._old_checkpoint = _old_checkpoint
        torch.utils.checkpoint.checkpoint = _new_checkpoint
        try:
            # Fix the old version of transformers.
            import transformers.modeling_utils
            transformers.modeling_utils.checkpoint = _new_checkpoint
        except (ImportError, AttributeError):
            pass

    def _prepare_gradient_checkpointing(self, model) -> None:
        from swift.llm import HfConfigFactory, deep_getattr, dynamic_gradient_checkpointing
        args = self.args
        HfConfigFactory.set_model_config_attr(model, 'use_cache', False)
        if args.gradient_checkpointing or args.vit_gradient_checkpointing:
            dynamic_gradient_checkpointing(model, args.vit_gradient_checkpointing)
        gc_kwargs = {}
        parameters = inspect.signature(model.gradient_checkpointing_enable).parameters
        if 'gradient_checkpointing_kwargs' in parameters:
            gc_kwargs['gradient_checkpointing_kwargs'] = args.gradient_checkpointing_kwargs
        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable(**gc_kwargs)
            model.enable_input_require_grads()

        model_meta = model.model_meta
        model_arch = model_meta.model_arch
        if model_meta.is_multimodal and model_arch:
            for vision_tower_name in model_arch.vision_tower:
                vision_tower = deep_getattr(model, vision_tower_name)
                if hasattr(vision_tower, 'enable_input_require_grads'):
                    try:
                        if args.vit_gradient_checkpointing:
                            vision_tower.gradient_checkpointing_enable(**gc_kwargs)
                            vision_tower.enable_input_require_grads()
                        else:
                            vision_tower.gradient_checkpointing_disable()
                            vision_tower.disable_input_require_grads()
                    except (NotImplementedError, AttributeError) as e:
                        logger.warning(f'prepare gradient_checkpointing failed: {e}')
        # Avoid vit_gradient_checkpointing being overwritten by transformers.Trainer.gradient_checkpointing_enable.
        self.args.gradient_checkpointing = False

    def train(self, *args, **kwargs):
        if self.model_meta.is_multimodal:
            models = []
            for model_name in ['model', 'ref_model', 'value_model', 'teacher_model']:
                model = getattr(self, model_name, None)
                if isinstance(model, nn.Module):
                    models.append(model)

            reward_model = getattr(self, 'reward_model', None)
            if reward_model is not None:
                if isinstance(reward_model, list):
                    models.extend([m for m in reward_model if isinstance(m, nn.Module)])
                elif isinstance(reward_model, nn.Module):
                    models.append(reward_model)

            models = list(set(self.accelerator.unwrap_model(model) for model in models))  # Deduplicate
            logger.info(
                "Registering post_encode hook(s) for modules: %s",
                [model.__class__.__name__ for model in models],
            )
            self.template.register_post_encode_hook(models)
            logger.info(f'Successfully registered post_encode hook: {[model.__class__.__name__ for model in models]}.')
        self._save_initial_model(self.args.output_dir)

        # gradient_checkpointing
        gradient_checkpointing = self.args.gradient_checkpointing
        self._prepare_gradient_checkpointing(self.accelerator.unwrap_model(self.model))
        with self.hub.patch_hub(), self._fix_grad_norm_nan(), self._patch_skip_first_batches(
        ), self._patch_deepspeed_load_checkpoint():
            res = super().train(*args, **kwargs)
        if self.model_meta.is_multimodal:
            logger.info('Removing post_encode hook(s) after training completes.')
            self.template.remove_post_encode_hook()
        self.args.gradient_checkpointing = gradient_checkpointing  # recover
        return res

    def push_to_hub(self, *args, **kwargs):
        with self.hub.patch_hub():
            return super().push_to_hub(*args, **kwargs)

    @staticmethod
    def compute_custom_metrics(metrics, key_prefix: str = ''):
        logs = {}
        # Synchronize keys to avoid getting stuck.
        if dist.is_initialized():
            all_keys = [None] * dist.get_world_size()
            dist.all_gather_object(all_keys, list(metrics.keys()))
            for key in set().union(*all_keys):
                if key not in metrics:
                    metrics[key]

        for k, metric in sorted(metrics.items()):
            k = f'{key_prefix}{k}'
            value = metric.compute()
            metric.reset()
            if isinstance(value, dict):
                if len(value) == 1:
                    val = list(value.values())[0]
                    logs[k] = val
                else:
                    for k_suffix, val in value.items():
                        new_k = f'{k}_{k_suffix}'
                        logs[new_k] = val
            else:
                logs[k] = value
        for k in list(logs.keys()):
            if logs[k] is None:
                logs.pop(k)
        return logs

    def log(self, logs: Dict[str, float], *args, **kwargs) -> None:
        mode = 'train' if self.model.training else 'eval'
        metrics = self.custom_metrics[mode]
        prefix = 'eval_' if mode == 'eval' else ''
        logs.update(self.compute_custom_metrics(metrics, prefix))
        return super().log(logs, *args, **kwargs)

    def _maybe_log_save_evaluate(self, tr_loss, *args, **kwargs):
        import torch.distributed as dist
        is_dist = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if is_dist else 0
        
        # VERIFICATION: Print to confirm all ranks reach this function
        print(f"[RANK {rank}] _maybe_log_save_evaluate() ENTRY - step={getattr(self.state, 'global_step', 'unknown')}", flush=True)
        
        if self.control.should_log and self.state.global_step > self._globalstep_last_logged:
            self.control.should_log = False

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()
            loss = tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged)
            logs: Dict[str, float] = {'loss': loss}  # loss first
            if version.parse(transformers.__version__) >= version.parse('4.38'):
                grad_norm = args[0]
                if grad_norm is not None:
                    logs['grad_norm'] = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            logs['learning_rate'] = self._get_learning_rate()
            tr_loss -= tr_loss
            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()
            self.log(logs)

        if self.args.eval_use_evalscope and self.control.should_evaluate:
            try:
                self._evalscope_eval()
            except Exception as e:
                logger.warning(f'Failed to call EvalScope evaluation function: {e}.')

            if not self.eval_dataset:
                self.control.should_evaluate = False
        
        # CRITICAL: We need to synchronize should_save across all ranks BEFORE calling super()
        # because super() will handle log → evaluate → save in the correct order.
        # However, we need to ensure ALL ranks participate in the save part to avoid deadlocks.
        should_save_local = getattr(self.control, 'should_save', False)
        print(f"[RANK {rank}] Checking should_save before super(): should_save={should_save_local}", flush=True)
        logger.info(f"Rank {rank}: Checking should_save before super(): should_save={should_save_local}")
        
        # In distributed mode, synchronize should_save across all ranks
        # Rank 0 broadcasts its should_save value to all ranks
        if is_dist:
            if torch.cuda.is_available():
                device = torch.device(f'cuda:{rank}')
            else:
                device = torch.device('cpu')
                logger.warning(f"Rank {rank}: CUDA not available, using CPU for broadcast (may fail with NCCL backend)")
            
            should_save_tensor = torch.tensor(1 if should_save_local else 0, dtype=torch.int, device=device)
            dist.broadcast(should_save_tensor, src=0)
            should_save_synced = bool(should_save_tensor.item())
            print(f"[RANK {rank}] After sync: should_save={should_save_synced} (local={should_save_local})", flush=True)
            logger.info(f"Rank {rank}: After sync: should_save={should_save_synced} (local={should_save_local})")
        else:
            should_save_synced = should_save_local
        
        # Store the synced should_save value so _save_checkpoint() can use it
        # This ensures all ranks know if they should participate in saving
        if should_save_synced:
            object.__setattr__(self.args, '_swift_should_save_synced', True)
        
        # CRITICAL: Call super() FIRST to handle log → evaluate → save in the correct order.
        # The super() method will call _save_checkpoint() which we've overridden to ensure
        # all ranks participate in saving.
        print(f"[RANK {rank}] Calling super()._maybe_log_save_evaluate() (will handle log → evaluate → save)...", flush=True)
        logger.info(f"Rank {rank}: Calling super()._maybe_log_save_evaluate() (will handle log → evaluate → save)...")
        
        try:
            super()._maybe_log_save_evaluate(tr_loss, *args, **kwargs)
            print(f"[RANK {rank}] ✅ Completed super()._maybe_log_save_evaluate()", flush=True)
            logger.info(f"Rank {rank}: ✅ Completed super()._maybe_log_save_evaluate()")
        except Exception as e:
            print(f"[RANK {rank}] ❌ Error in super()._maybe_log_save_evaluate(): {e}", flush=True)
            logger.error(f"Rank {rank}: ❌ Error in super()._maybe_log_save_evaluate(): {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
        
        # CRITICAL: After super()._maybe_log_save_evaluate() completes, we need to ensure all ranks
        # wait for rank 0 to finish _save_checkpoint(). HuggingFace Trainer only calls _save_checkpoint()
        # on rank 0, so non-zero ranks need to wait at a barrier here.
        # 
        # IMPORTANT: Rank 0 might be stuck in _save_checkpoint() if saving hangs. This barrier ensures
        # all ranks wait for rank 0 to complete the save before proceeding to the next training step.
        if is_dist:
            print(f"[RANK {rank}] Waiting for all ranks to complete _maybe_log_save_evaluate() (including _save_checkpoint on rank 0)...", flush=True)
            logger.info(f"Rank {rank}: Waiting for all ranks to complete _maybe_log_save_evaluate()...")
            dist.barrier()
            print(f"[RANK {rank}] ✅ All ranks completed _maybe_log_save_evaluate(), proceeding to next batch", flush=True)
            logger.info(f"Rank {rank}: ✅ All ranks completed _maybe_log_save_evaluate(), proceeding to next batch")
        
        # Clean up flags after checkpoint saving is complete
        if hasattr(self.args, '_swift_pre_saved'):
            object.__delattr__(self.args, '_swift_pre_saved')
        if hasattr(self.args, '_swift_should_save_synced'):
            object.__delattr__(self.args, '_swift_should_save_synced')

    def create_loss_and_metric(self, args):
        res = {}
        if args.metric is not None:
            res['compute_metrics'], res['preprocess_logits_for_metrics'] = get_metric(args.metric)
        if args.loss_type is not None:
            res['compute_loss_func'] = get_loss_func(args.loss_type)
        return res

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        import torch.distributed as dist
        is_dist = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if is_dist else 0
        
        print(f"[RANK {rank}] create_optimizer_and_scheduler() ENTRY - optimizer={getattr(self.args, 'optimizer', None)}", flush=True)
        logger.info(f"Rank {rank}: create_optimizer_and_scheduler() ENTRY - optimizer={getattr(self.args, 'optimizer', None)}")
        
        if self.args.optimizer is not None:
            from swift.plugin import optimizers_map
            print(f"[RANK {rank}] Creating custom optimizer: {self.args.optimizer}", flush=True)
            logger.info(f"Rank {rank}: Creating custom optimizer: {self.args.optimizer}")
            optimizer_callback = optimizers_map[self.args.optimizer]
            print(f"[RANK {rank}] Calling optimizer_callback...", flush=True)
            logger.info(f"Rank {rank}: Calling optimizer_callback...")
            self.optimizer, self.lr_scheduler = optimizer_callback(self.args, self.model, self.train_dataset)
            print(f"[RANK {rank}] ✅ Optimizer callback completed - optimizer type: {type(self.optimizer).__name__}", flush=True)
            logger.info(f"Rank {rank}: ✅ Optimizer callback completed - optimizer type: {type(self.optimizer).__name__}")
            
            # Check if OGD trainer was created
            if hasattr(self.args, '_ogd_trainer') and self.args._ogd_trainer is not None:
                print(f"[RANK {rank}] ✅ OGD trainer found after optimizer creation!", flush=True)
                logger.info(f"Rank {rank}: ✅ OGD trainer found after optimizer creation!")
            else:
                print(f"[RANK {rank}] ⚠️ OGD trainer NOT found after optimizer creation (use_ogd={getattr(self.args, 'use_ogd', False)})", flush=True)
                logger.warning(f"Rank {rank}: ⚠️ OGD trainer NOT found after optimizer creation (use_ogd={getattr(self.args, 'use_ogd', False)})")
            
            if self.optimizer is None:
                print(f"[RANK {rank}] Optimizer is None, calling create_optimizer()...", flush=True)
                logger.info(f"Rank {rank}: Optimizer is None, calling create_optimizer()...")
                self.create_optimizer()
            if self.lr_scheduler is None:
                print(f"[RANK {rank}] LR scheduler is None, calling create_scheduler()...", flush=True)
                logger.info(f"Rank {rank}: LR scheduler is None, calling create_scheduler()...")
                self.create_scheduler(num_training_steps=num_training_steps, optimizer=self.optimizer)
        else:
            print(f"[RANK {rank}] Using default optimizer (calling super().create_optimizer_and_scheduler())", flush=True)
            logger.info(f"Rank {rank}: Using default optimizer (calling super().create_optimizer_and_scheduler())")
            super().create_optimizer_and_scheduler(num_training_steps=num_training_steps)
        
        print(f"[RANK {rank}] Optimizer and scheduler creation completed", flush=True)
        logger.info(f"Rank {rank}: Optimizer and scheduler creation completed")
        
        # After optimizer is created, trigger OGD memory update if OGD trainer exists
        # This is needed because OGD trainer is created in create_ogd_multiview_graph_optimizer()
        # which is called above, but on_train_begin() is called before this method
        if hasattr(self.args, '_ogd_trainer') and self.args._ogd_trainer is not None:
            ogd_trainer = self.args._ogd_trainer
            if not (hasattr(ogd_trainer.memory, '_memory_updated') and ogd_trainer.memory._memory_updated):
                # Trigger OGD memory update
                try:
                    eval_dataset = getattr(self, 'eval_dataset', None)
                    train_dataset = getattr(self, 'train_dataset', None)
                    dataset_for_memory = eval_dataset if eval_dataset is not None else train_dataset
                    if dataset_for_memory is not None:
                        print("[OGD] Triggering memory update after optimizer creation...", flush=True)
                        logger.info("[OGD] Triggering memory update after optimizer creation...")
                        from torch.utils.data import DataLoader
                        
                        # Use the trainer's data collator to properly handle variable-length sequences
                        # (e.g., variable-length graphs, images, etc.)
                        data_collator = getattr(self, 'data_collator', None)
                        if data_collator is None:
                            logger.warning("[OGD] No data_collator found, using default collate (may fail with variable-length sequences)")
                        
                        batch_size = min(32, getattr(self.args, 'per_device_eval_batch_size', 32))
                        memory_dataloader = DataLoader(
                            dataset_for_memory,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=0,
                            pin_memory=False,
                            collate_fn=data_collator  # Use trainer's collator to handle variable-length sequences
                        )
                        num_batches = getattr(self.args, 'ogd_update_memory_batches', 100)
                        loss_fn = getattr(self, 'compute_loss', None)
                        ogd_trainer.update_memory(memory_dataloader, num_batches=num_batches, loss_fn=loss_fn)
                        ogd_trainer.memory._memory_updated = True
                        ogd_trainer.enable_projection()
                        print("[OGD] ✅ Memory update complete after optimizer creation", flush=True)
                        logger.info("[OGD] ✅ Memory update complete after optimizer creation")
                except Exception as e:
                    print(f"[OGD] ⚠️ Error updating memory after optimizer creation: {e}", flush=True)
                    logger.warning(f"[OGD] Error updating memory after optimizer creation: {e}")
        
        # Restore scheduler step for continual learning (after scheduler is created)
        if hasattr(self, '_saved_scheduler_step') and self.lr_scheduler is not None:
            saved_step = self._saved_scheduler_step
            # Use module-level logger (already imported at top of file)
            try:
                # Restore the step count to continue the learning rate schedule
                if hasattr(self.lr_scheduler, 'last_epoch'):
                    self.lr_scheduler.last_epoch = saved_step
                    logger.info(
                        f"✅ Restored scheduler to step {saved_step} for continual learning. "
                        f"Learning rate continues from step {saved_step} (prevents LR jump that could harm pre-trained model)."
                    )
                # Also step the scheduler to the correct position if needed
                elif hasattr(self.lr_scheduler, 'step'):
                    current_step = getattr(self.lr_scheduler, '_step_count', 0)
                    if saved_step > current_step:
                        for _ in range(current_step, saved_step):
                            self.lr_scheduler.step()
                        logger.info(f"✅ Advanced scheduler to step {saved_step}")
            except Exception as e:
                logger.warning(f"Could not restore scheduler step: {e}")
            finally:
                # Clean up
                delattr(self, '_saved_scheduler_step')

    def _compute_acc(self, outputs, labels, cu_seqlens=None) -> None:
        args = self.args
        logits = outputs.logits
        metrics = None
        if getattr(args, 'loss_type', None) in {'generative_reranker', 'listwise_generative_reranker'} \
                and logits is not None and logits.dim() == 3:
            tokenizer = getattr(self, 'processing_class', None)
            if tokenizer is None and getattr(self, 'template', None) is not None:
                tokenizer = self.template.tokenizer
            if tokenizer is None:
                raise RuntimeError('tokenizer not available for generative_reranker acc')

            positive_token = os.environ.get('GENERATIVE_RERANKER_POSITIVE_TOKEN', 'yes')
            negative_token = os.environ.get('GENERATIVE_RERANKER_NEGATIVE_TOKEN', 'no')

            try:
                positive_token_id = tokenizer.convert_tokens_to_ids(positive_token)
                negative_token_id = tokenizer.convert_tokens_to_ids(negative_token)
            except Exception as e:
                logger.warning(f'Failed to convert reranker tokens to ids: {e}')
                positive_token_id = None
                negative_token_id = None

            if isinstance(positive_token_id, int) and isinstance(negative_token_id, int) \
                    and positive_token_id >= 0 and negative_token_id >= 0:
                positive_logits = logits[:, -1, positive_token_id]
                negative_logits = logits[:, -1, negative_token_id]
                binary_preds = (positive_logits > negative_logits).long()
                metrics = compute_acc(
                    binary_preds,
                    labels.long(),
                    acc_strategy=args.acc_strategy,
                    is_encoder_decoder=self.template.is_encoder_decoder,
                    cu_seqlens=cu_seqlens)
        elif logits.dim() == 1 or (logits.dim() == 2 and logits.size(-1) == 1):
            if logits.dim() == 2:
                logits = logits.squeeze(-1)
            binary_preds = (logits > 0).long()
            metrics = compute_acc(
                binary_preds,
                labels.long(),
                acc_strategy=args.acc_strategy,
                is_encoder_decoder=self.template.is_encoder_decoder,
                cu_seqlens=cu_seqlens)
        else:
            preds = logits.argmax(dim=-1)
            if self.template.sequence_parallel_size > 1:
                from swift.trainers.sequence_parallel import sequence_parallel
                # Gather preds and labels across the sp group
                if isinstance(preds, np.ndarray):
                    preds = torch.from_numpy(preds).to(get_current_device())
                if isinstance(labels, np.ndarray):
                    labels = torch.from_numpy(labels).to(get_current_device())
                assert labels.shape[1] == preds.shape[1]

                if sequence_parallel.rp_world_size > 1:
                    position_ids = sequence_parallel.real_position_ids
                    position_ids = sequence_parallel.pad(position_ids, padding_value=-1, position_ids=position_ids)
                else:
                    position_ids = None
                preds_output = sequence_parallel.gather(preds, dim=1, position_ids=position_ids)
                labels_output = sequence_parallel.gather(labels, dim=1, position_ids=position_ids)
                # roll back to fit compute_acc
                labels_output = torch.roll(labels_output, shifts=1, dims=1)
                preds = preds_output
                labels = labels_output.int()

            metrics = compute_acc(
                preds,
                labels,
                acc_strategy=args.acc_strategy,
                is_encoder_decoder=self.template.is_encoder_decoder,
                cu_seqlens=cu_seqlens)

        if metrics:
            mode = 'train' if self.model.training else 'eval'
            for k, v in metrics.items():
                self.custom_metrics[mode][k].update(v)

    @torch.no_grad()
    def _evalscope_eval(self):
        from ..llm.eval.utils import EvalModel
        from evalscope import TaskConfig, run_task

        self.model.eval()
        # prepare task config
        task_config_kwargs = dict(
            model=EvalModel(
                model_name=f'model-step{self.state.global_step}',
                model=self.model,
                template=self.template,
                max_batch_size=self.args.per_device_eval_batch_size,
            ),
            eval_type='swift_custom',
            datasets=self.args.eval_dataset,
            dataset_args=self.args.eval_dataset_args,
            limit=self.args.eval_limit,
            work_dir=os.path.join(self.args.output_dir, 'eval'),
            eval_batch_size=self.args.per_device_eval_batch_size,
            generation_config=self.args.eval_generation_config or {'max_tokens': 512},
        )
        task_config_kwargs.update(self.args.extra_eval_args or {})
        task_config = TaskConfig(**task_config_kwargs)
        # start evaluation
        eval_report = run_task(task_config)
        # convert to dict
        eval_dict = {f'test_{k}': v.score for k, v in eval_report.items()}
        self.log(eval_dict)

        self.model.train()
        return eval_dict

    def prepare_logits_to_keep(self, inputs):
        labels = inputs['labels']
        loss_scale = inputs.get('loss_scale')
        if self.template.sequence_parallel_size > 1:
            raise NotImplementedError()
        if labels.shape[0] == 1 and not is_mp():
            # device_map may encounter device mismatch issues.
            loss_mask = (labels != -100)[0]
            labels = labels[:, loss_mask]
            labels = nn.functional.pad(labels, (1, 0), value=-100)
            if loss_scale is not None:
                loss_scale = loss_scale[:, loss_mask]
                inputs['loss_scale'] = nn.functional.pad(loss_scale, (1, 0), value=0)
            logits_to_keep = nn.functional.pad(loss_mask[1:], (0, 1), value=True)
        else:
            logits_to_keep = labels.shape[-1] - ((labels != -100).int().argmax(-1).min().item()) + 1
            assert logits_to_keep > 0
            labels = labels[:, -logits_to_keep:]
            if loss_scale is not None:
                inputs['loss_scale'] = loss_scale[:, -logits_to_keep:]
        inputs['labels'] = labels
        inputs['logits_to_keep'] = logits_to_keep

    def get_cu_seqlens(self, position_ids, logits_to_keep) -> torch.Tensor:
        from swift.llm import get_packed_seq_params
        cu_seqlens = get_packed_seq_params(position_ids)['cu_seq_lens_q']
        res_cu_seqlens = cu_seqlens.clone()
        if isinstance(logits_to_keep, torch.Tensor):
            for i in range(cu_seqlens.shape[0] - 1):
                start, end = cu_seqlens[i], cu_seqlens[i + 1]
                res_cu_seqlens[i + 1:] -= (~logits_to_keep[start:end]).sum()
        elif isinstance(logits_to_keep, int):
            res_cu_seqlens[1:] -= position_ids.shape[-1] + 1 - logits_to_keep
        return res_cu_seqlens

    @contextmanager
    def _patch_skip_first_batches(self):
        from transformers import trainer
        origin_skip_first_batches = trainer.skip_first_batches

        def skip_first_batches(dataloader, num_batches=0):
            if isinstance(dataloader, (DataLoaderShard, DataLoaderDispatcher)):
                # DataLoaderMixin
                return self.get_train_dataloader(skip_batches=num_batches)
            else:
                return origin_skip_first_batches(dataloader, num_batches)

        trainer.skip_first_batches = skip_first_batches
        try:
            yield
        finally:
            trainer.skip_first_batches = origin_skip_first_batches


class DataLoaderMixin:

    def get_sp_dataloader(self, dataset, batch_size, skip_batches=0):
        from swift.trainers.sequence_parallel import sequence_parallel
        from swift.trainers.sequence_parallel.utils import SequenceParallelSampler
        from swift.trainers.sequence_parallel.utils import SequenceParallelDispatcher
        data_collator = self.data_collator
        if isinstance(dataset, datasets.Dataset):
            dataset = self._remove_unused_columns(dataset, description='training')
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description='training')
        if hasattr(dataset, '__len__'):
            sampler = SequenceParallelSampler(sequence_parallel, dataset, seed=42)
            dataloader_params = {
                'batch_size': batch_size,
                'collate_fn': data_collator,
                'num_workers': self.args.dataloader_num_workers,
                'pin_memory': self.args.dataloader_pin_memory,
                'persistent_workers': self.args.dataloader_persistent_workers,
            }

            if not isinstance(dataset, torch.utils.data.IterableDataset):
                if skip_batches > 0:
                    from accelerate.data_loader import SkipBatchSampler
                    sampler = SkipBatchSampler(sampler, skip_batches=skip_batches * batch_size)
                dataloader_params['sampler'] = sampler
                dataloader_params['drop_last'] = self.args.dataloader_drop_last
                dataloader_params['worker_init_fn'] = partial(
                    seed_worker, num_workers=self.args.dataloader_num_workers, rank=self.args.process_index)

            return DataLoaderShard(dataset, device=self.accelerator.device, **dataloader_params)
        else:
            dataloader_params = {
                'collate_fn': data_collator,
                'num_workers': self.args.dataloader_num_workers,
                'pin_memory': self.args.dataloader_pin_memory,
                'persistent_workers': self.args.dataloader_persistent_workers,
                'prefetch_factor': self.args.dataloader_prefetch_factor
            }
            if dist.is_initialized() and dataloader_params['prefetch_factor']:
                dataloader_params['prefetch_factor'] = dataloader_params['prefetch_factor'] * dist.get_world_size()
            dataloader = DataLoader(dataset, batch_size=batch_size, **dataloader_params)
            dataloader = SequenceParallelDispatcher(
                dataloader, sequence_parallel, self.accelerator.device, skip_batches=skip_batches)
            return dataloader

    def get_train_dataloader(self, skip_batches=0):
        dataloader = None
        if self.template.sequence_parallel_size > 1:
            dataloader = self.get_sp_dataloader(self.train_dataset, self._train_batch_size, skip_batches=skip_batches)
        if dataloader is None:
            # Higher efficiency
            if self.train_dataset is None:
                raise ValueError('Trainer: training requires a train_dataset.')
            args = self.args
            train_dataset = self.train_dataset

            dataloader_params = {
                'collate_fn': self.data_collator,
                'num_workers': args.dataloader_num_workers,
                'pin_memory': args.dataloader_pin_memory,
                'persistent_workers': args.dataloader_persistent_workers,
                'prefetch_factor': args.dataloader_prefetch_factor
            }
            batch_sampler_params = {
                'drop_last':
                args.dataloader_drop_last,
                'shuffle':
                args.train_dataloader_shuffle,
                'data_seed':
                args.data_seed,
                'tp_size':
                args.deepspeed['tensor_parallel']['autotp_size']
                if args.deepspeed and 'tensor_parallel' in args.deepspeed else 1,
            }

            if hasattr(train_dataset, '__len__'):
                batch_sampler = BatchSamplerShard(
                    len(train_dataset), batch_size=self._train_batch_size, **batch_sampler_params)
                dataloader_params['worker_init_fn'] = partial(
                    seed_worker, num_workers=self.args.dataloader_num_workers, rank=self.args.process_index)
                if skip_batches > 0:
                    from accelerate.data_loader import SkipBatchSampler
                    batch_sampler = SkipBatchSampler(batch_sampler, skip_batches=skip_batches)
                dataloader_params['batch_sampler'] = batch_sampler
                dataloader = DataLoaderShard(train_dataset, device=self.accelerator.device, **dataloader_params)
            else:
                # IterableDataset
                if dist.is_initialized() and dataloader_params['prefetch_factor']:
                    dataloader_params['prefetch_factor'] = dataloader_params['prefetch_factor'] * dist.get_world_size()
                dataloader = DataLoader(train_dataset, batch_size=self._train_batch_size, **dataloader_params)
                dataloader = DataLoaderDispatcher(dataloader, self.accelerator.device, skip_batches=skip_batches)
        return dataloader

    def get_eval_dataloader(self, eval_dataset=None):
        dataloader = None
        if self.template.sequence_parallel_size > 1:
            if eval_dataset is None and self.eval_dataset is None:
                raise ValueError('Trainer: evaluation requires an eval_dataset.')
            eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
            dataloader = self.get_sp_dataloader(eval_dataset, self.args.eval_batch_size)
        if dataloader is None:
            return super().get_eval_dataloader(eval_dataset=eval_dataset)
        return dataloader
