# Copyright (c) Alibaba, Inc. and its affiliates.
import inspect
import os
from typing import List, Union

import torch
import torch.nn as nn
import transformers
from packaging import version
from transformers import TrainingArguments

from swift.llm import TrainArguments, deep_getattr
from swift.plugin import Tuner, extra_tuners
from swift.tuners import Swift
from swift.tuners.peft import PeftModel
from swift.utils import activate_parameters, find_all_linears, find_embedding, find_norm, freeze_parameters, get_logger

logger = get_logger()


def apply_liger(model_type: str):
    try:
        from liger_kernel.transformers import (apply_liger_kernel_to_llama, apply_liger_kernel_to_mistral,
                                               apply_liger_kernel_to_mixtral, apply_liger_kernel_to_gemma,
                                               apply_liger_kernel_to_qwen2, apply_liger_kernel_to_qwen3,
                                               apply_liger_kernel_to_qwen2_vl, apply_liger_kernel_to_qwen2_5_vl,
                                               apply_liger_kernel_to_phi3, apply_liger_kernel_to_mllama)
        from swift.llm import ModelType
        if model_type in (ModelType.llama, ModelType.llama3, ModelType.llama3_1, ModelType.llama3_2):
            apply_liger_kernel_to_llama()
        elif model_type in (ModelType.mistral):
            apply_liger_kernel_to_mistral()
        elif model_type in (ModelType.mixtral):
            apply_liger_kernel_to_mixtral()
        elif model_type in (ModelType.gemma, ModelType.gemma2):
            apply_liger_kernel_to_gemma()
        elif model_type in (ModelType.gemma3_text):
            from liger_kernel.transformers import apply_liger_kernel_to_gemma3_text
            apply_liger_kernel_to_gemma3_text()
        elif model_type in (ModelType.gemma3_vision, ModelType.gemma3n):
            from liger_kernel.transformers import apply_liger_kernel_to_gemma3
            apply_liger_kernel_to_gemma3()
        elif model_type in (ModelType.qwen2, ModelType.qwen2_5):
            apply_liger_kernel_to_qwen2()
        elif model_type in (ModelType.qwen3, ModelType.qwen3_guard, ModelType.qwen3_thinking,
                            ModelType.qwen3_nothinking, ModelType.qwen3_coder):
            apply_liger_kernel_to_qwen3()
        elif model_type in (ModelType.qwen3_moe, ModelType.qwen3_moe_thinking, ModelType.qwen3_coder):
            from liger_kernel.transformers import apply_liger_kernel_to_qwen3_moe
            apply_liger_kernel_to_qwen3_moe()
        elif model_type in (ModelType.qwen3_next, ModelType.qwen3_next_thinking):
            from liger_kernel.transformers import apply_liger_kernel_to_qwen3_next
            apply_liger_kernel_to_qwen3_next()
        elif model_type in (ModelType.phi3):
            apply_liger_kernel_to_phi3()
        elif model_type in (ModelType.llama3_2_vision):
            apply_liger_kernel_to_mllama()
        elif model_type in (ModelType.qwen2_vl):
            apply_liger_kernel_to_qwen2_vl()
        elif model_type in (ModelType.qwen2_5_vl, ModelType.qwen3_vl, ModelType.qwen3_moe_vl, ModelType.qvq):
            apply_liger_kernel_to_qwen2_5_vl()
        elif model_type in (ModelType.glm4, ModelType.glm4_0414, ModelType.glm4_z1_rumination):
            from liger_kernel.transformers import apply_liger_kernel_to_glm4
            apply_liger_kernel_to_glm4()
        elif model_type in (ModelType.glm4v, ModelType.glm4_1v):
            from liger_kernel.transformers import apply_liger_kernel_to_glm4v
            apply_liger_kernel_to_glm4v()
        elif model_type in (ModelType.glm4_5v):
            from liger_kernel.transformers import apply_liger_kernel_to_glm4v_moe
            apply_liger_kernel_to_glm4v_moe()
        elif model_type in (ModelType.internvl_hf, ModelType.internvl_gpt_hf):
            from liger_kernel.transformers import apply_liger_kernel_to_internvl
            apply_liger_kernel_to_internvl()
        elif model_type in (ModelType.llama4):
            from liger_kernel.transformers import apply_liger_kernel_to_llama4
            apply_liger_kernel_to_llama4()
        elif model_type in (ModelType.llava1_5_hf, ModelType.llava_llama3_hf, ModelType.pixtral):
            from liger_kernel.transformers import apply_liger_kernel_to_llava
            apply_liger_kernel_to_llava()
        elif model_type in (ModelType.paligemma):
            from liger_kernel.transformers import apply_liger_kernel_to_paligemma
            apply_liger_kernel_to_paligemma()
        else:
            raise ValueError(f'Unsupported liger model_type: {model_type}')
    except ImportError:
        raise ImportError('Please upgrade liger-kernel to apply liger kernel to this model '
                          'by running `pip install -U liger-kernel`')


def get_multimodal_target_regex(
        model,
        *,
        freeze_llm: bool = False,
        freeze_vit: bool = True,
        freeze_aligner: bool = True,
        include_embedding: bool = False,
) -> str:
    model_arch = model.model_meta.model_arch
    modules = []
    if not freeze_llm:
        modules += model_arch.language_model
    if not freeze_vit:
        modules += model_arch.vision_tower
    if not freeze_aligner:
        modules += model_arch.aligner
    assert len(modules) > 0, f'modules: {modules}'

    extra_layers = []
    if include_embedding:
        extra_layers.append(nn.Embedding)
    res = []
    for module in modules:
        rejected_modules = []
        if not freeze_vit:
            for aligner in model_arch.aligner:
                if aligner.startswith(f'{module}.'):
                    rejected_modules.append(aligner)

        sub_module = deep_getattr(model, module)
        target_modules = find_all_linears(sub_module, model_arch, extra_layers)
        if not target_modules:
            continue
        target_modules = [tm for tm in target_modules if tm]
        target_pattern = rf'.*\.({"|".join(target_modules)})' if target_modules else ''
        rejected_pattern = rf'(?!({"|".join(rejected_modules)}))' if rejected_modules else ''
        res.append(rf'{rejected_pattern}{module}{target_pattern}')

    return rf'^({"|".join(res)})$'


def get_target_modules(args, model) -> Union[str, List[str]]:
    """Replace all-linear to actual modules"""
    model_meta = model.model_meta
    if isinstance(args.target_modules, str):
        return args.target_modules
    target_modules = args.target_modules.copy()
    if 'all-linear' in target_modules:
        if model_meta.is_multimodal:
            return get_multimodal_target_regex(
                model,
                freeze_llm=args.freeze_llm,
                freeze_vit=args.freeze_vit,
                freeze_aligner=args.freeze_aligner,
                include_embedding='all-embedding' in target_modules)
        else:
            target_modules.remove('all-linear')
            target_modules += find_all_linears(model)
    if 'all-embedding' in target_modules:
        target_modules.remove('all-embedding')
        target_modules += find_embedding(model)
    return target_modules


def get_modules_to_save(args, model, task_type=None):
    modules_to_save = args.modules_to_save.copy()
    if 'all-embedding' in args.modules_to_save:
        modules_to_save.remove('all-embedding')
        modules_to_save += find_embedding(model)
    if 'all-norm' in args.modules_to_save:
        modules_to_save.remove('all-norm')
        modules_to_save += find_norm(model)
    if task_type and task_type.lower() == 'seq_cls':  # reward_model
        modules_to_save.append('v_head')
    # NOTE: graph_encoder is NOT added to modules_to_save because:
    # 1. It has 7B+ parameters, causing OOM when PEFT tries to deepcopy it
    # 2. It's already part of the model state_dict, so it will be saved with the checkpoint
    # 3. PEFT's modules_to_save mechanism is for small modules (like heads), not large encoders
    # The graph_encoder weights will be saved automatically in the model checkpoint
    return modules_to_save


def get_vera_target_modules(model, config):
    """This function is only useful on the vera tuner"""
    target_modules = config.target_modules
    modules_dict = {
        name: module.weight.shape
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear) and any([t in name for t in target_modules])
    }  # only Linear for now
    if len(set(modules_dict.values())) > 1:
        v = [t for t in target_modules if 'v' in t]
        if not v:
            raise ValueError('Please manually pass in `vera_target_modules`, do not use `all-linear`,'
                             'because Vera need all target linears to be the same size.')
        v = v[0]
        shape = [shape for name, shape in modules_dict.items() if v in name][0]
        names = [_name for _name, _shape in modules_dict.items() if _shape == shape]
        config.target_modules = [t for t in target_modules if any([t in name for name in names])]
    return config


def prepare_adapter(args: TrainArguments, model, *, template=None, train_dataset=None, task_type=None):
    from swift.tuners import (AdaLoraConfig, AdapterConfig, BOFTConfig, LLaMAProConfig, LongLoRAModelType, LoraConfig,
                              LoRAConfig, ReftConfig, Swift, VeraConfig)
    task_type = (task_type or args.task_type).upper()
    target_modules = get_target_modules(args, model)
    modules_to_save = get_modules_to_save(args, model, task_type)
    lora_kwargs = {
        'r': args.lora_rank,
        'target_modules': target_modules,
        'lora_alpha': args.lora_alpha,
        'lora_dropout': args.lora_dropout,
        'bias': args.lora_bias,
        'modules_to_save': modules_to_save,
        'use_rslora': args.use_rslora,
        'use_dora': args.use_dora,
        'lorap_lr_ratio': args.lorap_lr_ratio,
        'init_lora_weights': args.init_weights,
    }
    if args.train_type in ('lora', 'longlora'):
        if args.use_swift_lora:
            lora_config = LoRAConfig(lora_dtype=args.lora_dtype, **lora_kwargs)
            model = Swift.prepare_model(model, lora_config)
            logger.info(f'lora_config: {lora_config}')
        elif args.tuner_backend == 'peft':
            if task_type == 'EMBEDDING':
                task_type = None
            elif task_type == 'RERANKER':
                task_type = 'SEQ_CLS'
            elif task_type == 'GENERATIVE_RERANKER':
                task_type = 'CAUSAL_LM'
            if args.target_parameters is not None:
                lora_kwargs['target_parameters'] = args.target_parameters
            lora_config = LoraConfig(task_type=task_type, lora_dtype=args.lora_dtype, **lora_kwargs)
            if args.init_weights == 'lora-ga':
                try:
                    import lora_ga
                except ImportError as e:
                    error_message = """
                    Since 'LoRA-GA' is not implemented by PEFT, you will need to install it directly from GitHub.
                    Command: 'pip install git+https://github.com/lxline/LoRA-GA.git'.
                    """
                    logger.info(error_message)
                    raise RuntimeError(error_message) from e
                model = lora_ga.entrypoint.get_lora_ga_model(
                    model=model,
                    data_collator=template.data_collator,
                    dataset=train_dataset,
                    batch_size=args.lora_ga_batch_size,
                    num_iters=args.lora_ga_iters,
                    max_length=args.lora_ga_max_length,
                    direction=args.lora_ga_direction,
                    dtype=args.lora_dtype,
                    scale=args.lora_ga_scale,
                    stable_gamma=args.lora_ga_stable_gamma,
                )
            else:
                model = Swift.prepare_model(model, lora_config)
            logger.info(f'lora_config: {lora_config}')
        elif args.tuner_backend == 'unsloth':
            if args.resume_from_checkpoint is None:
                if args.model_meta.is_multimodal:
                    from unsloth import FastVisionModel as UnslothModel
                else:
                    from unsloth import FastLanguageModel as UnslothModel
                assert args.train_type == 'lora', 'Unsloth does not support LongLoRA'
                lora_kwargs.pop('lorap_lr_ratio')
                model = UnslothModel.get_peft_model(
                    model,
                    use_gradient_checkpointing='unsloth',
                    max_seq_length=args.max_length or 2048,  # 2048 is the default value of unsloth
                    **lora_kwargs,
                )
                logger.info(f'unsloth_config: {lora_kwargs}')
        if args.train_type == 'longlora':
            assert LongLoRAModelType.LLAMA in args.model_type
            assert version.parse(transformers.__version__) >= version.parse('4.39.3')
            from swift.tuners.longlora.llama import replace_llama_attn
            replace_llama_attn(model)
            model.config.group_size_ratio = 0.25
    elif args.train_type == 'adalora':
        lora_kwargs.pop('lorap_lr_ratio', None)
        lora_kwargs['rank_pattern'] = None
        from swift.plugin.optimizer import calculate_max_steps
        adalora_config = AdaLoraConfig(
            task_type=task_type,
            **lora_kwargs,
            target_r=args.adalora_target_r,
            init_r=args.adalora_init_r,
            tinit=args.adalora_tinit,
            tfinal=args.adalora_tfinal,
            deltaT=args.adalora_deltaT,
            beta1=args.adalora_beta1,
            beta2=args.adalora_beta2,
            orth_reg_weight=args.adalora_orth_reg_weight,
            total_step=calculate_max_steps(args.training_args, train_dataset),
        )
        model = Swift.prepare_model(model, adalora_config)
        logger.info(f'adalora_config: {adalora_config}')
    elif args.train_type == 'llamapro':
        llamapro_config = LLaMAProConfig(
            model_type=model.model_meta.model_arch.arch_name,
            num_new_blocks=args.llamapro_num_new_blocks,
            num_groups=args.llamapro_num_groups)
        model = Swift.prepare_model(model, llamapro_config)
        logger.info(f'llamapro_config: {llamapro_config}')
    elif args.train_type == 'adapter':
        model_arch = model.model_meta.model_arch
        mlp_key = model_arch.mlp
        mlp_key = mlp_key.split('.{}.')[1]
        adapter_config = AdapterConfig(
            dim=model.config.hidden_size,
            target_modules=[mlp_key],
            hidden_pos=0,
            adapter_length=args.adapter_length,
            act_layer=args.adapter_act)
        model = Swift.prepare_model(model, adapter_config)
        logger.info(f'adapter_config: {adapter_config}')
    elif args.train_type == 'vera':
        vera_config = VeraConfig(
            r=args.vera_rank,
            target_modules=target_modules,
            projection_prng_key=args.vera_projection_prng_key,
            vera_dropout=args.vera_dropout,
            d_initial=args.vera_d_initial,
            modules_to_save=args.modules_to_save,
        )
        vera_config = get_vera_target_modules(model, vera_config)
        model = Swift.prepare_model(model, vera_config)
        logger.info(f'vera_config: {vera_config}')
    elif args.train_type == 'boft':
        boft_config = BOFTConfig(
            boft_block_size=args.boft_block_size,
            boft_block_num=args.boft_block_num,
            boft_n_butterfly_factor=args.boft_n_butterfly_factor,
            target_modules=target_modules,
            boft_dropout=args.boft_dropout,
            modules_to_save=args.modules_to_save,
        )
        model = Swift.prepare_model(model, boft_config)
        logger.info(f'boft_config: {boft_config}')
    elif args.train_type == 'fourierft':
        from peft import FourierFTConfig
        fourier_config = FourierFTConfig(
            target_modules=target_modules,
            modules_to_save=args.modules_to_save,
            n_frequency=args.fourier_n_frequency,
            scaling=args.fourier_scaling,
        )
        model = Swift.prepare_model(model, fourier_config)
        logger.info(f'fourier_config: {fourier_config}')
    elif args.train_type == 'reft':
        reft_config = ReftConfig(
            model_type=model.model_meta.model_arch,
            layer_key=args.reft_layer_key,
            r=args.reft_rank,
            layers=args.reft_layers,
            intervention_type=args.reft_intervention_type,
            args=args.reft_args,
        )
        logger.info(f'reft config: {reft_config}')
        model = Swift.prepare_model(model, {'reft': reft_config})
    elif args.train_type == 'bone':
        # Version loosing
        from peft import BoneConfig
        bone_config = BoneConfig(
            target_modules=target_modules,
            r=args.reft_rank,
            init_weights=args.init_weights,
        )
        logger.info(f'bone config: {bone_config}')
        model = Swift.prepare_model(model, bone_config)
    else:
        raise ValueError(f'Unknown train_type: {args.train_type}')
    return model


class TunerMixin:

    @classmethod
    def prepare_model(cls, args, model, *, template=None, train_dataset=None, task_type=None):
        # transformers >= 4.45.0, apply liger in transformers https://github.com/huggingface/transformers/pull/32860
        # transformers < 4.45.0, apply liger in here
        if args.use_liger_kernel and 'use_liger_kernel' not in inspect.signature(TrainingArguments).parameters:
            # Apply liger
            apply_liger(args.model_type)

        # Initialize graph encoder for multi-view training BEFORE LoRA/adapters
        # This ensures graph encoder is part of base model for proper LoRA integration
        if getattr(args, 'use_graph_encoder', False) and not hasattr(model, 'graph_encoder'):
            # Set environment variable for graph_max_nodes so template can access it
            import os
            graph_max_nodes = getattr(args, 'graph_max_nodes', None)
            if graph_max_nodes is not None:
                os.environ['GRAPH_MAX_NODES'] = str(graph_max_nodes)
            elif 'GRAPH_MAX_NODES' in os.environ:
                # Clear if not set in args
                del os.environ['GRAPH_MAX_NODES']
            model = init_graph_encoder(model, args, template)

            # Verify graph tokens are present in tokenizer (should be added via ModelArguments._init_new_special_tokens)
            # This is a safety check to ensure tokens are present even if they weren't added automatically
            if template is not None and hasattr(template, 'tokenizer') and template.tokenizer is not None:
                vocab = template.tokenizer.get_vocab()
                graph_tokens = ['<|graph_pad|>', '<|graph_start|>', '<|graph_end|>']
                missing_tokens = [t for t in graph_tokens if t not in vocab]

                if missing_tokens:
                    logger.warning(f"⚠️ Missing graph tokens in tokenizer: {missing_tokens}")
                    logger.info(f"  Adding missing graph tokens to tokenizer...")

                    # Add missing tokens
                    num_added = template.tokenizer.add_tokens(missing_tokens)

                    if num_added > 0:
                        # Resize model embeddings to accommodate new tokens
                        if hasattr(model, 'resize_token_embeddings'):
                            model.resize_token_embeddings(len(template.tokenizer))
                            logger.info(f"  ✅ Added {num_added} graph tokens and resized model embeddings")
                        else:
                            logger.warning(
                                f"  ⚠️ Could not resize embeddings (model doesn't have resize_token_embeddings)")
                    else:
                        logger.warning(f"  ⚠️ Failed to add graph tokens to tokenizer")
                else:
                    logger.info(f"  ✅ Tokenizer has all required graph tokens")

        if args.is_adapter:
            if args.tuner_backend != 'unsloth' and args.train_type not in extra_tuners:
                # Fix the name of the layer in xcomposer that contains Plora.
                # Unsloth prepares and loads lora outside this function when
                # resume_from_checkpoint, so do not disable grad here
                model.requires_grad_(False)
            if args.resume_from_checkpoint or args.adapters:
                if args.train_type in extra_tuners:
                    tuner: Tuner = extra_tuners[args.train_type]
                else:
                    tuner = Swift
                assert not args.adapters or len(args.adapters) == 1, f'args.adapters: {args.adapters}'
                model = tuner.from_pretrained(model, args.resume_from_checkpoint or args.adapters[0], is_trainable=True)

                # Load graph encoder weights from checkpoint if resuming training
                # This must happen AFTER adapters are loaded so we can access the base model correctly
                if getattr(args, 'use_graph_encoder', False) and args.resume_from_checkpoint and os.path.exists(
                        args.resume_from_checkpoint):
                    checkpoint_path = args.resume_from_checkpoint
                    graph_encoder_path = os.path.join(checkpoint_path, 'graph_encoder.bin')
                    graph_config_path = os.path.join(checkpoint_path, 'graph_encoder_config.json')

                    if os.path.exists(graph_encoder_path):
                        logger.info(f"📦 Loading graph_encoder weights from checkpoint: {graph_encoder_path}")
                        try:
                            import json

                            # Get the actual model (unwrap PEFT if needed)
                            # Works for both Qwen2VL and Phi-3.5-vision (and other models)
                            base_model = model
                            if isinstance(model, PeftModel):
                                logger.debug(f"🔍 Unwrapping PeftModel to access base model for graph_encoder loading")
                                if hasattr(model.base_model, 'model'):
                                    base_model = model.base_model.model
                                    logger.debug(f"📍 Unwrapped to: {type(base_model).__name__} (via base_model.model)")
                                else:
                                    base_model = model.base_model
                                    logger.debug(f"📍 Unwrapped to: {type(base_model).__name__} (via base_model)")

                            if hasattr(base_model, 'graph_encoder'):
                                logger.info(f"  📍 Found graph_encoder in {type(base_model).__name__}")
                                # Load config if available
                                if os.path.exists(graph_config_path):
                                    with open(graph_config_path, 'r') as f:
                                        saved_config = json.load(f)
                                    logger.info(f"  ✅ Loaded graph_encoder config from checkpoint")

                                # Load state dict
                                graph_encoder_state_full = torch.load(graph_encoder_path, map_location='cpu')

                                # Filter: Keep only base trainable params, skip LoRA adapters
                                # LoRA params are saved in adapter_model.bin (PEFT's save)
                                graph_encoder_state = {}
                                lora_skipped = 0
                                qwen_skipped = 0

                                for key, value in graph_encoder_state_full.items():
                                    # Skip LoRA parameters (they're loaded separately as PEFT adapters)
                                    if 'lora_A' in key or 'lora_B' in key or 'lora_embedding' in key:
                                        lora_skipped += 1
                                        continue
                                    # Skip qwen_model parameters (they're shared with main model)
                                    if 'qwen_model' in key:
                                        qwen_skipped += 1
                                        continue
                                    # Keep base trainable params
                                    graph_encoder_state[key] = value

                                logger.info(
                                    f"     Filtered {len(graph_encoder_state_full)} → {len(graph_encoder_state)} parameters")
                                logger.info(
                                    f"     Skipped {lora_skipped} LoRA params, {qwen_skipped} qwen_model params")

                                # Load with strict=False (qwen_model parameters not in saved file)
                                missing_keys, unexpected_keys = base_model.graph_encoder.load_state_dict(
                                    graph_encoder_state, strict=False
                                )

                                # Log what was loaded
                                if missing_keys:
                                    # Missing 'qwen_model' params are expected (shared with main model)
                                    qwen_missing = [k for k in missing_keys if 'qwen_model' in k]
                                    other_missing = [k for k in missing_keys if 'qwen_model' not in k]
                                    if other_missing:
                                        logger.warning(f"  ⚠️ Missing non-shared parameters: {other_missing[:5]}")
                                    if qwen_missing:
                                        logger.debug(
                                            f"  ℹ️ Missing shared language model params (expected): {len(qwen_missing)}")
                                    logger.info(f"     ({len(qwen_missing)} qwen_model params use shared weights)")

                                if unexpected_keys:
                                    logger.warning(f"  ⚠️ Unexpected keys: {unexpected_keys[:5]}")

                                # Count parameters
                                total_params = sum(p.numel() for p in base_model.graph_encoder.parameters())
                                trainable_params = sum(
                                    p.numel() for p in base_model.graph_encoder.parameters() if p.requires_grad)
                                logger.info(
                                    f"  ✅ Loaded graph_encoder: {total_params:,} params ({trainable_params:,} trainable)")
                            else:
                                logger.warning(f"  ⚠️ Model does not have graph_encoder attribute after initialization")
                        except Exception as e:
                            logger.error(f"  ❌ Failed to load graph_encoder from checkpoint: {e}")
                            import traceback
                            traceback.print_exc()
                    else:
                        logger.info(f"  ℹ️ graph_encoder.bin not found in checkpoint (will use fresh initialization)")

                # Verify and add graph tokens to tokenizer if missing (important for continual learning)
                # This handles the case where we resume from an image-text checkpoint that doesn't have graph tokens
                if template is not None and hasattr(template, 'tokenizer') and template.tokenizer is not None:
                    vocab = template.tokenizer.get_vocab()
                    graph_tokens = ['<|graph_pad|>', '<|graph_start|>', '<|graph_end|>']
                    missing_tokens = [t for t in graph_tokens if t not in vocab]

                    if missing_tokens:
                        logger.warning(f"⚠️ Missing graph tokens in tokenizer: {missing_tokens}")
                        logger.info(f"  Adding missing graph tokens to tokenizer...")

                        # Add missing tokens
                        num_added = template.tokenizer.add_tokens(missing_tokens)

                        if num_added > 0:
                            # Resize model embeddings to accommodate new tokens
                            # Get the actual model (unwrap PEFT if needed)
                            base_model = model
                            if isinstance(model, PeftModel):
                                base_model = model.base_model.model if hasattr(model.base_model,
                                                                               'model') else model.base_model

                            # Resize embeddings
                            if hasattr(base_model, 'resize_token_embeddings'):
                                base_model.resize_token_embeddings(len(template.tokenizer))
                                logger.info(f"  ✅ Added {num_added} graph tokens and resized model embeddings")
                            else:
                                logger.warning(
                                    f"  ⚠️ Could not resize embeddings (model doesn't have resize_token_embeddings)")
                        else:
                            logger.warning(f"  ⚠️ Failed to add graph tokens to tokenizer")
                    else:
                        logger.info(f"  ✅ Tokenizer has all required graph tokens")
            else:
                if args.train_type in extra_tuners:
                    tuner: Tuner = extra_tuners[args.train_type]
                    model = tuner.prepare_model(args, model)
                else:
                    model = prepare_adapter(
                        args, model, template=template, train_dataset=train_dataset, task_type=task_type)
            # fix bug: Attempting to unscale FP16 gradients.
            #   peft: https://github.com/huggingface/peft/issues/1249
            for p in model.parameters():
                if p.requires_grad and p.dtype == torch.float16:
                    logger.info_once('Convert trainable parameters from fp16 to fp32.')
                    p.data = p.data.to(dtype=torch.float32)
        elif args.train_type == 'full':
            model.train()
            model.requires_grad_(True)

            freeze_parameters(model, args.freeze_parameters_ratio, args.freeze_parameters, args.freeze_parameters_regex)
            if args.trainable_parameters or args.trainable_parameters_regex:
                activate_parameters(model, args.trainable_parameters, args.trainable_parameters_regex)
        else:
            raise ValueError(f'args.train_type: {args.train_type}')

        if args.use_galore:
            from swift.trainers.optimizers.galore import GaLoreConfig
            if args.galore_target_modules is None:
                args.galore_target_modules = find_all_linears(model)
            if args.galore_with_embedding:
                args.galore_target_modules += find_embedding(model)
            args.galore_config = GaLoreConfig(
                target_modules=args.galore_target_modules,
                rank=args.galore_rank,
                update_proj_gap=args.galore_update_proj_gap,
                galore_scale=args.galore_scale,
                proj_type=args.galore_proj_type,
                optim_per_parameter=args.galore_optim_per_parameter,
                quantize=args.galore_quantization,
                proj_quant=args.galore_proj_quant,
                proj_bits=args.galore_proj_bits,
                proj_group_size=args.galore_proj_group_size,
                cos_threshold=args.galore_cos_threshold,
                gamma_proj=args.galore_gamma_proj,
                queue_size=args.galore_queue_size,
            )
            args.training_args.galore_config = args.galore_config

        return model


def init_graph_encoder(model, args, template):
    """
    Initialize graph encoder for multi-view embedding training.

    This function attaches a spatial graph encoder to the model for processing
    graph modality in multi-view contrastive learning.

    Supports:
    - Qwen2VL: model.language_model or model.model
    - Phi-3.5-vision: model.model (contains model.layers)
    - Other models with model.layers or model.model.layers

    Args:
        model: The base model (Qwen2VL, Phi-3.5-vision, etc.)
        args: Training arguments
        template: Template instance with tokenizer

    Returns:
        model: Model with graph_encoder attached
    """
    from swift.utils import get_logger
    logger = get_logger()

    logger.info("🌍 Initializing graph encoder for multi-view training...")

    from swift.llm.model.model.graph_encoder_spatial import TextAttributedGraphEncoderSpatial
    from swift.utils import deep_getattr

    # Handle PEFT-wrapped models: unwrap to access base model structure
    # Recursively unwrap PeftModel until we get the actual base model
    base_model = model
    if isinstance(model, PeftModel):
        logger.debug(f"🔍 Unwrapping PeftModel to access base model for graph_encoder initialization")
        # Unwrap PEFT: get base_model
        base_model = model.base_model
        logger.debug(f"📍 First unwrap: {type(base_model).__name__}")

        # Continue unwrapping if base_model is still a PeftModel (nested PEFT)
        while isinstance(base_model, PeftModel):
            base_model = base_model.base_model
            logger.debug(f"📍 Unwrapping nested PeftModel: {type(base_model).__name__}")

        # For models like Phi-3.5-vision, we need to check if base_model has a 'model' attribute
        if hasattr(base_model, 'model'):
            unwrapped_model = base_model.model
            logger.debug(f"📍 Unwrapped to: {type(unwrapped_model).__name__} (via base_model.model)")
            # Check if unwrapped_model is still a PeftModel and continue unwrapping
            while isinstance(unwrapped_model, PeftModel):
                unwrapped_model = unwrapped_model.base_model
                logger.debug(f"📍 Unwrapping unwrapped_model PeftModel: {type(unwrapped_model).__name__}")
                # If it has a 'model' attribute, get that
                if hasattr(unwrapped_model, 'model'):
                    unwrapped_model = unwrapped_model.model
                    logger.debug(
                        f"📍 Further unwrapped to: {type(unwrapped_model).__name__} (via unwrapped_model.model)")
        else:
            unwrapped_model = base_model
            logger.debug(f"📍 Unwrapped to: {type(unwrapped_model).__name__} (via base_model)")
            # Check if unwrapped_model is still a PeftModel and continue unwrapping
            while isinstance(unwrapped_model, PeftModel):
                unwrapped_model = unwrapped_model.base_model
                logger.debug(f"📍 Unwrapping unwrapped_model PeftModel: {type(unwrapped_model).__name__}")
    else:
        unwrapped_model = model
        base_model = model

    # Final verification: ensure unwrapped_model is not a PeftModel
    # Recursively unwrap until we get a non-PeftModel
    original_unwrapped = unwrapped_model
    unwrap_count = 0
    while isinstance(unwrapped_model, PeftModel) and unwrap_count < 10:  # Safety limit
        unwrap_count += 1
        logger.warning(f"⚠️ unwrapped_model is still a PeftModel (unwrap #{unwrap_count}), continuing unwrap")
        if hasattr(unwrapped_model, 'base_model'):
            if hasattr(unwrapped_model.base_model, 'model'):
                # Check if base_model.model is also a PeftModel
                temp = unwrapped_model.base_model.model
                if isinstance(temp, PeftModel):
                    unwrapped_model = unwrapped_model.base_model
                else:
                    unwrapped_model = temp
            else:
                unwrapped_model = unwrapped_model.base_model
            logger.debug(f"📍 Unwrap #{unwrap_count} result: {type(unwrapped_model).__name__}")
        else:
            break

    if isinstance(unwrapped_model, PeftModel):
        logger.error(f"❌ Failed to fully unwrap PeftModel after {unwrap_count} attempts. "
                     f"Final type: {type(unwrapped_model).__name__}, "
                     f"Original: {type(original_unwrapped).__name__}, "
                     f"Model: {type(model).__name__}")
        # Try one more time with a different approach - get base_model directly
        if hasattr(unwrapped_model, 'get_base_model'):
            try:
                unwrapped_model = unwrapped_model.get_base_model()
                logger.debug(f"📍 Using get_base_model(): {type(unwrapped_model).__name__}")
            except:
                pass

    # Get language model reference using model architecture
    target_model = model  # Keep original model for attaching graph_encoder
    language_model = None

    # Try to get model_meta - it might be on the original model even if unwrapped_model doesn't have it
    model_meta = None
    if hasattr(unwrapped_model, 'model_meta'):
        model_meta = unwrapped_model.model_meta
    elif hasattr(model, 'model_meta'):
        model_meta = model.model_meta
        logger.debug(f"📍 Using model_meta from original model")

    if model_meta is not None and hasattr(model_meta, 'model_arch'):
        model_arch = model_meta.model_arch
        model_type = getattr(model_meta, 'model_type', None)

        # Check if it's Phi-3.5-vision (special handling)
        if model_type == 'phi3_vision' or (hasattr(model_arch, 'language_model') and
                                           model_arch.language_model == 'model.layers'):
            # For Phi-3.5-vision: language_model path is 'model.layers', but we need the parent model
            # The actual model structure is: model.model.layers (or base_model.model.layers for PEFT)
            # Handle both regular models and PeftModel-wrapped models
            # Check if unwrapped_model is a PeftModel (use type name check as fallback)
            is_peft = isinstance(unwrapped_model, PeftModel) or 'PeftModel' in str(type(unwrapped_model))

            if is_peft:
                logger.debug(f"🔍 Detected PeftModel for Phi-3.5-vision, unwrapped_model type: {type(unwrapped_model)}")
                # For PeftModel, check base_model.model.layers
                if hasattr(unwrapped_model, 'base_model'):
                    base = unwrapped_model.base_model
                    logger.debug(f"📍 base_model type: {type(base)}, has 'model': {hasattr(base, 'model')}")

                    # For Phi-3.5-vision with PEFT, the structure is:
                    # PeftModel -> LoraModel -> Phi3VForCausalLM -> Phi3VModel -> layers
                    # So we need: base_model.model.model.layers
                    if hasattr(base, 'model'):
                        inner_model = base.model  # This is Phi3VForCausalLM
                        logger.debug(f"📍 base_model.model type: {type(inner_model)}")

                        # Check if inner_model has a 'model' attribute (Phi3VModel)
                        if hasattr(inner_model, 'model') and hasattr(inner_model.model, 'layers'):
                            language_model = inner_model.model  # This is Phi3VModel
                            logger.info(
                                f"📍 Phi-3.5-vision detected (PeftModel): using base_model.model.model as language_model")
                        elif hasattr(inner_model, 'layers'):
                            language_model = inner_model
                            logger.info(
                                f"📍 Phi-3.5-vision detected (PeftModel): using base_model.model as language_model")
                        else:
                            raise ValueError(f"Phi-3.5-vision model structure not recognized in PeftModel. "
                                             f"unwrapped_model type: {type(unwrapped_model)}, "
                                             f"base_model type: {type(base)}, "
                                             f"base_model.model type: {type(inner_model)}, "
                                             f"base_model.model has 'model': {hasattr(inner_model, 'model')}, "
                                             f"base_model.model.model has 'layers': {hasattr(inner_model.model, 'layers') if hasattr(inner_model, 'model') else False}, "
                                             f"base_model.model has 'layers': {hasattr(inner_model, 'layers')}")
                    elif hasattr(base, 'layers'):
                        language_model = base
                        logger.info(f"📍 Phi-3.5-vision detected (PeftModel): using base_model as language_model")
                    else:
                        raise ValueError(f"Phi-3.5-vision model structure not recognized in PeftModel. "
                                         f"unwrapped_model type: {type(unwrapped_model)}, "
                                         f"base_model type: {type(base)}, "
                                         f"base_model has 'model': {hasattr(base, 'model')}, "
                                         f"base_model has 'layers': {hasattr(base, 'layers')}")
                else:
                    raise ValueError(f"Phi-3.5-vision PeftModel has no base_model attribute. "
                                     f"Type: {type(unwrapped_model)}, "
                                     f"Attributes: {[attr for attr in dir(unwrapped_model) if not attr.startswith('_')][:10]}")
            elif hasattr(unwrapped_model, 'model') and hasattr(unwrapped_model.model, 'layers'):
                language_model = unwrapped_model.model
                logger.info(f"📍 Phi-3.5-vision detected: using unwrapped_model.model as language_model")
            elif hasattr(unwrapped_model, 'layers'):
                # Direct access (unlikely but possible)
                language_model = unwrapped_model
                logger.info(f"📍 Phi-3.5-vision detected: using unwrapped_model as language_model (direct)")
            else:
                # Last resort: check if it's actually a PeftModel but isinstance failed
                if 'PeftModel' in str(type(unwrapped_model)) or hasattr(unwrapped_model, 'base_model'):
                    logger.warning(
                        f"⚠️ unwrapped_model appears to be PeftModel but isinstance check failed, trying base_model access")
                    if hasattr(unwrapped_model, 'base_model'):
                        base = unwrapped_model.base_model
                        if hasattr(base, 'model') and hasattr(base.model, 'layers'):
                            language_model = base.model
                            logger.info(
                                f"📍 Phi-3.5-vision detected (fallback): using base_model.model as language_model")
                        elif hasattr(base, 'layers'):
                            language_model = base
                            logger.info(f"📍 Phi-3.5-vision detected (fallback): using base_model as language_model")
                        else:
                            raise ValueError(f"Phi-3.5-vision model structure not recognized (fallback). "
                                             f"Type: {type(unwrapped_model)}, "
                                             f"base_model type: {type(base)}, "
                                             f"base_model has 'model': {hasattr(base, 'model')}, "
                                             f"base_model has 'layers': {hasattr(base, 'layers')}")
                    else:
                        raise ValueError(f"Phi-3.5-vision model structure not recognized. "
                                         f"Expected unwrapped_model.model.layers or unwrapped_model.layers, "
                                         f"got: {type(unwrapped_model)} (original: {type(model)}), "
                                         f"attributes: {[attr for attr in dir(unwrapped_model) if not attr.startswith('_')][:10]}")
                else:
                    raise ValueError(f"Phi-3.5-vision model structure not recognized. "
                                     f"Expected unwrapped_model.model.layers or unwrapped_model.layers, "
                                     f"got: {type(unwrapped_model)} (original: {type(model)}), "
                                     f"attributes: {[attr for attr in dir(unwrapped_model) if not attr.startswith('_')][:10]}")

        elif hasattr(model_arch, 'language_model') and model_arch.language_model:
            # For Qwen2VL and other models: use language_model path from architecture
            language_model_path = model_arch.language_model
            if isinstance(language_model_path, list):
                language_model_path = language_model_path[0]  # Take first if multiple

            # Special handling for 'model.layers' path (need parent model)
            if language_model_path == 'model.layers':
                # Get the parent model that contains the layers
                if hasattr(unwrapped_model, 'model') and hasattr(unwrapped_model.model, 'layers'):
                    language_model = unwrapped_model.model
                elif hasattr(unwrapped_model, 'layers'):
                    language_model = unwrapped_model
                else:
                    raise ValueError(
                        f"Model with 'model.layers' path but no unwrapped_model.model or unwrapped_model.layers found")
            else:
                # Use deep_getattr for other paths like 'model.language_model'
                language_model = deep_getattr(unwrapped_model, language_model_path)
        else:
            # Fallback: try common structures
            if hasattr(unwrapped_model, 'model') and hasattr(unwrapped_model.model, 'layers'):
                language_model = unwrapped_model.model
            elif hasattr(unwrapped_model, 'layers'):
                language_model = unwrapped_model
            else:
                raise ValueError(f"Unsupported model structure for graph encoder: {type(unwrapped_model)}, "
                                 f"model_arch.language_model: {getattr(model_arch, 'language_model', None)}")

    # Fallback when model_meta is not available
    elif hasattr(unwrapped_model, 'model') and hasattr(unwrapped_model.model, 'layers'):
        # Common structure: model.model.layers (Qwen2VL, Phi-3.5-vision, etc.)
        language_model = unwrapped_model.model
        logger.info(f"📍 Using unwrapped_model.model as language_model (fallback)")
    elif hasattr(unwrapped_model, 'layers'):
        # Direct language model
        language_model = unwrapped_model
        logger.info(f"📍 Using unwrapped_model as language_model (direct, fallback)")
    else:
        raise ValueError(f"Unsupported model structure for graph encoder: {type(unwrapped_model)}. "
                         f"Model attributes: {[attr for attr in dir(unwrapped_model) if not attr.startswith('_')]}")

    # Get model configuration
    hidden_dim = language_model.config.hidden_size
    model_device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype

    # Create graph encoder with spatial features
    graph_encoder = TextAttributedGraphEncoderSpatial(
        qwen_model=language_model,
        tokenizer=template.tokenizer,
        hidden_dim=hidden_dim,
        output_dim=hidden_dim,
        num_layers=getattr(args, 'graph_num_layers', 2),
        edge_dim=getattr(args, 'edge_dim', 64),
        training_phase="frozen",
        # Spatial features
        use_spatial_encoding=getattr(args, 'use_spatial_encoding', True),
        spatial_embed_dim=(getattr(args, 'spatial_embed_dim', 128) or 128),
        spatial_frequency_num=(getattr(args, 'spatial_frequency_num', 16) or 16),
        use_edge_features=getattr(args, 'use_edge_features', True),
        edge_use_distance=getattr(args, 'edge_use_distance', True),
        edge_use_direction=getattr(args, 'edge_use_direction', True),
        edge_use_displacement=getattr(args, 'edge_use_displacement', True),
        use_gat=getattr(args, 'use_gat', True),
        gat_heads=getattr(args, 'gat_heads', 4),
    )

    # Attach to model
    # For PEFT models, attach to base_model to match how it's loaded from checkpoints
    if isinstance(model, PeftModel):
        base_model.add_module('graph_encoder', graph_encoder)
        attach_target = base_model
    else:
        target_model.add_module('graph_encoder', graph_encoder)
        attach_target = target_model

    # Move to correct device and dtype
    # Move graph encoder to device and dtype
    # Note: With DeepSpeed ZeRO, this will be managed by DeepSpeed
    # We still need to set device/dtype for initial placement
    attach_target.graph_encoder = attach_target.graph_encoder.to(device=model_device, dtype=model_dtype)

    # Ensure graph encoder is registered as a module that DeepSpeed can manage
    # This is important for ZeRO-2/3 to properly distribute the graph encoder
    if not hasattr(attach_target, 'graph_encoder'):
        raise RuntimeError("Graph encoder was not properly attached to model")

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in attach_target.graph_encoder.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in attach_target.graph_encoder.parameters())

    logger.info(f"✅ Graph encoder initialized successfully")
    logger.info(f"   Device: {model_device}")
    logger.info(f"   Dtype: {model_dtype}")
    logger.info(f"   Hidden dim: {hidden_dim}")
    logger.info(f"   Spatial encoding: {graph_encoder.use_spatial_encoding}")
    logger.info(f"   Edge features: {graph_encoder.use_edge_features}")
    logger.info(f"   GNN type: {'GATv2' if graph_encoder.use_gat else 'GCN'}")
    logger.info(f"   GNN layers: {graph_encoder.num_layers}")
    logger.info(f"   Trainable params: {trainable_params:,} / {total_params:,}")

    return model
