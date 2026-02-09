# Copyright (c) Alibaba, Inc. and its affiliates.
import math
import os
import sys
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch.nn as nn
from peft import PeftModel
from transformers import Trainer

from swift.trainers.optimizers.galore import create_optimizer_and_scheduler
from swift.utils import get_dist_setting, get_logger

if TYPE_CHECKING:
    from swift.trainers import TrainingArguments


def calculate_max_steps(args: 'TrainingArguments', dataset) -> int:
    if args.max_steps and args.max_steps > 0:
        max_steps = args.max_steps
    else:
        len_dataset = len(dataset)
        _, _, world_size, _ = get_dist_setting()
        total_train_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps * world_size
        num_update_steps_per_epoch = len_dataset // total_train_batch_size
        num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
        max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
    return max_steps


def create_galore_optimizer(args: 'TrainingArguments', model, dataset):
    training_steps = calculate_max_steps(args, dataset)
    optimizer, lr_scheduler = create_optimizer_and_scheduler(
        model, args, args.galore_config, training_steps, lr=args.learning_rate, weight_decay=args.weight_decay)
    # trainer cannot serialize galore_config
    args.galore_config = None
    return optimizer, lr_scheduler


def create_lorap_optimizer(args: 'TrainingArguments', model, dataset):
    optimizer_grouped_parameters = None
    if hasattr(model, 'create_optimizer_param_groups'):
        # Lora+ parameter groups
        optimizer_grouped_parameters = model.create_optimizer_param_groups(
            lr=args.learning_rate, weight_decay=args.weight_decay)

    if optimizer_grouped_parameters is None:
        # Default parameter groups
        decay_parameters = Trainer.get_decay_parameter_names(None, model)
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in model.named_parameters() if (n in decay_parameters and p.requires_grad)],
                'weight_decay': args.weight_decay,
            },
            {
                'params': [p for n, p in model.named_parameters() if (n not in decay_parameters and p.requires_grad)],
                'weight_decay': 0.0,
            },
        ]
    optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(args)
    return optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs), None


def create_muon_optimizer(args: 'TrainingArguments', model, dataset):
    from swift.llm import git_clone_github
    if not args.local_repo_path:
        args.local_repo_path = git_clone_github('https://github.com/MoonshotAI/Moonlight.git')
    sys.path.append(os.path.join(args.local_repo_path, 'examples'))
    from toy_train import Muon

    # parse args.optim_args
    optim_args = {}
    if args.optim_args:
        for mapping in args.optim_args.replace(' ', '').split(','):
            key, value = mapping.split('=')
            optim_args[key] = value

    model_arch = model.model_meta.model_arch
    embed_key = getattr(model_arch, 'embedding', None) or 'embed_tokens'
    lm_head_key = getattr(model_arch, 'lm_head', None) or 'lm_head'
    muon_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad and p.ndim >= 2 and embed_key not in n and lm_head_key not in n
    ]
    adamw_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad and not (p.ndim >= 2 and embed_key not in n and lm_head_key not in n)
    ]

    return Muon(
        lr=args.learning_rate,
        wd=args.weight_decay,
        muon_params=muon_params,
        adamw_params=adamw_params,
        adamw_betas=(args.adam_beta1, args.adam_beta2),
        adamw_eps=args.adam_epsilon,
        **optim_args,
    ), None


def get_param_startswith(model,
                         chosen_prefix: List[str],
                         rejected_prefix: Optional[List[str]] = None) -> List[Tuple[str, nn.Parameter]]:
    chosen_prefix = chosen_prefix or []
    rejected_prefix = rejected_prefix or []
    res = []
    if not chosen_prefix:
        return res
    is_peft_model = isinstance(model, PeftModel)
    if is_peft_model:
        model = model.model
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_rejected = False
        for prefix in rejected_prefix:
            if n.startswith(prefix):
                is_rejected = True
                break
        if is_rejected:
            continue
        for prefix in chosen_prefix:
            if n.startswith(prefix):
                if is_peft_model:
                    n = f'base_model.model.{n}'
                res.append((n, p))
                break
    return res


def create_multimodal_optimizer(args: 'TrainingArguments', model, dataset):
    """ViT/Aligner/LLM use different learning rates."""
    decay_parameters = set(Trainer.get_decay_parameter_names(None, model))
    model_arch = model.model_meta.model_arch
    vit_parameters = get_param_startswith(model, model_arch.vision_tower, model_arch.aligner)
    aligner_parameters = get_param_startswith(model, model_arch.aligner)
    llm_parameters = get_param_startswith(model, model_arch.language_model)
    optimizer_grouped_parameters = []
    for lr, parameters in zip([args.vit_lr, args.aligner_lr, args.learning_rate],
                              [vit_parameters, aligner_parameters, llm_parameters]):
        if lr is None:
            lr = args.learning_rate
        for use_wd, wd in zip([False, True], [0., args.weight_decay]):
            if use_wd:
                params = [p for n, p in parameters if n in decay_parameters]
            else:
                params = [p for n, p in parameters if n not in decay_parameters]
            if not params:
                continue
            optimizer_grouped_parameters.append({
                'params': params,
                'weight_decay': wd,
                'lr': lr,
            })
    optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(args, model)
    return optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs), None


def create_multiview_graph_optimizer(args: 'TrainingArguments', model, dataset):
    """
    Optimizer for multiview training with graph modality.
    
    This optimizer uses different learning rates for:
    - Image encoder (ViT): smaller LR to preserve image-text alignment
    - Text encoder (LLM): smaller LR to preserve image-text alignment  
    - Graph encoder: normal LR for learning graph representations
    
    This helps maintain the relative positions of image and text embeddings
    in the embedding space when graph modality is introduced.
    
    This optimizer works by controlling learning rates to preserve image-text alignment,
    without requiring any additional loss terms.
    """
    decay_parameters = set(Trainer.get_decay_parameter_names(None, model))
    model_arch = model.model_meta.model_arch
    
    # Get parameters for different components
    vit_parameters = get_param_startswith(model, model_arch.vision_tower, model_arch.aligner)
    aligner_parameters = get_param_startswith(model, model_arch.aligner)
    llm_parameters = get_param_startswith(model, model_arch.language_model)
    
    # Get graph encoder parameters
    # IMPORTANT: Exclude qwen_model parameters from graph_encoder since they're shared
    # with the main LLM and should only be in the LLM parameter group
    graph_parameters = []
    is_peft_model = isinstance(model, PeftModel)
    base_model = model.model if is_peft_model else model
    if hasattr(base_model, 'graph_encoder'):
        for n, p in base_model.graph_encoder.named_parameters():
            if not p.requires_grad:
                continue
            # Skip qwen_model parameters - they're shared with main LLM
            if 'qwen_model' in n:
                continue
            if is_peft_model:
                n = f'base_model.model.graph_encoder.{n}'
            else:
                n = f'graph_encoder.{n}'
            graph_parameters.append((n, p))
    
    # Set learning rates
    # Use smaller LR for image/text encoders to preserve alignment
    image_text_lr_scale = getattr(args, 'image_text_lr_scale', 0.1)  # Default: 10% of base LR
    graph_lr = getattr(args, 'graph_lr', args.learning_rate)  # Default: same as base LR
    
    vit_lr = args.vit_lr if args.vit_lr is not None else args.learning_rate * image_text_lr_scale
    aligner_lr = args.aligner_lr if args.aligner_lr is not None else args.learning_rate * image_text_lr_scale
    llm_lr = args.learning_rate * image_text_lr_scale
    
    optimizer_grouped_parameters = []
    
    # Add parameter groups for ViT, Aligner, LLM (with smaller LR)
    for lr, parameters in zip([vit_lr, aligner_lr, llm_lr],
                              [vit_parameters, aligner_parameters, llm_parameters]):
        for use_wd, wd in zip([False, True], [0., args.weight_decay]):
            if use_wd:
                params = [p for n, p in parameters if n in decay_parameters]
            else:
                params = [p for n, p in parameters if n not in decay_parameters]
            if not params:
                continue
            optimizer_grouped_parameters.append({
                'params': params,
                'weight_decay': wd,
                'lr': lr,
            })
    
    # Add parameter groups for graph encoder (with normal LR)
    if graph_parameters:
        for use_wd, wd in zip([False, True], [0., args.weight_decay]):
            if use_wd:
                params = [p for n, p in graph_parameters if n in decay_parameters]
            else:
                params = [p for n, p in graph_parameters if n not in decay_parameters]
            if not params:
                continue
            optimizer_grouped_parameters.append({
                'params': params,
                'weight_decay': wd,
                'lr': graph_lr,
            })
    
    optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(args, model)
    optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
    
    # Log learning rate configuration
    logger = get_logger()
    logger.info(f"Multiview Graph Optimizer Configuration:")
    logger.info(f"  Image encoder (ViT) LR: {vit_lr}")
    logger.info(f"  Aligner LR: {aligner_lr}")
    logger.info(f"  Text encoder (LLM) LR: {llm_lr}")
    logger.info(f"  Graph encoder LR: {graph_lr}")
    logger.info(f"  Image/Text LR scale: {image_text_lr_scale}x")
    
    return optimizer, None


def create_ogd_multiview_graph_optimizer(args: 'TrainingArguments', model, dataset):
    """
    OGD-enabled optimizer for multiview training with graph modality.
    
    This optimizer:
    1. Uses different learning rates for image/text vs graph (like multiview_graph)
    2. Wraps the optimizer with OGD to project gradients orthogonally
    3. Preserves learned image-text similarity when adding graph modality
    
    OGD works by:
    - Storing feature vectors from pretrained model (image-text alignment)
    - Projecting gradients orthogonally to these features during training
    - Preventing catastrophic forgetting of image-text relationships
    """
    from swift.plugin.ogd_core import OGDTrainer, OGDOptimizer, identify_qwen2vl_layers, identify_phi3_vision_layers
    
    logger = get_logger()  # Use module-level import
    print("[OGD] create_ogd_multiview_graph_optimizer() ENTRY", flush=True)
    logger.info("[OGD] create_ogd_multiview_graph_optimizer() ENTRY")
    
    # First create the base optimizer (same as multiview_graph)
    print("[OGD] Creating base multiview_graph optimizer...", flush=True)
    logger.info("[OGD] Creating base multiview_graph optimizer...")
    base_optimizer, _ = create_multiview_graph_optimizer(args, model, dataset)
    print("[OGD] ✅ Base optimizer created", flush=True)
    logger.info("[OGD] ✅ Base optimizer created")
    
    # Initialize OGD trainer if enabled
    use_ogd = getattr(args, 'use_ogd', False)
    
    # Also check if we can access the original TrainArguments via training_args attribute
    if not use_ogd and hasattr(args, 'training_args'):
        # Try to get use_ogd from the original TrainArguments
        original_args = args.training_args
        if hasattr(original_args, 'use_ogd'):
            use_ogd = original_args.use_ogd
            print(f"[OGD] Found use_ogd={use_ogd} from original TrainArguments", flush=True)
            logger.info(f"[OGD] Found use_ogd={use_ogd} from original TrainArguments")
    
    # FORCE OGD creation when optimizer='ogd_multiview_graph' is explicitly set
    # This ensures OGD is created even if use_ogd flag is not properly propagated
    if not use_ogd:
        print(f"[OGD] use_ogd=False, but optimizer='ogd_multiview_graph' is set - FORCING OGD creation", flush=True)
        logger.warning(f"[OGD] use_ogd=False, but optimizer='ogd_multiview_graph' is set - FORCING OGD creation")
        use_ogd = True  # Force OGD creation
    
    print(f"[OGD] OGD enabled: use_ogd={use_ogd}", flush=True)
    logger.info(f"[OGD] OGD enabled: use_ogd={use_ogd}")
    
    print("[OGD] Initializing OGD for multiview graph training...", flush=True)
    logger.info("[OGD] Initializing OGD for multiview graph training...")
    
    # Identify protected layers
    if getattr(args, 'ogd_protected_layers', None):
        protected_layers = args.ogd_protected_layers
        logger.info(f"[OGD] Using user-specified protected layers: {len(protected_layers)}")
    else:
        # Auto-detect layers based on model type
        model_type = getattr(args, 'model_type', None) or (getattr(args.model_meta, 'model_type', None) if hasattr(args, 'model_meta') else None)
        
        # Check if it's Phi-3.5-vision (multimodal: vision + text)
        if model_type == 'phi3_vision' or (hasattr(model, 'config') and hasattr(model.config, 'model_type') and 'phi' in str(model.config.model_type).lower()):
            # Try to get number of layers from config
            num_llm_layers = getattr(model.config, 'num_hidden_layers', 32) if hasattr(model, 'config') else 32
            # Phi-3.5-vision vision encoder size (adjust if known)
            num_vision_blocks = 24  # Default, adjust based on actual model
            protected_layers = identify_phi3_vision_layers(model, num_vision_blocks=num_vision_blocks, num_llm_layers=num_llm_layers)
            logger.info(f"[OGD] Auto-detected {len(protected_layers)} layers to protect for Phi-3.5-vision")
        else:
            # Default to Qwen2VL layer identification
            protected_layers = identify_qwen2vl_layers(model)
            logger.info(f"[OGD] Auto-detected {len(protected_layers)} layers to protect (Qwen2VL)")
    
    if not protected_layers:
        logger.warning("[OGD] No protected layers found! OGD will not work. Disabling OGD.")
        return base_optimizer, None
    
    # Create OGD trainer
    ogd_trainer = OGDTrainer(
        model=model,
        protected_layers=protected_layers,
        memory_size=getattr(args, 'ogd_memory_size', 1000),
        use_ogd_plus=getattr(args, 'ogd_use_ogd_plus', False),
        use_gradients=getattr(args, 'ogd_use_gradients', False),
        device=next(model.parameters()).device
    )
    
    # Store OGD trainer in args for later use (memory update, etc.)
    # Store in both args and training_args for compatibility
    args._ogd_trainer = ogd_trainer
    if hasattr(args, 'training_args'):
        args.training_args._ogd_trainer = ogd_trainer
    
    # Wrap optimizer with OGD
    ogd_optimizer = OGDOptimizer(base_optimizer, ogd_trainer)
    
    print("[OGD] ✅ OGD optimizer created successfully", flush=True)
    print(f"[OGD] Memory size: {ogd_trainer.memory_size} features per layer", flush=True)
    print(f"[OGD] Protected layers: {len(protected_layers)}", flush=True)
    logger.info("[OGD] OGD optimizer created successfully")
    logger.info(f"[OGD] Memory size: {ogd_trainer.memory_size} features per layer")
    logger.info(f"[OGD] Protected layers: {len(protected_layers)}")
    
    return ogd_optimizer, None


# Add your own optimizers here, use --optimizer xxx to train
optimizers_map = {
    'galore': create_galore_optimizer,
    'lorap': create_lorap_optimizer,
    'muon': create_muon_optimizer,
    'multimodal': create_multimodal_optimizer,
    'multiview_graph': create_multiview_graph_optimizer,
    'ogd_multiview_graph': create_ogd_multiview_graph_optimizer,
}
