# Copyright (c) Alibaba, Inc. and its affiliates.
from functools import partial
from types import MethodType
from typing import Any, Dict

from transformers import AutoConfig

from swift.llm import TemplateType
from swift.utils import get_device, get_env_args
from ..constant import LLMModelType, MLLMModelType
from ..model_arch import ModelArch
from ..patcher import patch_ignore_check_imports, patch_output_clone
from ..register import (Model, ModelGroup, ModelMeta, get_model_tokenizer_multimodal,
                        get_model_tokenizer_with_flash_attn, register_model)
from ..utils import ModelInfo, use_submodel_func


def get_model_tokenizer_phi3_vision(model_dir: str,
                                    model_info: ModelInfo,
                                    model_kwargs: Dict[str, Any],
                                    load_model: bool = True,
                                    **kwargs):
    processor_kwargs = {}
    if 'num_crops' in kwargs:
        processor_kwargs['num_crops'] = get_env_args('num_crops', int, kwargs['num_crops'])
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True, **processor_kwargs)
    model, tokenizer = get_model_tokenizer_with_flash_attn(
        model_dir, model_info, model_kwargs, load_model, tokenizer=processor.tokenizer, **kwargs)

    if load_model:
        patch_output_clone(model.model.vision_embed_tokens.wte)

    return model, processor


register_model(
    ModelMeta(
        MLLMModelType.phi3_vision,
        [
            ModelGroup([
                Model('LLM-Research/Phi-3-vision-128k-instruct', 'microsoft/Phi-3-vision-128k-instruct'),
                Model('LLM-Research/Phi-3.5-vision-instruct', 'microsoft/Phi-3.5-vision-instruct'),
            ])
        ],
        TemplateType.phi3_vision,
        partial(get_model_tokenizer_phi3_vision, num_crops=4),
        architectures=['Phi3VForCausalLM'],
        model_arch=ModelArch.phi3_vision,
        requires=['transformers>=4.36'],
        tags=['vision'],
    ))


def get_model_tokenizer_phi4_multimodal(*args, **kwargs):
    # Patch Phi4MMImageEmbedding before model loading to fix pe_weight.size() error
    # The error: ValueError: not enough values to unpack (expected 2, got 1)
    # occurs when pe_weight is 1D but code expects 2D in Phi4MMImageEmbedding.__init__
    # This is a known issue with the model - we patch the class when it's loaded
    _patch_phi4mm_image_embedding_before_load()
    
    model, processor = get_model_tokenizer_multimodal(*args, **kwargs)
    
    processor.audio_processor.audio_compression_rate = processor.audio_processor.compression_rate
    processor.audio_processor.audio_downsample_rate = processor.audio_processor.qformer_compression_rate
    processor.audio_processor.audio_feat_stride = processor.audio_processor.feat_stride
    del processor.audio_processor.feature_size
    del processor.audio_processor.sampling_rate
    del processor.audio_processor.padding_value
    del processor.__class__.chat_template
    processor.chat_template = None
    if model is not None:
        model.set_lora_adapter(['vision', 'speech'])
    return model, processor


def _patch_phi4mm_image_embedding_before_load():
    """Patch Phi4MMImageEmbedding class before model loading to fix pe_weight error"""
    import sys
    
    # First, try to patch any already-loaded classes in sys.modules
    for module_name, module in list(sys.modules.items()):
        if 'phi' in module_name.lower() and 'multimodal' in module_name.lower():
            if hasattr(module, 'Phi4MMImageEmbedding'):
                _patch_phi4mm_image_embedding_class(getattr(module, 'Phi4MMImageEmbedding'))
    
    # Monkey patch transformers.dynamic_module_utils to intercept class loading
    try:
        from transformers import dynamic_module_utils
        
        if hasattr(dynamic_module_utils, '_swift_phi4mm_patched'):
            return
        
        original_get_class = dynamic_module_utils.get_class_from_dynamic_module
        
        def patched_get_class(module_path, class_name, cache_dir=None):
            cls = original_get_class(module_path, class_name, cache_dir)
            if class_name == 'Phi4MMImageEmbedding':
                _patch_phi4mm_image_embedding_class(cls)
            return cls
        
        dynamic_module_utils.get_class_from_dynamic_module = patched_get_class
        dynamic_module_utils._swift_phi4mm_patched = True
    except Exception:
        # If patching fails, we'll try to patch after loading
        pass


def _patch_phi4mm_image_embedding_class(cls):
    """Patch a Phi4MMImageEmbedding class to handle 1D pe_weight"""
    if hasattr(cls, '_swift_phi4mm_patched'):
        return
    
    original_init = cls.__init__
    import torch
    
    def patched_init(self, config, **kwargs):
        try:
            return original_init(self, config, **kwargs)
        except ValueError as e:
            error_msg = str(e)
            if 'not enough values to unpack' in error_msg and 'pe_weight' in str(e):
                # The error occurs at line 91: L, D = pe_weight.size()
                # pe_weight is 1D but code expects 2D
                # We need to find where pe_weight is created and fix it
                
                # Check if pe_weight is in the config or kwargs
                pe_weight = None
                if hasattr(config, 'pe_weight'):
                    pe_weight = config.pe_weight
                elif 'pe_weight' in kwargs:
                    pe_weight = kwargs['pe_weight']
                elif hasattr(config, 'image_pe_weight'):
                    pe_weight = config.image_pe_weight
                
                if pe_weight is not None and isinstance(pe_weight, torch.Tensor):
                    if pe_weight.dim() == 1:
                        # Reshape to 2D - try square first
                        numel = pe_weight.numel()
                        size = int(numel ** 0.5)
                        if size * size == numel:
                            pe_weight = pe_weight.view(size, size)
                        else:
                            # Fallback: create 2D with shape (1, numel)
                            pe_weight = pe_weight.view(1, numel)
                        
                        # Update the config/kwargs
                        if hasattr(config, 'pe_weight'):
                            config.pe_weight = pe_weight
                        elif 'pe_weight' in kwargs:
                            kwargs['pe_weight'] = pe_weight
                        elif hasattr(config, 'image_pe_weight'):
                            config.image_pe_weight = pe_weight
                
                # Retry initialization with fixed pe_weight
                try:
                    return original_init(self, config, **kwargs)
                except Exception:
                    # If still fails, this might be a deeper issue
                    # Log and re-raise
                    from swift.utils import get_logger
                    logger = get_logger()
                    logger.warning(
                        "Failed to fix Phi4MMImageEmbedding pe_weight error. "
                        "This might be a transformers version compatibility issue. "
                        "Try updating transformers to a version compatible with Phi-4-multimodal."
                    )
                    raise
            raise
    
    cls.__init__ = patched_init
    cls._swift_phi4mm_patched = True


register_model(
    ModelMeta(
        MLLMModelType.phi4_multimodal,
        [ModelGroup([
            Model('LLM-Research/Phi-4-multimodal-instruct', 'microsoft/Phi-4-multimodal-instruct'),
        ])],
        TemplateType.phi4_multimodal,
        get_model_tokenizer_phi4_multimodal,
        architectures=['Phi4MMForCausalLM'],
        model_arch=ModelArch.phi4_multimodal,
        requires=['transformers>=4.36,<4.49', 'backoff', 'soundfile'],
        tags=['vision', 'audio'],
    ))


def get_model_tokenizer_florence(model_dir: str,
                                 model_info: ModelInfo,
                                 model_kwargs: Dict[str, Any],
                                 load_model: bool = True,
                                 **kwargs):
    model_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    model_config.vision_config.model_type = 'davit'  # fix merge-lora
    if model_kwargs['device_map'] == 'auto':
        model_kwargs['device_map'] = get_device()
    kwargs['model_config'] = model_config
    with patch_ignore_check_imports():
        model, processor = get_model_tokenizer_multimodal(model_dir, model_info, model_kwargs, load_model, **kwargs)

    if model is not None:
        model.vision_tower.enable_checkpoint = True
        use_submodel_func(model, 'language_model', ['generate', 'forward'])
    return model, processor


register_model(
    ModelMeta(
        MLLMModelType.florence,
        [
            # llama2
            ModelGroup([
                Model('AI-ModelScope/Florence-2-base-ft', 'microsoft/Florence-2-base-ft'),
                Model('AI-ModelScope/Florence-2-base', 'microsoft/Florence-2-base'),
                Model('AI-ModelScope/Florence-2-large', 'microsoft/Florence-2-large'),
                Model('AI-ModelScope/Florence-2-large-ft', 'microsoft/Florence-2-large-ft'),
            ]),
        ],
        TemplateType.florence,
        get_model_tokenizer_florence,
        architectures=['Florence2ForConditionalGeneration'],
        model_arch=ModelArch.florence,
        tags=['vision'],
    ))


def get_model_tokenizer_phi3_small(model_dir: str,
                                   model_info: ModelInfo,
                                   model_kwargs: Dict[str, Any],
                                   load_model: bool = True,
                                   **kwargs):
    model, tokenizer = get_model_tokenizer_with_flash_attn(model_dir, model_info, model_kwargs, load_model, **kwargs)

    def rotary_emb(self, query_states, key_states, **kwargs):
        q_type = query_states.dtype
        k_type = key_states.dtype
        query_states, key_states = self.rotory_emb_origin(query_states, key_states, **kwargs)
        query_states = query_states.to(q_type)
        key_states = key_states.to(k_type)
        return query_states, key_states

    if model is not None:
        for i in range(32):
            re = model.model.layers[i].self_attn.rotary_emb
            re.rotory_emb_origin = re.forward
            re.forward = MethodType(rotary_emb, re)
    return model, tokenizer


register_model(
    ModelMeta(
        LLMModelType.phi3_small,
        [
            ModelGroup([
                Model('LLM-Research/Phi-3-small-8k-instruct', 'microsoft/Phi-3-small-8k-instruct'),
                Model('LLM-Research/Phi-3-small-128k-instruct', 'microsoft/Phi-3-small-128k-instruct'),
            ]),
        ],
        TemplateType.phi3,
        get_model_tokenizer_phi3_small,
        architectures=['Phi3SmallForCausalLM'],
        model_arch=ModelArch.phi3_small,
        requires=['transformers>=4.36'],
    ))


def get_model_tokenizer_phi(model_dir: str,
                            model_info: ModelInfo,
                            model_kwargs: Dict[str, Any],
                            load_model: bool = True,
                            **kwargs):
    # TODO: check
    return get_model_tokenizer_with_flash_attn(model_dir, model_info, model_kwargs, load_model, **kwargs)


register_model(
    ModelMeta(
        LLMModelType.phi2,
        [
            ModelGroup([
                Model('AI-ModelScope/phi-2', 'microsoft/phi-2'),
            ]),
        ],
        TemplateType.default,
        get_model_tokenizer_phi,
        architectures=['PhiForCausalLM'],
        model_arch=ModelArch.phi2,
    ))

register_model(
    ModelMeta(
        LLMModelType.phi3,
        [
            ModelGroup([
                Model('LLM-Research/Phi-3-mini-4k-instruct', 'microsoft/Phi-3-mini-4k-instruct'),
                Model('LLM-Research/Phi-3-mini-128k-instruct', 'microsoft/Phi-3-mini-128k-instruct'),
                Model('LLM-Research/Phi-3-medium-4k-instruct', 'microsoft/Phi-3-medium-4k-instruct'),
                Model('LLM-Research/Phi-3-medium-128k-instruct', 'microsoft/Phi-3-medium-128k-instruct'),
                Model('LLM-Research/Phi-3.5-mini-instruct', 'microsoft/Phi-3.5-mini-instruct'),
            ]),
            ModelGroup([Model('LLM-Research/Phi-4-mini-instruct', 'microsoft/Phi-4-mini-instruct')])
        ],
        TemplateType.phi3,
        get_model_tokenizer_with_flash_attn,
        architectures=['Phi3ForCausalLM'],
        requires=['transformers>=4.36'],
        model_arch=ModelArch.phi3,
    ))

register_model(
    ModelMeta(
        LLMModelType.phi4,
        [
            ModelGroup([
                Model('LLM-Research/phi-4', 'microsoft/phi-4'),
            ]),
        ],
        TemplateType.phi4,
        get_model_tokenizer_with_flash_attn,
        architectures=['Phi3ForCausalLM'],
        requires=['transformers>=4.36'],
        model_arch=ModelArch.phi3,
    ))

register_model(
    ModelMeta(
        LLMModelType.phi3_moe,
        [
            ModelGroup([
                Model('LLM-Research/Phi-3.5-MoE-instruct', 'microsoft/Phi-3.5-MoE-instruct'),
            ]),
        ],
        TemplateType.phi3,
        get_model_tokenizer_with_flash_attn,
        architectures=['PhiMoEForCausalLM'],
        requires=['transformers>=4.36'],
    ))
