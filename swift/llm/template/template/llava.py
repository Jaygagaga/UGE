# Copyright (c) Alibaba, Inc. and its affiliates.
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

import torch
import transformers
from packaging import version

from swift.utils import get_env_args
from ..base import Template
from ..constant import MLLMTemplateType
from ..register import TemplateMeta, register_template
from ..template_inputs import StdTemplateInputs
from ..utils import Context, Prompt, findall
from ..vision_utils import load_video_llava
from .llama import Llama3TemplateMeta
from .qwen import QwenTemplateMeta
from .utils import ChatmlTemplateMeta
from typing import Tuple
import torch


class LlavaHfTemplate(Template):
    placeholder_tokens = ['<image>']

    @property
    def image_token_index(self):
        if not hasattr(self, '_image_token_index'):
            self._image_token_index = self.tokenizer.convert_tokens_to_ids(self.processor.image_token)
        return self._image_token_index

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        assert media_type == 'image'
        return ['<image>\n']

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        images = inputs.images
        if images:
            image_processor = self.processor.image_processor
            image_inputs = image_processor(images, return_tensors='pt').to(self.model_info.torch_dtype)
            encoded['pixel_values'] = image_inputs['pixel_values']
            if 'image_sizes' in image_inputs:
                encoded['image_sizes'] = image_inputs['image_sizes']
            if version.parse(transformers.__version__) >= version.parse('4.47'):
                input_ids = encoded['input_ids']
                labels = encoded['labels']
                idx_list = findall(input_ids, self.image_token_index)  # <image>
                height, width = image_inputs['pixel_values'][0].shape[-2:]
                added_tokens_len = 0
                for i, idx in enumerate(idx_list):
                    if 'image_sizes' in image_inputs:
                        orig_height, orig_width = image_inputs['image_sizes'][i].tolist()
                        num_image_tokens = self.processor._get_number_of_features(orig_height, orig_width, height,
                                                                                  width)
                    else:
                        num_image_tokens = (height // self.processor.patch_size) * (
                            width // self.processor.patch_size) + self.processor.num_additional_image_tokens
                    if self.processor.vision_feature_select_strategy == 'default':
                        num_image_tokens -= 1
                    input_ids = input_ids[:added_tokens_len + idx] + [self.image_token_index] * num_image_tokens \
                        + input_ids[added_tokens_len + idx + 1:]
                    if labels is not None:
                        labels = labels[:added_tokens_len + idx] + [-100] * num_image_tokens \
                            + labels[added_tokens_len + idx + 1:]
                    added_tokens_len += num_image_tokens - 1
                encoded['input_ids'] = input_ids
                encoded['labels'] = labels
        return encoded


register_template(
    TemplateMeta(
        MLLMTemplateType.llava1_5_hf,
        prefix=['<s>'],
        prompt=['USER: {{QUERY}}\nASSISTANT:'],
        chat_sep=['</s>'],
        suffix=['</s>'],
        system_prefix=['<s>{{SYSTEM}}\n'],
        template_cls=LlavaHfTemplate,
    ))


class LlavaVideoHfTemplate(Template):

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index,
                    inputs: StdTemplateInputs) -> List[Context]:
        if media_type == 'image':
            return ['<image>\n']
        assert media_type == 'video'
        media_file = inputs.videos[index]
        if media_file.rsplit('.', 1)[-1] in {'jpg', 'png'}:
            return ['<image>\n']
        else:
            inputs.videos[index] = load_video_llava(inputs.videos[index])
            return ['<video>\n']

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        images = inputs.images or []
        videos = inputs.videos or []
        if len(videos) > 0:
            video_processor = self.processor.video_processor
            video_inputs = video_processor(videos, return_tensors='pt').to(self.model_info.torch_dtype)
            encoded['pixel_values_videos'] = video_inputs['pixel_values_videos']
        if len(images) > 0:
            image_processor = self.processor.image_processor
            image_inputs = image_processor(images, return_tensors='pt').to(self.model_info.torch_dtype)
            encoded['pixel_values'] = image_inputs['pixel_values']
            encoded['image_sizes'] = image_inputs['image_sizes']
        return encoded


register_template(
    TemplateMeta(
        MLLMTemplateType.llava_next_video_hf,
        prefix=['{{SYSTEM}} '],
        prompt=['USER: {{QUERY}} ASSISTANT:'],
        chat_sep=[' '],
        suffix=[['eos_token_id']],
        template_cls=LlavaVideoHfTemplate,
        auto_add_bos=True,
    ))


class Llava1_6HfTemplate(LlavaHfTemplate):

    def _data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        for b in batch:
            pixel_values = b.get('pixel_values')
            if pixel_values is not None:
                b['pixel_values'] = pixel_values.squeeze(0)  # 5d -> 4d
        res = super()._data_collator(batch, padding_to=padding_to)
        return res


@dataclass
class LlavaMistralTemplateMeta(TemplateMeta):
    prefix: Prompt = field(default_factory=lambda: ['<s>[INST] '])
    prompt: Prompt = field(default_factory=lambda: ['{{QUERY}} [/INST]'])
    chat_sep: Optional[Prompt] = field(default_factory=lambda: ['</s>[INST] '])
    suffix: Prompt = field(default_factory=lambda: ['</s>'])
    system_prefix: Optional[Prompt] = field(default_factory=lambda: ['<<SYS>>\n{{system}}\n<</SYS>>\n\n'])


register_template(LlavaMistralTemplateMeta(MLLMTemplateType.llava1_6_mistral_hf, template_cls=Llava1_6HfTemplate))


class Llava1_6MistralGraphTemplate(Llava1_6HfTemplate):
    """Extended LLaVA-1.6-Mistral template for multi-view training with graph modality"""
    placeholder_tokens = ['<image>', '<graph>']
    graph_token_id = -201  # Use a negative token ID as placeholder (similar to image token -200)
    
    def replace_tag(self, media_type: Literal['image', 'video', 'audio', 'graph'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        if media_type == 'graph':
            # Handle graph tokens for multi-view training
            if not inputs.graphs or index >= len(inputs.graphs):
                return ['<graph>\n']
            
            # Load graph if it's a file path
            if isinstance(inputs.graphs[index], str):
                from swift.llm.template.vision_utils import load_graph
                import os
                # Get max_nodes from environment variable (set by training args)
                max_nodes = os.environ.get('GRAPH_MAX_NODES')
                max_nodes = int(max_nodes) if max_nodes is not None and max_nodes.isdigit() else None
                inputs.graphs[index] = load_graph(inputs.graphs[index], max_nodes=max_nodes)
            
            if inputs.graphs[index] is None:
                return ['<graph>\n']
            
            # Get number of nodes for graph tokens
            num_nodes = 1
            if hasattr(inputs.graphs[index], 'num_nodes'):
                num_nodes = inputs.graphs[index].num_nodes
            elif hasattr(inputs.graphs[index], 'number_of_nodes'):
                num_nodes = inputs.graphs[index].number_of_nodes()
            
            # Return: <graph> + (num_nodes × <graph> tokens)
            # We'll use the same <graph> token multiple times, similar to how images work
            return ['<graph>\n'] + ['<graph>'] * num_nodes
        else:
            return super().replace_tag(media_type, index, inputs)
    
    def _pre_tokenize(self, context_list: List[Context], loss_scale_list: List[float],
                      inputs: StdTemplateInputs) -> Tuple[List[Context], List[float]]:
        """Override to handle <graph> tokens in addition to base template handling."""
        # First call parent to handle images, videos, audios
        context_list, loss_scale_list = super()._pre_tokenize(context_list, loss_scale_list, inputs)
        
        # Reset graph_idx
        if not hasattr(inputs, 'graph_idx'):
            inputs.graph_idx = 0
        
        # Process <graph> tokens
        res: List[Context] = []
        res_loss_scale: List[float] = []
        
        for context, loss_scale in zip(context_list, loss_scale_list):
            # If <graph> appears inside a larger string, split it into separate tokens
            if isinstance(context, str) and '<graph>' in context and context != '<graph>':
                # Split and interleave '<graph>' markers
                split_tokens: List[Context] = []
                parts = context.split('<graph>')
                for i, part in enumerate(parts):
                    if part:
                        split_tokens.append(part)
                    if i < len(parts) - 1:
                        split_tokens.append('<graph>')
            else:
                split_tokens = [context]

            for token in split_tokens:
                # Check for standalone <graph> token
                if token == '<graph>' and inputs.is_multimodal and inputs.graphs and inputs.graph_idx < len(inputs.graphs):
                    c_list = self.replace_tag('graph', inputs.graph_idx, inputs)
                    inputs.graph_idx += 1
                    ls = 0. if self.template_backend == 'swift' else 1.
                    res += c_list
                    res_loss_scale += [ls] * len(c_list)
                else:
                    res.append(token)
                    res_loss_scale.append(loss_scale)
        
        return res, res_loss_scale
    
    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        # Add graphs to encoded output for data collator
        if inputs.graphs:
            encoded['graphs'] = inputs.graphs
            encoded['has_graphs'] = True
            
            # Process graph tokens similar to how images are processed
            # We need to expand <graph> tokens in input_ids
            input_ids = encoded['input_ids']
            labels = encoded['labels']
            
            # Try to get graph token ID from tokenizer
            try:
                graph_token_id = self.tokenizer.convert_tokens_to_ids("<graph>")
            except:
                # If <graph> token doesn't exist, we'll handle it in _post_encode
                graph_token_id = None
            
            if graph_token_id is not None:
                # Find all graph token positions
                idx_list = findall(input_ids, graph_token_id)
                added_tokens_len = 0
                
                for i, idx in enumerate(idx_list):
                    if i < len(inputs.graphs) and inputs.graphs[i] is not None:
                        # Get number of nodes for this graph
                        graph = inputs.graphs[i]
                        if hasattr(graph, 'num_nodes'):
                            num_nodes = graph.num_nodes
                        elif hasattr(graph, 'number_of_nodes'):
                            num_nodes = graph.number_of_nodes()
                        else:
                            num_nodes = 1
                        
                        # Replace single <graph> token with num_nodes graph tokens
                        # Use the same graph_token_id for all graph tokens
                        input_ids = input_ids[:added_tokens_len + idx] + [graph_token_id] * num_nodes \
                            + input_ids[added_tokens_len + idx + 1:]
                        if labels is not None:
                            labels = labels[:added_tokens_len + idx] + [-100] * num_nodes \
                                + labels[added_tokens_len + idx + 1:]
                        added_tokens_len += num_nodes - 1
                
                encoded['input_ids'] = input_ids
                encoded['labels'] = labels
        else:
            encoded['has_graphs'] = False
        return encoded
    
    def _post_encode(self, model, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # First call parent to handle image embeddings
        result = super()._post_encode(model, inputs)
        
        if 'inputs_embeds' not in result:
            return result
        
        # Safety check: inputs must be a dict and contain required keys
        if not isinstance(inputs, dict) or 'input_ids' not in inputs:
            return result
        
        inputs_embeds = result['inputs_embeds']
        input_ids = inputs['input_ids']
        
        # Process graphs with graph encoder
        if inputs is not None and 'graphs' in inputs and hasattr(model, 'graph_encoder'):
            graphs = inputs.get('graphs')
            if graphs:
                try:
                    from swift.utils import get_logger
                    logger = get_logger()
                    
                    # Call graph encoder to get node embeddings
                    with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
                        batch_node_embeddings = model.graph_encoder(graphs)
                    
                    # Get graph token ID
                    # Try to find if there's a graph token in the tokenizer
                    graph_token_id = None
                    try:
                        graph_token_id = self.tokenizer.convert_tokens_to_ids("<graph>")
                    except:
                        # If <graph> token doesn't exist, we need to track it differently
                        # We'll use the approach similar to how images are handled
                        # In _encode, we expanded <graph> tokens, so we need to find them
                        # For now, we'll use a placeholder approach
                        pass
                    
                    # For each graph in the batch
                    # Note: batch_node_embeddings is a list of node embeddings for each graph
                    # We need to match them to the correct positions in the sequence
                    for graph_idx, node_embeddings in enumerate(batch_node_embeddings):
                        if graph_idx >= len(graphs):
                            continue
                            
                        node_embeddings = node_embeddings.to(inputs_embeds.device, inputs_embeds.dtype)
                        num_nodes = node_embeddings.shape[0]
                        
                        for batch_idx in range(input_ids.size(0)):
                            seq = input_ids[batch_idx]
                            
                            # Find graph token positions
                            # If we have a graph_token_id, use it
                            if graph_token_id is not None:
                                graph_token_positions = (seq == graph_token_id).nonzero(as_tuple=True)[0].tolist()
                            else:
                                # If no graph token ID, we need to track graph positions differently
                                # Since we expanded graph tokens in _encode, we can use a heuristic:
                                # Look for sequences of tokens that were marked as graph tokens
                                # For now, we'll skip this batch if we can't find the token
                                # In practice, you might want to add a special marker token
                                continue
                            
                            if graph_token_positions:
                                # Find the graph tokens for this specific graph
                                # Graph tokens are expanded in _encode, so we need to find consecutive sequences
                                # We'll use the first num_nodes graph tokens we find
                                if len(graph_token_positions) >= num_nodes:
                                    # Use the first num_nodes positions for this graph
                                    # Note: This assumes graphs appear in order in the sequence
                                    start_idx = graph_idx * num_nodes if graph_idx > 0 else 0
                                    end_idx = start_idx + num_nodes
                                    
                                    if end_idx <= len(graph_token_positions):
                                        graph_positions = graph_token_positions[start_idx:end_idx]
                                        
                                        # Inject node embeddings into sequence
                                        for i, pos in enumerate(graph_positions):
                                            if i < num_nodes:
                                                inputs_embeds[batch_idx, pos, :] = node_embeddings[i]
                                
                except Exception as e:
                    from swift.utils import get_logger
                    logger = get_logger()
                    logger.error(f"Error processing graph embeddings in LLaVA template: {e}")
                    import traceback
                    traceback.print_exc()
        
        # Remove 'graphs' from inputs and result after processing to avoid passing it to model.forward()
        # The graph embeddings have already been injected into inputs_embeds
        if 'graphs' in inputs:
            inputs.pop('graphs')
        if 'has_graphs' in inputs:
            inputs.pop('has_graphs')
        if 'graphs' in result:
            result.pop('graphs')
        if 'has_graphs' in result:
            result.pop('has_graphs')
        
        result['inputs_embeds'] = inputs_embeds
        return result
    
    def _data_collator_mm_data(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        res = super()._data_collator_mm_data(batch)
        # Gather graphs from batch
        graphs = self.gather_list(batch, 'graphs')
        if graphs:
            res['graphs'] = graphs
        return res


register_template(LlavaMistralTemplateMeta(MLLMTemplateType.llava1_6_mistral_graph_hf, template_cls=Llava1_6MistralGraphTemplate))

register_template(
    TemplateMeta(
        MLLMTemplateType.llava1_6_vicuna_hf,
        prefix=['<s>'],
        prompt=['USER: {{QUERY}} ASSISTANT:'],
        chat_sep=['</s>'],
        suffix=['</s>'],
        default_system=('A chat between a curious human and an artificial intelligence assistant. '
                        "The assistant gives helpful, detailed, and polite answers to the human's questions."),
        system_prefix=['<s>{{SYSTEM}} '],
        template_cls=Llava1_6HfTemplate))


class LLava1_6YiHfTemplate(Llava1_6HfTemplate):

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index,
                    inputs: StdTemplateInputs) -> List[Context]:
        if self.mode == 'vllm':
            return [[64000], '\n']
        else:
            return super().replace_tag(media_type, index, inputs)


register_template(ChatmlTemplateMeta(
    MLLMTemplateType.llava1_6_yi_hf,
    template_cls=LLava1_6YiHfTemplate,
))

register_template(Llama3TemplateMeta(
    MLLMTemplateType.llama3_llava_next_hf,
    template_cls=Llava1_6HfTemplate,
))

register_template(QwenTemplateMeta(MLLMTemplateType.llava_next_qwen_hf, template_cls=Llava1_6HfTemplate))


class LlavaOneVisionHfTemplate(Llava1_6HfTemplate):

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = Template._encode(self, inputs)
        images = inputs.images
        input_ids = encoded['input_ids']
        labels = encoded['labels']
        idx_list = findall(input_ids, 151646)  # <image>
        processor = self.processor
        if images:
            image_processor = processor.image_processor
            image_inputs = image_processor(images, return_tensors='pt').to(self.model_info.torch_dtype)
            height, width = image_inputs['pixel_values'][0].shape[-2:]
            added_tokens_len = 0
            for idx, pixel_v, image_size in zip(idx_list, image_inputs['pixel_values'], image_inputs['image_sizes']):
                if isinstance(image_size, torch.Tensor):
                    image_size = image_size.tolist()
                orig_height, orig_width = image_size
                num_image_tokens = processor._get_number_of_features(orig_height, orig_width, height, width)
                input_ids = input_ids[:added_tokens_len
                                      + idx] + [151646] * num_image_tokens + input_ids[added_tokens_len + idx + 1:]
                if labels is not None:
                    labels = labels[:added_tokens_len + idx] + [-100] * num_image_tokens + labels[added_tokens_len + idx
                                                                                                  + 1:]
                added_tokens_len += num_image_tokens - 1
            encoded['input_ids'] = input_ids
            encoded['labels'] = labels
            encoded['pixel_values'] = image_inputs['pixel_values']
            if 'image_sizes' in image_inputs:
                encoded['image_sizes'] = image_inputs['image_sizes']
        return encoded


register_template(
    QwenTemplateMeta(
        MLLMTemplateType.llava_onevision_hf,
        default_system=None,
        template_cls=LlavaOneVisionHfTemplate,
    ))


class LlavaLlama3_1HfTemplate(LlavaHfTemplate):
    # DaozeZhang
    system = ('You are a helpful language and vision assistant. '
              'You are able to understand the visual content that the user provides, '
              'and assist the user with a variety of tasks using natural language.')

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        if len(encoded['pixel_values'].shape) == 5:  # (1, num_patch, 3, H/W, W/H)
            encoded['pixel_values'] = torch.squeeze(encoded['pixel_values'], dim=0)  # (num_patch, 3, H/W, W/H)
        return encoded


register_template(
    Llama3TemplateMeta(
        MLLMTemplateType.llava_llama3_1_hf,
        default_system=LlavaLlama3_1HfTemplate.system,
        template_cls=LlavaLlama3_1HfTemplate,
    ))


class LLavaLlama3HfTemplate(Template):
    # xtuner
    image_placeholder = ['<image>\n']

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        raw_image = inputs.images
        if raw_image:
            pixel_values = self.processor.image_processor(raw_image, return_tensors='pt')['pixel_values']
            encoded['pixel_values'] = pixel_values.to(self.model_info.torch_dtype)
        return encoded


register_template(Llama3TemplateMeta(
    MLLMTemplateType.llava_llama3_hf,
    template_cls=LLavaLlama3HfTemplate,
))


class LLavaTemplate(Template):
    skip_prompt = False
    use_model = True

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index,
                    inputs: StdTemplateInputs) -> List[Context]:
        assert media_type == 'image'
        return [[-200], '\n']

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        images = inputs.images or []
        image_sizes = [x.size for x in images]
        from llava.mm_utils import process_images
        model = self.model.model
        if not hasattr(model, 'vision_tower'):
            model = model.model
        image_processor = model.vision_tower.image_processor
        if images:
            images_tensor = process_images(images, image_processor, model.config)
            encoded['images'] = images_tensor.to(model.dtype).squeeze(0)
            encoded['image_sizes'] = image_sizes
        return encoded

    def _data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = super()._data_collator(batch, padding_to=padding_to)
        images = [b['images'] for b in batch if 'images' in b]
        if images:
            res['images'] = images
            res['image_sizes'] = sum([b['image_sizes'] for b in batch if 'image_sizes' in b], start=[])
        return res


register_template(LlavaMistralTemplateMeta(MLLMTemplateType.llava1_6_mistral, template_cls=LLavaTemplate))

register_template(ChatmlTemplateMeta(MLLMTemplateType.llava1_6_yi, template_cls=LLavaTemplate))

register_template(
    Llama3TemplateMeta(
        MLLMTemplateType.llama3_llava_next,
        template_cls=LLavaTemplate,
        default_system=('You are a helpful language and vision assistant. '
                        'You are able to understand the visual content that the user provides, '
                        'and assist the user with a variety of tasks using natural language.'),
    ))

register_template(QwenTemplateMeta(MLLMTemplateType.llava_next_qwen, template_cls=LLavaTemplate))


class LLavaOneVision1_5Template(Template):
    image_token_id = 151655
    video_token_id = 151656
    placeholder_tokens = ['<|image_pad|>', '<|video_pad|>']
    use_model = True
    support_padding_free = True

    def init_env_args(self):
        super().init_env_args()
        self.bbox_format = get_env_args('QWENVL_BBOX_FORMAT', str, 'legacy')

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        from qwen_vl_utils import fetch_image, fetch_video
        assert media_type in {'image', 'video'}
        if media_type == 'image':
            inputs.images[index] = fetch_image({'image': inputs.images[index]})
            if self.mode == 'lmdeploy':
                return ['<|vision_start|>', [-100], '<|vision_end|>']
            else:
                return ['<|vision_start|><|image_pad|><|vision_end|>']
        else:
            video = inputs.videos[index]
            video, video_kwargs = fetch_video({'video': video}, return_video_sample_fps=True)
            inputs.mm_processor_kwargs.setdefault('fps', []).append(video_kwargs)
            tokens = ['<|vision_start|><|video_pad|><|vision_end|>']
            if isinstance(video, torch.Tensor):
                video = video.to(torch.uint8)
            inputs.videos[index] = video
            return tokens

    def replace_ref(self, ref: str, index: int, inputs: StdTemplateInputs) -> List[Context]:
        if self.bbox_format == 'legacy':
            return [f'<|object_ref_start|>{ref}<|object_ref_end|>']
        else:
            return [ref]

    def replace_bbox(self, bbox: List[int], index: int, inputs: StdTemplateInputs) -> List[Context]:
        if self.bbox_format == 'legacy':
            return [f'<|box_start|>{self._get_bbox_str(bbox)}<|box_end|>']
        else:
            return [str(bbox)]

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        processor = self.processor
        input_ids = encoded['input_ids']
        labels = encoded['labels']
        loss_scale = encoded.get('loss_scale', None)
        for media_type in ['images', 'videos']:
            mm_data = getattr(inputs, media_type)
            if mm_data:
                if media_type == 'images':
                    media_token = self.image_token_id
                    media_inputs = processor.image_processor(images=mm_data, return_tensors='pt', do_resize=False)
                    media_grid_thw = media_inputs['image_grid_thw']
                else:
                    kwargs = {}
                    if hasattr(processor, 'video_processor'):
                        processor_func = processor.video_processor
                    else:
                        processor_func = processor.image_processor
                        kwargs['images'] = None
                    media_inputs = processor_func(videos=mm_data, return_tensors='pt', do_resize=False, **kwargs)
                    media_grid_thw = media_inputs['video_grid_thw']
                    media_token = self.video_token_id
                idx_list = findall(input_ids, media_token)
                merge_length = processor.image_processor.merge_size**2

                def _get_new_tokens(i):
                    token_len = (media_grid_thw[i].prod() // merge_length)
                    return [media_token] * token_len

                input_ids, labels, loss_scale = self._extend_tokens(input_ids, labels, loss_scale, idx_list,
                                                                    _get_new_tokens)
                encoded.update(media_inputs)

        encoded['input_ids'] = input_ids
        encoded['labels'] = labels
        encoded['loss_scale'] = loss_scale
        return encoded

    def _post_encode(self, model, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if not self.is_training:
            return inputs
        input_ids = inputs['input_ids']
        base_model = self.get_base_model(model)
        if hasattr(base_model.model, 'embed_tokens'):
            inputs_embeds = base_model.model.embed_tokens(input_ids)
        else:
            inputs_embeds = base_model.model.language_model.embed_tokens(input_ids)
        inputs_embeds = self._get_inputs_embeds_hf(inputs_embeds, inputs, model.visual, self.processor, model.config)
        return {'inputs_embeds': inputs_embeds}


register_template(QwenTemplateMeta(MLLMTemplateType.llava_onevision1_5, template_cls=LLavaOneVision1_5Template))
