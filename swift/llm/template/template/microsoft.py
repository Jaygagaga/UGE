# Copyright (c) Alibaba, Inc. and its affiliates.
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

import json
import torch
from torch import nn

from ..base import Template
from ..constant import LLMTemplateType, MLLMTemplateType
from ..register import TemplateMeta, register_template
from ..template_inputs import StdTemplateInputs
from ..utils import Context, Prompt, findall
from ..vision_utils import load_file


class FlorenceTemplate(Template):
    # If it's an encoder-decoder architecture, the default settings are
    # loss_scale: 'last_round' and skip_prompt: False.
    is_encoder_decoder = True

    @staticmethod
    def _add_default_tags(inputs: StdTemplateInputs) -> None:
        return

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        return []

    def replace_bbox(self, bbox: List[int], index: int, inputs: StdTemplateInputs) -> List[Context]:
        return [''.join(f'<loc_{box}>' for box in bbox)]

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        processor = self.processor
        inputs.query = inputs.to_history()['query']
        new_query = processor._construct_prompts([inputs.query])[0]
        for i in reversed(range(len(inputs.messages))):
            if inputs.messages[i]['role'] == 'user':
                inputs.messages[i]['content'] = new_query
                break
        encoded = super()._encode(inputs)
        input_ids = encoded['prompt_input_ids']
        images = inputs.images or []
        labels = encoded['answer_labels']
        if labels is not None:
            labels = [0] + labels
        if images:
            pixel_values = processor.image_processor(
                images, return_tensors='pt')['pixel_values'].to(self.model_info.torch_dtype)
            encoded['pixel_values'] = pixel_values
        encoded['input_ids'] = input_ids
        encoded['labels'] = labels
        return encoded

    def _post_encode(self, model: nn.Module, inputs: Dict[str, Any]) -> Dict[str, Any]:
        inputs_embeds = model.get_input_embeddings()(inputs['input_ids'])
        pixel_values = inputs.get('pixel_values')
        if pixel_values is not None:
            image_features = model._encode_image(pixel_values)
            inputs_embeds, inputs['attention_mask'] = model._merge_input_ids_with_image_features(
                image_features, inputs_embeds)
        return {'inputs_embeds': inputs_embeds}

    def decode(self, generate_ids: List[int], **kwargs) -> Any:
        response = super().decode(generate_ids, **kwargs)
        template_inputs = kwargs.get('template_inputs')
        images = template_inputs.images
        image_size = None
        if images:
            image_size = (images[0].width, images[0].height)
        query_before, query_sep, query_after = template_inputs.query.partition('>')
        task = query_before + query_sep if query_sep else ''
        return json.dumps(self.processor.post_process_generation(response, task=task, image_size=image_size))


register_template(
    TemplateMeta(
        MLLMTemplateType.florence,
        prefix=['<s>'],
        prompt=['{{QUERY}}</s>'],
        chat_sep=None,
        suffix=['</s>'],
        template_cls=FlorenceTemplate,
    ))


@dataclass
class Phi3TemplateMeta(TemplateMeta):
    prefix: Prompt = field(default_factory=list)
    prompt: Prompt = field(default_factory=lambda: ['<|user|>\n{{QUERY}}<|end|>\n<|assistant|>\n'])
    chat_sep: Optional[Prompt] = field(default_factory=lambda: ['<|end|>\n'])
    suffix: Prompt = field(default_factory=lambda: ['<|end|>'])
    system_prefix: Optional[Prompt] = field(default_factory=lambda: ['<|system|>\n{{SYSTEM}}<|end|>\n'])
    auto_add_bos: bool = True


register_template(Phi3TemplateMeta(LLMTemplateType.phi3))


@dataclass
class Phi4TemplateMeta(TemplateMeta):
    prefix: Prompt = field(default_factory=list)
    prompt: Prompt = field(
        default_factory=lambda: ['<|im_start|>user<|im_sep|>{{QUERY}}<|im_end|><|im_start|>assistant<|im_sep|>'])
    chat_sep: Optional[Prompt] = field(default_factory=lambda: ['<|im_end|>'])
    suffix: Prompt = field(default_factory=lambda: ['<|im_end|>'])
    system_prefix: Optional[Prompt] = field(
        default_factory=lambda: ['<|im_start|>system<|im_sep|>{{SYSTEM}}<|im_end|>'])
    auto_add_bos: bool = True


register_template(Phi4TemplateMeta(LLMTemplateType.phi4))


class Phi3VisionTemplate(Template):
    image_placeholder = ['<|image|><s>\n']  # <|image|>\n

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        if self.mode == 'vllm':
            return [f'<|image_{index + 1}|>\n']  # <|image_1|>\n
        else:
            return super().replace_tag(media_type, index, inputs)

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        images = inputs.images or []
        encoded = super()._encode(inputs)
        input_ids = encoded['input_ids']
        labels = encoded['labels']
        idx_list = findall(input_ids, 32044)  # '<|image|>'

        if len(images) > 0:
            processor = self.processor
            encoded.update(processor.image_processor(images, return_tensors='pt'))
            assert len(idx_list) == len(images), f'len(idx_list): {len(idx_list)}, len(images): {len(images)}'
            res_input_ids = []
            res_labels = []
            num_img_tokens = encoded.pop('num_img_tokens').tolist()
            idx_list.insert(0, -1)
            for i in range(len(idx_list) - 1):
                image_token_id = -i - 1
                res_input_ids += input_ids[idx_list[i] + 1:idx_list[i + 1]] + [image_token_id] * num_img_tokens[i]
                if labels is not None:
                    res_labels += labels[idx_list[i] + 1:idx_list[i + 1]] + [-100] * num_img_tokens[i]
            res_input_ids += input_ids[idx_list[-1] + 1:]
            input_ids = res_input_ids
            if labels is not None:
                res_labels += labels[idx_list[-1] + 1:]
                labels = res_labels

        encoded['input_ids'] = input_ids
        encoded['labels'] = labels
        return encoded


class Phi4MMTemplate(Template):
    placeholder_tokens = ['<|endoftext10|>', '<|endoftext11|>']

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        if media_type == 'image':
            if self.mode == 'vllm':
                return [f'<|image_{index + 1}|>']  # <|image_1|>
            return [[-100]]
        elif media_type == 'audio':
            import soundfile as sf
            inputs.audios[index] = sf.read(load_file(inputs.audios[index]))
            return [[-200]]

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        input_ids = encoded['input_ids']
        labels = encoded['labels']
        loss_scale = encoded.get('loss_scale', None)
        images_idx = findall(input_ids, -100)
        audios_idx = findall(input_ids, -200)
        text = '\n'.join(['<|image_1|>'] * len(inputs.images) + ['<|audio_1|>'] * len(inputs.audios))
        new_encoded = self.processor(
            text=text, images=inputs.images or None, audios=inputs.audios or None, return_tensors='pt')
        placeholders = self._split_list(new_encoded.pop('input_ids')[0].tolist(), 198)

        def _get_new_tokens(i):
            return placeholders[i]

        encoded['input_ids'], encoded['labels'], encoded['loss_scale'] = self._extend_tokens(
            input_ids, labels, loss_scale, images_idx + audios_idx, _get_new_tokens)
        new_encoded.pop('attention_mask')
        encoded.update(new_encoded)
        return encoded

    def _data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = super()._data_collator(batch, padding_to=padding_to)
        keys = [
            'input_image_embeds', 'image_sizes', 'image_attention_mask', 'input_audio_embeds', 'audio_embed_sizes',
            'input_mode'
        ]
        inputs = self.fetch_inputs(batch, keys)
        for k, v in inputs.items():
            inputs[k] = torch.concat(v)
        res.update(inputs)
        return res


register_template(Phi3TemplateMeta(MLLMTemplateType.phi3_vision, template_cls=Phi3VisionTemplate))


class Phi3VisionGraphTemplate(Phi3VisionTemplate):
    """Extended Phi3Vision template for multi-view training with graph modality (image + text + graph)"""
    placeholder_tokens = ['<|graph_pad|>']
    special_tokens = ['<graph>']
    
    def replace_tag(self, media_type: Literal['image', 'video', 'audio', 'graph'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        if media_type == 'graph':
            # Handle graph tokens for multi-view training
            if not inputs.graphs or index >= len(inputs.graphs):
                return ['<|graph_start|>', '<|graph_pad|>', '<|graph_end|>']
            
            # Load graph if it's a file path
            if isinstance(inputs.graphs[index], str):
                from swift.llm.template.vision_utils import load_graph
                import os
                # Get max_nodes from environment variable (set by training args)
                max_nodes = os.environ.get('GRAPH_MAX_NODES')
                max_nodes = int(max_nodes) if max_nodes is not None and max_nodes.isdigit() else None
                inputs.graphs[index] = load_graph(inputs.graphs[index], max_nodes=max_nodes)
            
            if inputs.graphs[index] is None:
                return ['<|graph_start|>', '<|graph_pad|>', '<|graph_end|>']
            
            # Get number of nodes for graph pad tokens
            num_nodes = 1
            if hasattr(inputs.graphs[index], 'num_nodes'):
                num_nodes = inputs.graphs[index].num_nodes
            elif hasattr(inputs.graphs[index], 'number_of_nodes'):
                num_nodes = inputs.graphs[index].number_of_nodes()
            
            # Return: <graph_start> + (num_nodes × <graph_pad>) + <graph_end>
            return ['<|graph_start|>'] + ['<|graph_pad|>'] * num_nodes + ['<|graph_end|>']
        else:
            return super().replace_tag(media_type, index, inputs)
    
    def _pre_tokenize(self, context_list: List[Context], loss_scale_list: List[float],
                      inputs: StdTemplateInputs) -> Tuple[List[Context], List[float]]:
        """Override to handle <graph> tokens in addition to base template handling."""
        # First call parent to handle images
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
                if token == '<graph>' and inputs.graphs and inputs.graph_idx < len(inputs.graphs):
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
        else:
            encoded['has_graphs'] = False
        return encoded
    
    def _post_encode(self, model, inputs: Dict[str, Any]) -> Dict[str, Any]:
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
                    
                    # Attach central_node_id to each graph (for geo-typing task)
                    # This allows the graph encoder to only sanitize the target node
                    # For other tasks (perception, etc.), central_node_ids will be None/empty - no harm done!
                    if 'central_node_ids' in inputs and inputs['central_node_ids']:
                        central_node_ids = inputs['central_node_ids']
                        for graph, node_id in zip(graphs, central_node_ids):
                            if graph is not None and node_id is not None:
                                graph.central_node_id = node_id
                    
                    # Call graph encoder to get node embeddings
                    with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
                        batch_node_embeddings = model.graph_encoder(graphs)
                        # batch_node_embeddings is a list of tensors, one per graph
                        try:
                            import os
                            if os.environ.get('SWIFT_DEBUG_GRAPHS') == '1':
                                logger.info(f"Phi3VisionGraphTemplate: processed {len(batch_node_embeddings)} graphs")
                                if batch_node_embeddings:
                                    logger.info(f"  First graph embeddings shape: {batch_node_embeddings[0].shape}")
                        except Exception:
                            pass
                    
                    # Cache node embeddings on each graph object for reuse (e.g., spatial aux loss)
                    for graph, node_embeds in zip(graphs, batch_node_embeddings):
                        # try:
                        graph.cached_node_embeddings = node_embeds
                        # except Exception:
                        #     pass
                    
                    # Get token IDs for graph markers
                    GRAPH_PAD_ID = self.tokenizer.convert_tokens_to_ids("<|graph_pad|>")
                    GRAPH_START_ID = self.tokenizer.convert_tokens_to_ids("<|graph_start|>")
                    GRAPH_END_ID = self.tokenizer.convert_tokens_to_ids("<|graph_end|>")
                    
                    # Replace <graph_pad> tokens with actual node embeddings
                    for graph_idx, node_embeddings in enumerate(batch_node_embeddings):
                        node_embeddings = node_embeddings.to(inputs_embeds.device, inputs_embeds.dtype)
                        
                        for batch_idx in range(input_ids.size(0)):
                            seq = input_ids[batch_idx]
                            
                            # Find graph boundaries
                            graph_starts = (seq == GRAPH_START_ID).nonzero(as_tuple=True)[0].tolist()
                            graph_ends = (seq == GRAPH_END_ID).nonzero(as_tuple=True)[0].tolist()
                            
                            if graph_idx < len(graph_starts) and graph_idx < len(graph_ends):
                                start_idx = graph_starts[graph_idx]
                                end_idx = graph_ends[graph_idx]
                                
                                # Find all <graph_pad> positions
                                graph_pad_indices = [i for i in range(start_idx + 1, end_idx) 
                                                    if seq[i] == GRAPH_PAD_ID]
                                
                                if graph_pad_indices:
                                    num_pad = len(graph_pad_indices)
                                    num_nodes = node_embeddings.shape[0]
                                    
                                    # Match node embeddings to pad positions
                                    if num_nodes == num_pad:
                                        nodes_to_use = node_embeddings
                                    elif num_nodes > num_pad:
                                        nodes_to_use = node_embeddings[:num_pad]
                                    else:
                                        # Repeat nodes if fewer than pads
                                        repeats = (num_pad + num_nodes - 1) // num_nodes
                                        nodes_to_use = node_embeddings.repeat(repeats, 1)[:num_pad]
                                    
                                    # Inject node embeddings into sequence
                                    inputs_embeds[batch_idx, graph_pad_indices, :] = nodes_to_use
                                    try:
                                        import os
                                        if os.environ.get('SWIFT_DEBUG_GRAPHS') == '1':
                                            logger.info(f"[graphs-debug] _post_encode: injected {len(graph_pad_indices)} node embeds for graph_idx={graph_idx} batch_idx={batch_idx}")
                                    except Exception:
                                        pass
                
                except Exception as e:
                    from swift.utils import get_logger
                    logger = get_logger()
                    logger.error(f"Error processing graph embeddings: {e}")
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
    
    def _data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = super()._data_collator(batch, padding_to=padding_to)
        # Gather graphs from batch
        graphs = self.gather_list(batch, 'graphs')
        if graphs:
            res['graphs'] = graphs
        return res


# Register the graph template for Phi-3.5-vision
register_template(Phi3TemplateMeta(MLLMTemplateType.phi3_vision_graph, template_cls=Phi3VisionGraphTemplate))

register_template(Phi3TemplateMeta(
    MLLMTemplateType.phi4_multimodal,
    template_cls=Phi4MMTemplate,
))


class Phi4MMGraphTemplate(Phi4MMTemplate):
    """Extended Phi4MM template for multi-view training with graph modality (image + audio + text + graph)"""
    placeholder_tokens = ['<|endoftext10|>', '<|endoftext11|>', '<|graph_pad|>']
    special_tokens = ['<graph>']
    
    def replace_tag(self, media_type: Literal['image', 'video', 'audio', 'graph'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        if media_type == 'graph':
            # Handle graph tokens for multi-view training
            if not inputs.graphs or index >= len(inputs.graphs):
                return ['<|graph_start|>', '<|graph_pad|>', '<|graph_end|>']
            
            # Load graph if it's a file path
            if isinstance(inputs.graphs[index], str):
                from swift.llm.template.vision_utils import load_graph
                import os
                # Get max_nodes from environment variable (set by training args)
                max_nodes = os.environ.get('GRAPH_MAX_NODES')
                max_nodes = int(max_nodes) if max_nodes is not None and max_nodes.isdigit() else None
                inputs.graphs[index] = load_graph(inputs.graphs[index], max_nodes=max_nodes)
            
            if inputs.graphs[index] is None:
                return ['<|graph_start|>', '<|graph_pad|>', '<|graph_end|>']
            
            # Get number of nodes for graph pad tokens
            num_nodes = 1
            if hasattr(inputs.graphs[index], 'num_nodes'):
                num_nodes = inputs.graphs[index].num_nodes
            elif hasattr(inputs.graphs[index], 'number_of_nodes'):
                num_nodes = inputs.graphs[index].number_of_nodes()
            
            # Return: <graph_start> + (num_nodes × <graph_pad>) + <graph_end>
            return ['<|graph_start|>'] + ['<|graph_pad|>'] * num_nodes + ['<|graph_end|>']
        else:
            return super().replace_tag(media_type, index, inputs)
    
    def _pre_tokenize(self, context_list: List[Context], loss_scale_list: List[float],
                      inputs: StdTemplateInputs) -> Tuple[List[Context], List[float]]:
        """Override to handle <graph> tokens in addition to base template handling."""
        # First call parent to handle images and audio
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
                if token == '<graph>' and inputs.graphs and inputs.graph_idx < len(inputs.graphs):
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
        else:
            encoded['has_graphs'] = False
        return encoded
    
    def _post_encode(self, model, inputs: Dict[str, Any]) -> Dict[str, Any]:
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
                    
                    # Attach central_node_id to each graph (for geo-typing task)
                    # This allows the graph encoder to only sanitize the target node
                    # For other tasks (perception, etc.), central_node_ids will be None/empty - no harm done!
                    if 'central_node_ids' in inputs and inputs['central_node_ids']:
                        central_node_ids = inputs['central_node_ids']
                        for graph, node_id in zip(graphs, central_node_ids):
                            if graph is not None and node_id is not None:
                                graph.central_node_id = node_id
                    
                    # Call graph encoder to get node embeddings
                    with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
                        batch_node_embeddings = model.graph_encoder(graphs)
                        # batch_node_embeddings is a list of tensors, one per graph
                        try:
                            import os
                            if os.environ.get('SWIFT_DEBUG_GRAPHS') == '1':
                                logger.info(f"Phi4MMGraphTemplate: processed {len(batch_node_embeddings)} graphs")
                                if batch_node_embeddings:
                                    logger.info(f"  First graph embeddings shape: {batch_node_embeddings[0].shape}")
                        except Exception:
                            pass
                    
                    # Cache node embeddings on each graph object for reuse (e.g., spatial aux loss)
                    for graph, node_embeds in zip(graphs, batch_node_embeddings):
                        graph.cached_node_embeddings = node_embeds
                    
                    # Get token IDs for graph markers
                    GRAPH_PAD_ID = self.tokenizer.convert_tokens_to_ids("<|graph_pad|>")
                    GRAPH_START_ID = self.tokenizer.convert_tokens_to_ids("<|graph_start|>")
                    GRAPH_END_ID = self.tokenizer.convert_tokens_to_ids("<|graph_end|>")
                    
                    # Replace <graph_pad> tokens with actual node embeddings
                    for graph_idx, node_embeddings in enumerate(batch_node_embeddings):
                        node_embeddings = node_embeddings.to(inputs_embeds.device, inputs_embeds.dtype)
                        
                        for batch_idx in range(input_ids.size(0)):
                            seq = input_ids[batch_idx]
                            
                            # Find graph boundaries
                            graph_starts = (seq == GRAPH_START_ID).nonzero(as_tuple=True)[0].tolist()
                            graph_ends = (seq == GRAPH_END_ID).nonzero(as_tuple=True)[0].tolist()
                            
                            if graph_idx < len(graph_starts) and graph_idx < len(graph_ends):
                                start_idx = graph_starts[graph_idx]
                                end_idx = graph_ends[graph_idx]
                                
                                # Find all <graph_pad> positions
                                graph_pad_indices = [i for i in range(start_idx + 1, end_idx) 
                                                    if seq[i] == GRAPH_PAD_ID]
                                
                                if graph_pad_indices:
                                    num_pad = len(graph_pad_indices)
                                    num_nodes = node_embeddings.shape[0]
                                    
                                    # Match node embeddings to pad positions
                                    if num_nodes == num_pad:
                                        nodes_to_use = node_embeddings
                                    elif num_nodes > num_pad:
                                        nodes_to_use = node_embeddings[:num_pad]
                                    else:
                                        # Repeat nodes if fewer than pads
                                        repeats = (num_pad + num_nodes - 1) // num_nodes
                                        nodes_to_use = node_embeddings.repeat(repeats, 1)[:num_pad]
                                    
                                    # Inject node embeddings into sequence
                                    inputs_embeds[batch_idx, graph_pad_indices, :] = nodes_to_use
                                    try:
                                        import os
                                        if os.environ.get('SWIFT_DEBUG_GRAPHS') == '1':
                                            logger.info(f"[graphs-debug] _post_encode: injected {len(graph_pad_indices)} node embeds for graph_idx={graph_idx} batch_idx={batch_idx}")
                                    except Exception:
                                        pass
                
                except Exception as e:
                    from swift.utils import get_logger
                    logger = get_logger()
                    logger.error(f"Error processing graph embeddings: {e}")
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
    
    def _data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = super()._data_collator(batch, padding_to=padding_to)
        # Gather graphs from batch
        graphs = self.gather_list(batch, 'graphs')
        if graphs:
            res['graphs'] = graphs
        return res


# Register the graph template for Phi-4-multimodal
register_template(Phi3TemplateMeta(
    MLLMTemplateType.phi4_multimodal_graph,
    template_cls=Phi4MMGraphTemplate,
))
