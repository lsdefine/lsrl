import torch
import torch.nn as nn

def decoding_layer_forward(self, hidden_states, attention_mask,
        position_ids, past_key_value, output_attentions, use_cache,
        cache_position, position_embeddings, **kwargs
    ):
        obj_device = next(self.parameters()).device
        if hidden_states.device != obj_device:
            hidden_states = hidden_states.to(obj_device)
        if position_embeddings[0].device != obj_device:
            position_embeddings = [x.to(obj_device) for x in position_embeddings]
        if attention_mask is not None: attention_mask = attention_mask.to(obj_device)
        if position_ids is not None: position_ids = position_ids.to(obj_device)
        if cache_position is not None: cache_position = cache_position.to(obj_device)
        # the next are same
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        outputs = (hidden_states,)
        if output_attentions: outputs += (self_attn_weights,)
        return outputs

def patch_qwen2_for_multi_gpus(model):
    for layer in model.model.layers:
        layer.forward = decoding_layer_forward.__get__(layer, nn.Module)
    layers = [model.model.embed_tokens]
    layers += [x for x in model.model.layers]
    layers += [model.model.norm, model.lm_head]
    device_count = torch.cuda.device_count()
    print(f"Detected {device_count} GPUs.")
    chunk_size = (len(layers)+3) // device_count + 1
    ids = [i // chunk_size for i in range(len(layers)+3)][2:-1]
    for layer, k in zip(layers, ids): layer.to(f'cuda:{k}')
    print(f'Make sure that the input should in cuda:0 and the output in cuda:{device_count - 1}...')