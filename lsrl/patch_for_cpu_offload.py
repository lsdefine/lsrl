import torch, time
import torch.nn as nn
from transformers.models.qwen2.modeling_qwen2 import BaseModelOutputWithPast, DynamicCache

def qwen2_model_forward(self, input_ids=None,
    attention_mask=None, position_ids=None,
    past_key_values=None, inputs_embeds=None,
    use_cache=None, output_attentions=None, output_hidden_states=None,
    cache_position=None,**flash_attn_kwargs,
):
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    device = torch.device('cuda')
    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids).to(device)
    if use_cache and past_key_values is None:
        past_key_values = DynamicCache()
    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device)

    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    causal_mask = self._update_causal_mask(attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions)
    hidden_states = inputs_embeds
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    
    prefetch_stream = torch.cuda.Stream()     
    group  = 8

    n = self.config.num_hidden_layers
    layer_idx = 0
    self.layers[:group].to(device, non_blocking=True)
    while layer_idx < n:
        end = min(layer_idx + group, n)
        next_start = end  
        next_end = min(next_start + group, n)

        with torch.cuda.stream(prefetch_stream):
            self.layers[next_start:next_end].to(device, non_blocking=True)

        for ii in range(layer_idx, end):
            layer = self.layers[ii]
            hidden_states = layer(hidden_states, 
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **flash_attn_kwargs,)[0]

        for ii in range(layer_idx, end):
            if ii < 12: continue
            self.layers[ii].to('cpu', non_blocking=True)

        layer_idx = end
        torch.cuda.synchronize()  # 不能去掉

    self.norm.to(device)
    hidden_states = self.norm(hidden_states)
    torch.cuda.empty_cache()    
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values if use_cache else None,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )

def patch_qwen2(model):
    tic = time.time()
    model.model.forward = qwen2_model_forward.__get__(model.model, nn.Module)
    print('preparing model...  need 20~80s')
    model.lm_head.to('cuda')
    for p in model.parameters():
        if p.device.type == 'cpu': p.data = p.data.pin_memory() 
    print(f'Model prepared in {time.time() - tic:.2f} seconds.')
