
import io, json
import torch

def tensor_to_bytes(t):
    buffer = io.BytesIO()
    torch.save(t, buffer)
    return buffer.getvalue()
def bytes_to_tensor(b):
    return torch.load(io.BytesIO(b), weights_only=True)
def make_bytes_list(blist):
    buffer = io.BytesIO()
    buffer.write(len(blist).to_bytes(4, 'big'))
    for b in blist:
        buffer.write(len(b).to_bytes(4, 'big'))
        buffer.write(b)
    return buffer.getvalue()
def bytes_list_to_list(b):
    buffer = io.BytesIO(b)
    num = int.from_bytes(buffer.read(4), 'big')
    blist = []
    for _ in range(num):
        l = int.from_bytes(buffer.read(4), 'big')
        blist.append(buffer.read(l))
    return blist

def json_to_bytes_list(data):
    tensors = [(k,v) for k, v in data.items() if isinstance(v, torch.Tensor)]
    others = {k:v for k, v in data.items() if not isinstance(v, torch.Tensor)}
    others['#tensors'] = [k for k, v in tensors]
    blist = [json.dumps(others).encode()]
    for _, v in tensors: blist.append(tensor_to_bytes(v))
    return make_bytes_list(blist)

def bytes_list_to_json(b):
    blist = bytes_list_to_list(b)
    if len(blist) < 1: return {}
    others = json.loads(blist[0])
    tkeys = others.pop('#tensors', [])
    tensors = {k:bytes_to_tensor(v) for k, v in zip(tkeys, blist[1:])}
    return {**others, **tensors}

def save_model(name, model, tokenizer=None):
    state_dict = model.state_dict()
    state_dict = type(state_dict)({k: v.cpu() for k, v in state_dict.items()})
    model.save_pretrained(name, state_dict=state_dict)
    if tokenizer is not None: tokenizer.save_pretrained(name)

def enable_gradient_checkpointing(model, ratio=1):
    model.train()
    model.gradient_checkpointing_enable()
    if ratio >= 1: return
    layers = model.model.layers
    total_layers = len(layers)
    start_idx = total_layers - int(total_layers * ratio)
    for i, layer in enumerate(layers):
        layer.gradient_checkpointing = i >= start_idx