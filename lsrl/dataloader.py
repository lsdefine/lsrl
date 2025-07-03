import random
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
class BaseBatchDataset(Dataset):
    def __init__(self, batches): self.batches = batches
    def __len__(self): return len(self.batches)
    def __getitem__(self, idx): return self.batches[idx]

def make_dataloader(batches):
    dataset = BaseBatchDataset(batches)
    sampler = DistributedSampler(dataset) if torch.distributed.is_initialized() else None
    return DataLoader(dataset, batch_size=1, shuffle=False, sampler=sampler, collate_fn=lambda x:x[0])

def convert_to_io_format(item):
    if isinstance(item, list): item = item[0]
    input_parts = [
        item.get("instruction", ""),
        item.get("input", ""),
        item.get("prompt", "")
    ]
    return {
        "input": "\n\n".join(part for part in input_parts if part).strip(),
        "output": item.get("output") or item.get("response", "")
    }

def convert_to_ua_format(item, system_prompt=None):
    input_text = f'<|im_start|>system\n{system_prompt}\n<|im_end|>\n' if system_prompt else ''
    input_text += f'<|im_start|>user\n{item["input"]}\n<|im_end|>\n'
    input_text += f'<|im_start|>assistant\n'
    output_text = f'{item["output"]}\n<|im_end|><|endoftext|>'
    return {"input": input_text, "output": output_text}

def convert_to_token_ids(item, tokenizer):
    input_ids = tokenizer.encode(item["input"], add_special_tokens=False)
    output_ids = tokenizer.encode(item["output"], add_special_tokens=False)
    return {
        "input_ids": input_ids + output_ids,
        "labels": [-100] * len(input_ids) + output_ids
    }

def convert_to_batches(data, tokenizer, max_len=4096, mode='packing', batch_size=4, test=False):
    def pad_and_tensorize(input_ids, labels, item=None, pad_length=0):
        if max(labels) == -100:
            print(f"Found labels all -100, skipping item ...  item length: {len(item['input_ids'])}")
            return None
        if pad_length > 0:
            input_ids += [tokenizer.pad_token_id] * (pad_length - len(input_ids))
            labels += [-100] * (pad_length - len(labels))
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long).unsqueeze(0),
            'labels': torch.tensor(labels, dtype=torch.long).unsqueeze(0)
        }
    batches = []
    clever_append = lambda b, x: b.append(x) if x is not None else None
    if mode == 'padding':
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            blen = min(max(len(item['input_ids']) for item in batch), max_len)
            padded_batch = []
            for item in batch:
                input_ids, labels = item['input_ids'][:blen], item['labels'][:blen]
                clever_append(padded_batch, pad_and_tensorize(input_ids, labels, item, blen))
            if padded_batch:
                batches.append({
                    'input_ids': torch.cat([item['input_ids'] for item in padded_batch]),
                    'labels': torch.cat([item['labels'] for item in padded_batch])
                })
            if test and len(batches) >= 3: break
    elif mode == 'packing':
        packed_input_ids, packed_labels = [], []
        last_item = None
        for item in data:
            input_ids, labels = item['input_ids'][:max_len], item['labels'][:max_len]
            if len(packed_input_ids) + len(input_ids) > max_len:
                clever_append(batches, pad_and_tensorize(packed_input_ids, packed_labels, last_item))
                packed_input_ids, packed_labels = [], []    
            packed_input_ids.extend(input_ids)
            packed_labels.extend(labels)
            last_item = item
            if test and len(batches) >= 3: break
        if packed_input_ids:
            clever_append(batches, pad_and_tensorize(packed_input_ids, packed_labels, last_item))
    return batches

lmap = lambda f, x: list(z for z in map(f, x) if z is not None)
class SFTDataHandler:
    def __init__(self, data, tokenizer, max_len=4096, mode='packing', batch_size=4, 
                 system_prompt=None, case_study=True, custom_convert_fn=None,
                 custom_ua_format_fn=None):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mode = mode
        assert mode in ['padding', 'packing'], "Mode must be 'padding' or 'packing'"
        self.batch_size = batch_size
        self.remake_num = 0

        if custom_convert_fn:
            self.data = lmap(lambda x:custom_convert_fn(x, tokenizer), data)
            firstitem = self.data[0]
            if not isinstance(firstitem, dict) or \
                    len(firstitem.get("input_ids", '0')) != len(firstitem.get("labels", '01')):
                print(firstitem)
                raise ValueError("custom_convert_fn should return {'input_ids': [...], 'labels': [...]} for each item")
        elif custom_ua_format_fn:
            iodata = lmap(convert_to_io_format, data)
            uadata = lmap(custom_ua_format_fn, iodata)
            firstitem = uadata[0]
            if not isinstance(firstitem, dict) or not isinstance(firstitem.get("input"), str) or \
                    not isinstance(firstitem.get("output"), str):
                print(firstitem)
                raise ValueError("custom_ua_format_fn should return {'input': str, 'output': str} for each item")
            self.data = lmap(lambda x:convert_to_token_ids(x, tokenizer), uadata)
        else:
            iodata = lmap(convert_to_io_format, data)
            uadata = lmap(lambda x:convert_to_ua_format(x, system_prompt=system_prompt), iodata)
            self.data = lmap(lambda x:convert_to_token_ids(x, tokenizer), uadata)
            
        if case_study:
            batches = convert_to_batches(self.data, self.tokenizer, self.max_len, self.mode, self.batch_size, test=True)
            self.display_case_study(batches[0])

    def display_case_study(self, batch):
        input_ids, labels = batch['input_ids'], batch['labels']
        input_texts = lmap(lambda x:self.tokenizer.decode(x, skip_special_tokens=False), input_ids)
        dot = self.tokenizer.encode('.')[0]
        output_texts = lmap(lambda x:self.tokenizer.decode([z if z > 0 else dot for z in x], 
                                                            skip_special_tokens=False), labels)
        print('\nTrainging Batch Case Study:')
        print('##input_ids:', input_ids.shape)
        for t in input_texts: print(t.replace('\n', '\\n'))
        print('-'*30 + '\n##labels:', labels.shape)
        for t in output_texts: print(t.replace('\n', '\\n'))

    def get_dataloader(self, shuffle=True):
        if shuffle: 
            random.seed(self.remake_num)
            random.shuffle(self.data)
        self.remake_num += 1
        batches = convert_to_batches(self.data, self.tokenizer, self.max_len, self.mode, self.batch_size)
        return make_dataloader(batches)

if __name__ == "__main__":
    def my_convert_fn(item, tokenizer):
        tokenize = lambda x: tokenizer.encode(x, add_special_tokens=False)
        if len(item) > 4: return None
        inp = tokenize('<BOS>System: I am agent X\n')
        out = [-100] * len(inp)
        for z in item:
            zq = tokenize('Q:' + z["instruction"] + '\nA:')
            za = tokenize(z["output"] + '\n')
            inp.extend(zq)
            out.extend([-100] * len(zq))
            inp.extend(za)
            out.extend(za)
        inp.extend(tokenize('<EOS>'))
        out.extend(tokenize('<EOS>'))
        return {
            "input_ids": inp,
            "labels": out
        }