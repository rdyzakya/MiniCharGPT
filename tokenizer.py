import constant
import torch

class CharTokenizer:
    def __init__(self):
        self.pad_token = constant.pad_token
        self.end_token = constant.end_token
        self.char2id = {c : i for i, c in enumerate(constant.all_char)}
        self.char2id[self.pad_token] = len(self.char2id)
        self.char2id[self.end_token] = len(self.char2id)
        self.id2char = {i : c for c, i in self.char2id.items()}
    
    def encode(self, text, truncate=False, padding=False, max_length=128):
        input_ids = [self.char2id[c] for c in text]
        input_ids = [self.char2id[self.pad_token] for i in range(max_length - (len(text) + 1))] + input_ids if padding else input_ids
        input_ids.append(self.char2id[self.end_token])
        input_ids = torch.tensor(input_ids)

        attention_mask = torch.ones_like(input_ids)
        attention_mask[:max(0,max_length - (len(text) + 1))] = 0

        input_ids = input_ids[:max_length] if truncate else input_ids
        attention_mask = attention_mask[:max_length] if truncate else attention_mask

        return {
            "input_ids" : input_ids,
            "attention_mask" : attention_mask
        }
    
    def batch_encode(self, texts, truncate=False, padding=False, max_length=128):
        result = [self.encode(text, truncate=truncate, padding=padding, max_length=max_length) for text in texts]
        result = {
            "input_ids" : torch.vstack([el["input_ids"] for el in result]),
            "attention_mask" : torch.vstack([el["attention_mask"] for el in result])
        }
        return result
    
    def decode(self, input_ids, remove_special=False):
        result = [self.id2char[i] for i in input_ids]
        if remove_special:
            result = [el for el in result if el != self.pad_token and el != self.end_token]
        return result
    
    def __call__(self, texts, **kwargs):
        if isinstance(texts, str):
            return self.encode(texts, **kwargs)
        return self.batch_encode(texts, **kwargs)