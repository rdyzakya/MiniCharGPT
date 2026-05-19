import torch
import torch.nn.functional as F
from string import ascii_lowercase

ALL_CHAR = ascii_lowercase + ' '
PAD_TOKEN = "<PAD>"

class CharTokenizer:
    def __init__(self):
        self.char2id = {c : i for i, c in enumerate(ALL_CHAR)}
        self.char2id[PAD_TOKEN] = len(self.char2id)
        self.id2char = {i : c for c, i in self.char2id.items()}
        self.n_vocab = len(self.char2id)
    
    def tokenize(self, text):
        input_ids = [self.char2id[c] for c in text]
        return input_ids        

    def encode(self, text):
        input_ids = self.tokenize(text)
        input_ids = torch.tensor(input_ids)

        attention_mask = torch.ones_like(input_ids)

        return {
            "input_ids" : input_ids,
            "attention_mask" : attention_mask
        }
    
    def batch_encode(self, texts, truncate=False, padding='longest', max_length=None):
        assert padding in ['longest', 'max_length'], "padding must be either 'longest' or 'max_length'"
        result = [self.encode(text) for text in texts]
        longest_length = max([len(el["input_ids"]) for el in result])
        if max_length is None:
            max_length = longest_length
        if truncate:
            result = [{
                "input_ids" : el["input_ids"][:max_length],
                "attention_mask" : el["attention_mask"][:max_length]
            } for el in result]
        max_length = max_length if padding == 'max_length' else longest_length
        result = [{
            "input_ids" : F.pad(el["input_ids"], (0, max_length - len(el["input_ids"])), value=self.char2id[PAD_TOKEN]),
            "attention_mask" : F.pad(el["attention_mask"], (0, max_length - len(el["attention_mask"])), value=0)
        } for el in result]
        return {
            "input_ids" : torch.vstack([el["input_ids"] for el in result]),
            "attention_mask" : torch.vstack([el["attention_mask"] for el in result])
        }
    
    def decode(self, input_ids, remove_special=False):
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.tolist()
        text = "".join([self.id2char[i] for i in input_ids])
        if remove_special:
            text = text.replace(PAD_TOKEN, "")
        return text
    
    def batch_decode(self, batch_input_ids, remove_special=False):
        if isinstance(batch_input_ids, torch.Tensor):
            batch_input_ids = batch_input_ids.tolist()
        return [self.decode(input_ids, remove_special=remove_special) for input_ids in batch_input_ids]
    
    def __call__(self, texts, **kwargs):
        if isinstance(texts, str):
            return self.encode(texts, **kwargs)
        return self.batch_encode(texts, **kwargs)
    
    def collate(self, tokenized, truncate=False, padding='longest', max_length=None):
        num_batch = len(tokenized)
        max_len_tokenized = max([len(seq) for seq in tokenized])
        if truncate and max_length:
            max_len_tokenized = min(max_length, max_len_tokenized)
        
        if padding == "max_length" and max_length:
            seq_len = max_length
        else:
            seq_len = max_len_tokenized

        input_ids = torch.full((num_batch, seq_len), -100, dtype=torch.int32)
        for i in range(num_batch):
            seq_len = len(tokenized[i])
            input_ids[i][:seq_len] = torch.tensor(tokenized[i], dtype=torch.int32)
        
        attention_mask = (input_ids != -100).int()

        return {
            "input_ids" : input_ids,
            "attention_mask" : attention_mask
        }