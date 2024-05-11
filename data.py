from torch.utils.data import Dataset
from tokenizer import CharTokenizer
import torch
import re
import constant

def remove_unk(text):
    return re.sub("[^a-z\s]", "", text).strip()

def expand(text):
    text = text + constant.end_token
    return [(text[:i], text[i]) for i in range(1, len(text))]

class CharDS(Dataset):
    def __init__(self, data, tokenizer=None, tokenizer_kwargs={}):
        self.data = data
        self.tokenizer = CharTokenizer() if tokenizer == None else tokenizer
        self.tokenizer_kwargs = tokenizer_kwargs
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        return self.tokenizer(self.data[i], **self.tokenizer_kwargs)
    
    def load_data(path, tokenizer=None, tokenizer_kwargs={}):
        with open(path, 'r', encoding="UTF-8") as fp:
            data = fp.read().strip().splitlines()
        data = [remove_unk(text.lower()) for text in data]
        return CharDS(data, tokenizer, tokenizer_kwargs)

class LanguageModelingDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, inputs):
        labels = torch.vstack([el["input_ids"] for el in inputs])

        input_ids = self.right_shift(labels)

        attention_mask = torch.vstack([el["attention_mask"] for el in inputs])

        labels = labels.masked_fill(attention_mask.bool().logical_not(), -100)

        return {
            "input_ids" : input_ids,
            "attention_mask" : attention_mask,
            "labels" : labels
        }

    def right_shift(self, input_tensor):
        shifted_tensor = torch.roll(input_tensor, shifts=1, dims=-1)

        if len(shifted_tensor.shape) == 1:
            shifted_tensor[0] = self.tokenizer.char2id[self.tokenizer.pad_token]
        elif len(shifted_tensor.shape) == 2:
            shifted_tensor[:,0] = self.tokenizer.char2id[self.tokenizer.pad_token]
        else:
            raise NotImplementedError("Not imlpemented for tensor more than 2 dimension")

        return shifted_tensor