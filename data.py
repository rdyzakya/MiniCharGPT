from torch.utils.data import Dataset
from tokenizer import CharTokenizer
import re
import constant

def remove_unk(text):
    return re.sub("[^a-z\s]", "", text)

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
        tokens = self.tokenizer(self.data[i][0], **self.tokenizer_kwargs)
        labels = self.tokenizer.char2id[self.data[i][1]]
        return tokens, labels
    
    def load_data(path, tokenizer=None, tokenizer_kwargs={}):
        with open(path, 'r') as fp:
            data = fp.read().strip().splitlines()
        data = [remove_unk(text.lower()) for text in data]
        result = []
        for text in data:
            result.extend(expand(text))
        return CharDS(result, tokenizer, tokenizer_kwargs)