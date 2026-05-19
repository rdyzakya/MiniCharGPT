from torch.utils.data import Dataset
import re
from tqdm import tqdm

def remove_unk(text):
    return re.sub(r"[^a-z\s]", "", text).strip()

class CharDS(Dataset):
    def __init__(self, path):
        with open(path, 'r', encoding="UTF-8") as fp:
            data = fp.read().strip().splitlines()
        data = [remove_unk(text.lower()) for text in data]
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        return self.data[i]
    
    def tokenize(self, tokenizer):
        self.data = [tokenizer.tokenize(text) for text in tqdm(self.data)]