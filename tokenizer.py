import constant

class CharTokenizer:
    def __init__(self):
        self.pad_token = constant.pad_token
        self.end_token = constant.end_token
        self.char2id = {c : i for i, c in enumerate(constant.all_char)}
        self.char2id[self.pad_token] = len(self.char2id)
        self.char2id[self.end_token] = len(self.char2id)
        self.id2char = {i : c for c, i in self.char2id.items()}
    
    def encode(self, text, truncate=False, padding=False, max_length=128):
        result = [self.char2id[c] for c in text]
        result = [self.char2id[self.pad_token] for i in range(max_length - len(text))] + result if padding else result
        result = result[:max_length] if truncate else result
        return result
    
    def decode(self, input_ids, remove_special=False):
        result = [self.id2char[i] for i in input_ids]
        if remove_special:
            result = [el for el in result if el != self.pad_token and el != self.end_token]
        return result
    
    def __call__(self, text, **kwargs):
        return self.encode(text, **kwargs)