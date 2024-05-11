import torch
import torch.nn.functional as F

class EmbeddingMatrix(torch.nn.Module):
    def __init__(self, n_token=20, dim=256, seed=42):
        super(EmbeddingMatrix, self).__init__()
        torch.manual_seed(seed)
        self.embedding_matrix = torch.nn.parameter.Parameter(torch.rand((n_token, dim)))
    
    def forward(self, x):
        x = torch.tensor(x) if not isinstance(x, torch.Tensor) else x
        x = self.embedding_matrix[x]
        return x

class PositionalEncoder(torch.nn.Module):
    def __init__(self):
        super(PositionalEncoder, self).__init__()
        self.device = torch.device("cpu")
    
    def positional_encoding(self, attention_mask, d_model):
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))
        max_length = (attention_mask != 0).sum()
        
        pos_ids = torch.arange(0, max_length)
        pos_ids = pos_ids.unsqueeze(1).float()

        pos_enc = torch.zeros((attention_mask.shape[0], d_model))

        pos_enc[attention_mask != 0,0::2] = torch.sin(pos_ids * div_term)
        pos_enc[attention_mask != 0,1::2] = torch.sin(pos_ids * div_term)
        
        return pos_enc

    def forward(self, x, attention_mask):
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        
        _, _, embedding_dim = x.shape

        pos_enc = torch.vstack(list(map(lambda x: self.positional_encoding(x, embedding_dim).unsqueeze(0), attention_mask)))

        return x + pos_enc.to(self.device)
    
    
    def to(self, device):
        self.device = device
        return super().to(device)
    
class MaskedAttentionBlock(torch.nn.Module):
    def __init__(self, h_dim=256): # half of the original, 8
        super(MaskedAttentionBlock, self).__init__()
        self.wq = torch.nn.Linear(in_features=h_dim, out_features=h_dim, bias=False)
        self.wk = torch.nn.Linear(in_features=h_dim, out_features=h_dim, bias=False)
        self.wv = torch.nn.Linear(in_features=h_dim, out_features=h_dim, bias=False)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.h_dim = h_dim
        self.device = torch.device("cpu")
    
    def forward(self, x, attention_mask):
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)
        
        qk_d = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.h_dim).float())
        # mask
        a = torch.arange(qk_d.shape[-1]).expand(x.shape[0], qk_d.shape[-1], -1)
        b = torch.arange(qk_d.shape[-1]).expand(x.shape[0], -1).unsqueeze(-1)
        c = (a > b).to(self.device)
        d = attention_mask.repeat(1,attention_mask.shape[-1]).bool().to(self.device)
        d = d.view(-1, attention_mask.shape[-1], attention_mask.shape[-1])
        e = d.transpose(-1,-2)

        mask = torch.tensor(-torch.inf).to(self.device)
        condition = c.logical_or(
            d.logical_and(e).logical_not()
        )

        qk_d = qk_d.masked_fill(condition, mask)
        att_score = self.softmax(qk_d)
        att_score = att_score.masked_fill(torch.isnan(att_score), 0.0)
        return torch.matmul(att_score, v)
    
    def to(self, device=torch.device("cpu")):
        self.device = device
        return super().to(device)
    
    def process_attention_maks_per_instance(self, attention_mask):
        a = attention_mask.repeat(attention_mask.shape[0],1).bool()
        b = a.transpose(-1,-2)

        return torch.logical_and(a,b).logical_not()

class DecoderBlock(torch.nn.Module):
    def __init__(self, n_head=4, h_dim=256, ff_dim=1024):
        super(DecoderBlock, self).__init__()
        self.att = torch.nn.ModuleList([MaskedAttentionBlock(h_dim=h_dim) for _ in range(n_head)])
        self.wo = torch.nn.Linear(in_features=n_head * h_dim, out_features=h_dim)
        self.layernorm1 = torch.nn.LayerNorm(h_dim)
        self.ff1 = torch.nn.Conv1d(h_dim, ff_dim, 1, stride=1)
        self.ff2 = torch.nn.Conv1d(ff_dim, h_dim, 1, stride=1)
        self.layernorm2 = torch.nn.LayerNorm(h_dim)
        self.device = torch.device("cpu")
    
    def forward(self, x, attention_mask):
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        att_outputs = torch.cat([att(x, attention_mask) for att in self.att], dim=-1)
        multihead_output = self.wo(att_outputs)
        add_norm_output1 = self.layernorm1(x + multihead_output)
        ff_output = self.ff2(F.relu(self.ff1(add_norm_output1.transpose(-1,-2))))
        ff_output = ff_output.transpose(-1,-2)
        out = self.layernorm2(add_norm_output1 + ff_output)
        return out
    
    def to(self, device=torch.device("cpu")):
        self.att = torch.nn.ModuleList([att.to(device) for att in self.att])
        return super().to(device)
    
class MaskedSequential(torch.nn.Sequential):
    def forward(self, x, attention_mask):
        for module in self._modules.values():
            x = module(x, attention_mask=attention_mask)
        return x

class MiniCharGPTLM(torch.nn.Module):
    def __init__(self, h_dim=512, ff_dim=1024, n_head=4, n_block=3, n_token=28):
        super(MiniCharGPTLM, self).__init__()
        self.embedding_matrix = EmbeddingMatrix(n_token=n_token, dim=h_dim)
        self.positional_encoder = PositionalEncoder()
        self.decoders = MaskedSequential(
            *[DecoderBlock(n_head=n_head, h_dim=h_dim, ff_dim=ff_dim) for _ in range(n_block)]
        )
        self.lm_head = torch.nn.Linear(in_features=h_dim, out_features=n_token, bias=False)
    
    def forward(self, input_ids, attention_mask=None):
        x = input_ids
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        if attention_mask is None:
            attention_mask = torch.ones_like(x)
        x = self.embedding_matrix(x)
        x = self.positional_encoder(x, attention_mask)
        x = self.decoders(x, attention_mask)
        x = self.lm_head(x)
        return x
    
    def to(self, device=torch.device("cpu")):
        self.positional_encoder = self.positional_encoder.to(device)
        self.decoders = MaskedSequential(
                    *[decoder.to(device) for decoder in self.decoders]
                )
        return super().to(device)