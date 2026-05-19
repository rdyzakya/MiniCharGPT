import torch
import torch.nn.functional as F

# use the original positional encoding from "Attention is All You Need"
class PositionalEncoding(torch.nn.Module):
    def __init__(self):
        super(PositionalEncoding, self).__init__()
    
    def forward(self, x):
        num_batch, seq_len, num_dim = x.shape
        pos = torch.arange(0, seq_len) # shape: (seq_len,)

        power_term = torch.arange(0, num_dim, 2)/num_dim # 2i/d_model
        denominator_term = torch.pow(10000, power_term) # 10000**2i/d_model ; shape: (num_dim//2, )

        denominator_term = denominator_term.repeat((2,1)) # shape: (2, num_dim//2)
        denominator_term = denominator_term.transpose(0,1) # shape: (num_dim//2, 2)
        denominator_term = denominator_term.flatten() # shape: (num_dim//2 * 2)

        if num_dim % 2 == 1: # odd
            denominator_term = denominator_term[:-1]

        division_term = pos.view(-1,1)/denominator_term.repeat(seq_len, 1) # shape: (seq_len, num_dim)

        # for even use sin, for odd use cos
        dimension_index = torch.arange(0, num_dim) # (num_dim,)
        is_even = dimension_index%2 == 0
        is_odd = ~is_even

        positional_encoding = division_term

        positional_encoding[:,is_even] = torch.sin(positional_encoding[:,is_even])
        positional_encoding[:,is_odd] = torch.cos(positional_encoding[:,is_odd])

        out = positional_encoding.repeat(num_batch, 1, 1)
        return out # shape: (num_batch, seq_len, num_dim)

class MaskedAttention(torch.nn.Module):
    def __init__(self, dim_head=64):
        super(MaskedAttention, self).__init__()
        self.wq = torch.nn.Linear(in_features=dim_head, out_features=dim_head, bias=False)
        self.wk = torch.nn.Linear(in_features=dim_head, out_features=dim_head, bias=False)
        self.wv = torch.nn.Linear(in_features=dim_head, out_features=dim_head, bias=False)
        self.h_dim = dim_head
    
    def forward(self, x, attention_mask):
        # x.shape = (num_batch, seq_len, num_dim)
        q = self.wq(x) # shape:(num_batch, seq_len, h_dim)
        k = self.wk(x)
        v = self.wv(x)
        
        qk_d = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.h_dim).float()) # shape: (num_batch, seq_len, seq_len)
        
        # masked attention
        a = torch.arange(qk_d.shape[-1]).expand(x.shape[0], qk_d.shape[-1], -1)
        b = torch.arange(qk_d.shape[-1]).expand(x.shape[0], -1).unsqueeze(-1)
        c = (a > b)
        d = attention_mask.repeat(1,attention_mask.shape[-1]).bool()
        d = d.view(-1, attention_mask.shape[-1], attention_mask.shape[-1])
        e = d.transpose(-1,-2)

        mask = torch.tensor(-torch.inf)
        condition = c.logical_or(
            d.logical_and(e).logical_not()
        )

        qk_d = qk_d.masked_fill(condition, mask)
        att_score = qk_d.softmax(-1)
        att_score = att_score.masked_fill(torch.isnan(att_score), 0.0)
        out = torch.matmul(att_score, v) # shape: (num_batch, seq_len, num_dim)
        return out, att_score

class MaskedMultiHeadAttention(torch.nn.Module):
    def __init__(self, dim_model=512, n_head=8):
        super(MaskedMultiHeadAttention, self).__init__()
        dim_head = dim_model//n_head
        self.att = torch.nn.ModuleList([MaskedAttention(dim_head=dim_head) for _ in range(n_head)])
        self.wo = torch.nn.Linear(in_features=n_head * dim_head, out_features=dim_model, bias=False)
    
    def forward(self, x, attention_mask):
        out_head_list = []
        att_score_list = []
        for att_head in self.att:
            out_head, att_score = att_head(x, attention_mask)
            out_head_list.append(out_head)
            att_score_list.append(att_score)

        concatenated_out = torch.cat(out_head_list, dim=-1)
        out = self.wo(concatenated_out)
        return out, att_score_list

class Decoder(torch.nn.Module):
    def __init__(self, dim_model=512, n_head=8, dim_ff=2048):
        super(Decoder, self).__init__()
        self.mha = MaskedMultiHeadAttention(dim_model=dim_model, n_head=n_head)
        self.layernorm1 = torch.nn.LayerNorm(dim_model)
        self.ff1 = torch.nn.Linear(in_features=dim_model, out_features=dim_ff, bias=True)
        self.ff2 = torch.nn.Linear(in_features=dim_ff, out_features=dim_model, bias=True)
        self.layernorm2 = torch.nn.LayerNorm(dim_model)
    
    def forward(self, x, attention_mask):
        out_multihead, att_scores = self.mha(x, attention_mask)
        add_norm_output1 = self.layernorm1(x + out_multihead) # shape: (num_batch, seq_len, num_dim)

        ff1_out = self.ff1(add_norm_output1) # shape: (num_batch, seq_len, dim_ff)
        ff2_out = self.ff2(F.relu(ff1_out)) # shape: (num_batch, seq_len, num_dim)
        out = self.layernorm2(add_norm_output1 + ff2_out)
        return out, att_scores

class GPT(torch.nn.Module):
    def __init__(self, dim_model=512, n_head=8, dim_ff=2048, n_block=3, n_vocab=28):
        super(GPT, self).__init__()
        self.embedding = torch.nn.Embedding(num_embeddings=n_vocab, embedding_dim=dim_model)
        self.pe = PositionalEncoding()
        self.decoders = torch.nn.ModuleList(
            [Decoder(dim_model=dim_model, n_head=n_head, dim_ff=dim_ff) for _ in range(n_block)]
        )
        self.lm_head = torch.nn.Linear(in_features=dim_model, out_features=n_vocab, bias=False)

        # tie weights
        self.lm_head.weight = self.embedding.weight
    
    def forward(self, input_ids, attention_mask=None):
        x = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(x)
        x = self.embedding(x)
        x = x + self.pe(x)
        for dec in self.decoders:
            x = dec(x, attention_mask)
        x = self.lm_head(x) # shape: (num_batch, seq_len, n_vocab)
        return x