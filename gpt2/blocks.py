

import torch
import torch.nn as nn
from torch.nn import functional as f
from transformers.pytorch_utils import Conv1D
from transformers.activations import NewGELUActivation
from configuration import GPT2Config



class GPT2MLP(nn.Module):

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config

        # expand by 4 times
        self.c_fc = Conv1D(4*config.n_embd, config.n_embd)

        # shrink by 4 times

        self.c_proj = Conv1D(config.n_embd, 4*config.n_embd)

        self.act = NewGELUActivation()

        self.dropout = nn.Dropout(p=config.dropout)

    def forward(self, x):

        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class GPT2Attention(nn.Module):

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config

        # make sure embeddings can be divided among heads equally
        assert config.n_embd % config.n_heads == 0

        self.n_head = config.n_heads
        self.head_dim = config.n_embd // config.n_heads

        # Q,K,V projection
        # params : (output_f, input_f) there_4 weight Shape : (n_embd, 3*n_embd)
        self.c_attn = Conv1D( 3 * config.n_embd, config.n_embd)
        self.c_proj = Conv1D(config.n_embd, config.n_embd)

        self.attn_dropout = nn.Dropout(self.config.dropout)
        self.resid_dropout = nn.Dropout(self.config.dropout)


    def forward(self, x):
        # x : (B, T, C) => C = n_embd
        B, T, C = x.shape

        qkv = self.c_attn(x)    # (B, T, 3*c)

        q, k, v = qkv.split(C, dim=2) # Each of them is not (B, T, C)

        # Split into multiple heads
        q = q.view(B, T, self.n_head, self.head_dim)
        k = k.view(B, T, self.n_head, self.head_dim)
        v = v.view(B, T, self.n_head, self.head_dim)
        # Shapes are (B, T, n_head, head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        # Now shapes became (B, n_head, T, head_dim)
        # this creates parallel B, n_head so that attention can be calculated on Tokens

        # K_t shape (B, n_head, head_dim, T)
        # that make (B, n_head, T, head_dim) @ (B, n_head, head_dim, T)
        # this will retrieve values with dedicated attention across tokens
        attn = q@k.transpose(-2, -1)    # (B, n_head, T, T)
        # Scale the attention
        attn = attn * (self.head_dim ** -0.5)

        # Apply the mask
        tril = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        attn = attn.masked_fill(tril==0, float('-inf'))

        attn = torch.softmax(attn, dim=-1)

        # apply dropout
        attn = self.attn_dropout(attn)

        # (B, n_head, T, T) @ (B, n_head, T, head_dim)
        out = attn@v    # (B, n_head, T, head_dim)

        # now rearrange and concat data from all the deads
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        out = self.c_proj(out)
        # Final dropout, just to match the architecture of gpt2
        out = self.resid_dropout(out)

        return out














class GPT2Block(nn.Module):
    """
        GPT2 Block :
            - ln_1
            - attn
            - ln_2
            - mlp
    """
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config

        self.ln_1 = nn.LayerNorm(self.config.n_embd, eps=self.config.eps)
        self.attn = GPT2Attention(config)
        self.ln_2 = nn.LayerNorm(self.config.n_embd, eps=self.config.eps)
        self.mlp = GPT2MLP(config)


    def forward(self, x):
        # idx = (B, T, n_embd)

        # self attention with residue
        x = x + self.attn(self.ln_1(x))

        # mlp with residue
        x = x + self.mlp(self.ln_2(x))

        return x


