

import torch
import torch.nn as nn
from torch.nn import functional as f


from blocks import GPT2Block
from configuration import GPT2Config


class GPT2Model(nn.Module):

    def __init__(self, config: GPT2Config):
        super().__init__()

        # word token embedding
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        # word positon embedding
        self.wpe = nn.Embedding(config.block_size, config.n_embd)

        # dropout
        self.drop = nn.Dropout(config.dropout)

        # GPT2Blocks
        self.h = nn.ModuleList([ GPT2Block(config) for _ in range(config.n_layers) ])

        # Normalization
        self.ln_f = nn.LayerNorm(config.n_embd)

    def forward(self, idx):
        # idx : (B, T)
        B, T = idx.shape
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # (T, )

        tok_embd = self.wte(idx) # (B, T) embed with n_embd -> (B, T, n_embd)
        pos_embd = self.wpe(pos) # (T,) embeed with n_embd -> (T, n_embd)

        x = tok_embd + pos_embd # (B, T, n_embd) + (T, n_embd) , 🔮 (BROADCASTING)
        x = self.drop(x)

        for head in self.h:
            x = head(x)

        # now pass through normalization
        x = self.ln_f(x)

        return x


