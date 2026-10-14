


import torch
import torch.nn as nn
from torch.nn import functional as f

from configuration import GPT2Config
from transformer import GPT2Model


class GPT2LMHeadModel(nn.Module):

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config
        # transformer block
        self.transformer = GPT2Model(config)
        # LM head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.lm_head.weight = self.transformer.wte.weight

    def forward(self, input):

        output = self.transformer(input)
        logits = self.lm_head(output)
        return logits

    @torch.no_grad()
    def generate_text(self, idx, max_tokens=100):
        self.eval()

        for _ in range(max_tokens):

            block_size = self.config.block_size
            idx_cond = idx[:, -block_size:]

            # forward pass
            logits = self.forward(idx_cond) # (B, T, Vocab_size)

            # take from last prediction
            logits = logits[:, -1, :] # (B, Vocab_size)
            probs = f.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1) # (B, 1)

            # add to existing sequence
            idx = torch.concat([idx, next_token], dim=1)

            # Break in case we encounter EOS token
            if (
                    idx.size(0) == 1
                    and self.config.eos_token_id is not None
                    and next_token.item() == self.config.eos_token_id
            ):
                break
        return idx