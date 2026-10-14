

from dataclasses import dataclass

@dataclass
class GPT2Config:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layers: int = 24
    n_heads: int = 16
    n_embd: int = 1024  # embedding dimensions
    dropout: float = 0.1
    bias: bool = True
    eps: float = 1e-5
    eos_token_id: int | None = None