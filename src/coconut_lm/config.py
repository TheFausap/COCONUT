from dataclasses import dataclass


@dataclass(slots=True)
class CoconutConfig:
    vocab_size: int
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1
    pad_token_id: int = 0
    latent_token_id: int | None = None

    def __post_init__(self) -> None:
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        if self.block_size < 2:
            raise ValueError("block_size must be at least 2")

