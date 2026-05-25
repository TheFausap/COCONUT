from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from coconut_lm.config import CoconutConfig


@dataclass(slots=True)
class CoconutOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None
    latent_embeddings: tuple[torch.Tensor, ...] = ()


class CausalSelfAttention(nn.Module):
    def __init__(self, config: CoconutConfig) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        mask = torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool))
        self.register_buffer("causal_mask", mask.view(1, 1, config.block_size, config.block_size))

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, seq_len, channels = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (self.head_dim**-0.5)
        att = att.masked_fill(~self.causal_mask[:, :, :seq_len, :seq_len], float("-inf"))
        if attention_mask is not None:
            key_mask = attention_mask[:, None, None, :seq_len].to(dtype=torch.bool)
            att = att.masked_fill(~key_mask, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, channels)
        return self.resid_dropout(self.proj(y))


class MLP(nn.Module):
    def __init__(self, config: CoconutConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DecoderBlock(nn.Module):
    def __init__(self, config: CoconutConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), attention_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class TinyCoconutLM(nn.Module):
    """A tiny GPT-style decoder with COCONUT continuous latent reasoning.

    Latent positions are marked by a special token id. Their discrete token
    embedding is replaced by the previous transformer hidden state before the
    final language-modeling pass.
    """

    def __init__(self, config: CoconutConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(DecoderBlock(config) for _ in range(config.n_layer))
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _run_transformer(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _, seq_len, _ = inputs_embeds.shape
        if seq_len > self.config.block_size:
            raise ValueError(f"sequence length {seq_len} exceeds block size {self.config.block_size}")

        positions = torch.arange(seq_len, device=inputs_embeds.device)
        x = inputs_embeds + self.position_embedding(positions)[None, :, :]
        x = self.drop(x)
        for block in self.blocks:
            x = block(x, attention_mask)
        return self.ln_f(x)

    def _latent_mask_from_ids(self, input_ids: torch.Tensor) -> torch.Tensor | None:
        if self.config.latent_token_id is None:
            return None
        return input_ids.eq(self.config.latent_token_id)

    def _inject_continuous_thoughts(
        self,
        inputs_embeds: torch.Tensor,
        latent_mask: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        inputs = inputs_embeds
        latent_states: list[torch.Tensor] = []
        positions = latent_mask.any(dim=0).nonzero(as_tuple=False).flatten().tolist()

        for pos in positions:
            if pos == 0:
                continue
            prefix_mask = attention_mask[:, :pos] if attention_mask is not None else None
            prefix_hidden = self._run_transformer(inputs[:, :pos, :], prefix_mask)
            continuous_thought = prefix_hidden[:, -1, :]
            row_mask = latent_mask[:, pos].view(-1, 1)
            replacement = torch.where(row_mask, continuous_thought, inputs[:, pos, :])
            inputs = torch.cat(
                [inputs[:, :pos, :], replacement[:, None, :], inputs[:, pos + 1 :, :]],
                dim=1,
            )
            latent_states.append(continuous_thought)

        return inputs, tuple(latent_states)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        latent_mask: torch.Tensor | None = None,
        return_latents: bool = False,
    ) -> CoconutOutput:
        if attention_mask is None:
            attention_mask = input_ids.ne(self.config.pad_token_id)
        if latent_mask is None:
            latent_mask = self._latent_mask_from_ids(input_ids)

        inputs_embeds = self.token_embedding(input_ids)
        latent_embeddings: tuple[torch.Tensor, ...] = ()
        if latent_mask is not None and bool(latent_mask.any()):
            inputs_embeds, latent_embeddings = self._inject_continuous_thoughts(
                inputs_embeds,
                latent_mask,
                attention_mask,
            )

        hidden = self._run_transformer(inputs_embeds, attention_mask)
        logits = self.lm_head(hidden)
        loss = self._loss(logits, targets, latent_mask)
        if not return_latents:
            latent_embeddings = ()
        return CoconutOutput(logits=logits, loss=loss, latent_embeddings=latent_embeddings)

    def _loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor | None,
        latent_mask: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if targets is None:
            return None
        labels = targets.clone()
        labels[labels == self.config.pad_token_id] = -100
        if latent_mask is not None:
            labels[latent_mask] = -100
        return F.cross_entropy(
            logits[:, :-1, :].contiguous().view(-1, logits.size(-1)),
            labels[:, 1:].contiguous().view(-1),
            ignore_index=-100,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        eos_token_id: int | None = None,
        temperature: float = 1.0,
        top_k: int | None = None,
        banned_token_ids: set[int] | None = None,
    ) -> torch.Tensor:
        self.eval()
        banned_token_ids = banned_token_ids or set()
        for _ in range(max_new_tokens):
            context = input_ids[:, -self.config.block_size :]
            logits = self(context).logits[:, -1, :]
            if banned_token_ids:
                logits[:, list(banned_token_ids)] = float("-inf")
            if temperature <= 0:
                next_id = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits = logits.masked_fill(logits < values[:, [-1]], float("-inf"))
                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_id], dim=1)
            if eos_token_id is not None and bool((next_id == eos_token_id).all()):
                break
        return input_ids
