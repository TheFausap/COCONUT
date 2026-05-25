from __future__ import annotations

from dataclasses import dataclass


SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<latent>"]


@dataclass(slots=True)
class CharTokenizer:
    stoi: dict[str, int]
    itos: list[str]

    @classmethod
    def build(cls, text: str, extra_tokens: list[str] | None = None) -> "CharTokenizer":
        tokens = list(SPECIAL_TOKENS)
        if extra_tokens:
            tokens.extend(token for token in extra_tokens if token not in tokens)
        chars = sorted(set(text))
        tokens.extend(char for char in chars if char not in tokens)
        return cls({token: idx for idx, token in enumerate(tokens)}, tokens)

    @classmethod
    def from_stoi(cls, stoi: dict[str, int]) -> "CharTokenizer":
        itos = [""] * len(stoi)
        for token, idx in stoi.items():
            itos[idx] = token
        return cls(stoi, itos)

    @property
    def pad_id(self) -> int:
        return self.stoi["<pad>"]

    @property
    def bos_id(self) -> int:
        return self.stoi["<bos>"]

    @property
    def eos_id(self) -> int:
        return self.stoi["<eos>"]

    @property
    def latent_id(self) -> int:
        return self.stoi["<latent>"]

    def encode(self, text: str, *, bos: bool = False, eos: bool = False) -> list[int]:
        ids: list[int] = []
        idx = 0
        while idx < len(text):
            special = next((token for token in SPECIAL_TOKENS if text.startswith(token, idx)), None)
            if special is not None:
                ids.append(self.stoi[special])
                idx += len(special)
                continue
            char = text[idx]
            if char not in self.stoi:
                raise ValueError(f"character {char!r} is not in the tokenizer vocabulary") from None
            ids.append(self.stoi[char])
            idx += 1
        if bos:
            ids.insert(0, self.bos_id)
        if eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int] | tuple[int, ...], *, skip_special: bool = True) -> str:
        pieces: list[str] = []
        for idx in ids:
            token = self.itos[int(idx)]
            if skip_special and token in SPECIAL_TOKENS:
                continue
            pieces.append(token)
        return "".join(pieces)
