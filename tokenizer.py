"""
tokenizer.py - Tokenizer Utilities for DyadicShelf LM
=====================================================
Provides unified tokenization support:
1. Fast subword BPE tokenizer with automatic ByteLevel decoding.
2. Pure Python Character/Byte-level tokenizer (zero external dependencies).
"""

import json
import os
from typing import List, Optional


class CharTokenizer:
    """Zero-dependency character-level tokenizer."""
    def __init__(self, chars: Optional[List[str]] = None):
        if chars is not None:
            self.chars = sorted(list(set(chars)))
            self.stoi = {ch: i for i, ch in enumerate(self.chars)}
            self.itos = {i: ch for i, ch in enumerate(self.chars)}
        else:
            self.chars = []
            self.stoi = {}
            self.itos = {}

    @classmethod
    def train_from_text(cls, text: str) -> "CharTokenizer":
        chars = sorted(list(set(text)))
        return cls(chars)

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> List[int]:
        return [self.stoi[ch] for ch in text if ch in self.stoi]

    def decode(self, tokens: List[int]) -> str:
        return "".join([self.itos.get(idx, "") for idx in tokens])

    def to_dict(self) -> dict:
        return {"type": "char", "chars": self.chars}

    @classmethod
    def from_dict(cls, data: dict) -> "CharTokenizer":
        return cls(data["chars"])


class BPETokenizerWrapper:
    """Wrapper around HuggingFace tokenizers BPE implementation."""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        # Ensure ByteLevel decoder is attached so tokens decode to human-readable text
        try:
            from tokenizers import decoders
            if self.tokenizer.decoder is None:
                self.tokenizer.decoder = decoders.ByteLevel()
        except Exception:
            pass

    @classmethod
    def train_from_file(cls, file_path: str, vocab_size: int = 8192, save_path: Optional[str] = None) -> "BPETokenizerWrapper":
        try:
            from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
        except ImportError:
            raise ImportError(
                "The 'tokenizers' library is required for BPE. Install with: pip install tokenizers"
            )

        tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=["<unk>", "<s>", "</s>", "<pad>"],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )
        tokenizer.train([file_path], trainer)
        if save_path:
            tokenizer.save(save_path)
        return cls(tokenizer)

    @classmethod
    def from_file(cls, json_path: str) -> "BPETokenizerWrapper":
        from tokenizers import Tokenizer
        try:
            tokenizer = Tokenizer.from_file(json_path)
        except Exception:
            # Cross-version compatibility for differences in merges serialization
            with open(json_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if "model" in raw and "merges" in raw["model"]:
                merges = raw["model"]["merges"]
                if merges and isinstance(merges[0], list):
                    raw["model"]["merges"] = [" ".join(m) for m in merges]
                elif merges and isinstance(merges[0], str):
                    raw["model"]["merges"] = [m.split(" ", 1) for m in merges]
            tokenizer = Tokenizer.from_str(json.dumps(raw))
        return cls(tokenizer)

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text).ids

    def decode(self, tokens: List[int]) -> str:
        text = self.tokenizer.decode(tokens)
        if "Ġ" in text or "Ċ" in text:
            try:
                from tokenizers import decoders
                self.tokenizer.decoder = decoders.ByteLevel()
                text = self.tokenizer.decode(tokens)
            except Exception:
                pass
            if "Ġ" in text or "Ċ" in text:
                text = (
                    text.replace("Ġ", " ")
                    .replace("Ċ", "\n")
                    .replace("ĉ", "\t")
                    .replace("č", "\r")
                )
        return text

    def to_json_str(self) -> str:
        return self.tokenizer.to_str()

    @classmethod
    def from_json_str(cls, json_str: str) -> "BPETokenizerWrapper":
        from tokenizers import Tokenizer
        try:
            tokenizer = Tokenizer.from_str(json_str)
        except Exception:
            raw = json.loads(json_str)
            if "model" in raw and "merges" in raw["model"]:
                merges = raw["model"]["merges"]
                if merges and isinstance(merges[0], list):
                    raw["model"]["merges"] = [" ".join(m) for m in merges]
                elif merges and isinstance(merges[0], str):
                    raw["model"]["merges"] = [m.split(" ", 1) for m in merges]
            tokenizer = Tokenizer.from_str(json.dumps(raw))
        return cls(tokenizer)


def load_or_create_tokenizer(
    data_path: str,
    tokenizer_type: str = "auto",
    vocab_size: int = 8192,
    save_dir: str = "checkpoints",
):
    """
    Automatically selects, trains, or loads a tokenizer.
    If 'auto', uses BPE if 'tokenizers' is available, else falls back to CharTokenizer.
    """
    os.makedirs(save_dir, exist_ok=True)
    bpe_save_path = os.path.join(save_dir, "tokenizer.json")

    if os.path.exists(bpe_save_path):
        try:
            return BPETokenizerWrapper.from_file(bpe_save_path)
        except Exception:
            pass

    has_tokenizers = False
    try:
        import tokenizers  # noqa: F401
        has_tokenizers = True
    except ImportError:
        has_tokenizers = False

    if (tokenizer_type == "bpe" or tokenizer_type == "auto") and has_tokenizers:
        print(f"[*] Training fast BPE tokenizer (vocab_size={vocab_size}) on {data_path}...")
        tok = BPETokenizerWrapper.train_from_file(
            data_path, vocab_size=vocab_size, save_path=bpe_save_path
        )
        print(f"[*] BPE tokenizer trained successfully. Vocabulary size: {tok.vocab_size}")
        return tok

    # Fallback to CharTokenizer
    print(f"[*] Building zero-dependency character-level tokenizer from {data_path}...")
    with open(data_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    tok = CharTokenizer.train_from_text(text)
    print(f"[*] Character tokenizer built. Unique characters / vocab size: {tok.vocab_size}")
    return tok
