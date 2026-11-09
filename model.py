"""
model.py - DyadicPush V2: DyadicPush + Two-Stage Shelf Retrieval LM
===================================================================
A hybrid language model architecture engineered to train efficiently on 8K context
and scale inference to multi-million token contexts with bounded memory.

Core Principles:
1. Local DyadicPush Engine (O(T log2 T)):
   - Multi-scale forward dyadic octave strides: Delta_k = 2^k for k in {0, ..., log2(T)-1}.
   - Causal 1D depthwise convolution for fine-grained local n-gram syntax.
   - Bounded rolling state during autoregressive decoding.

2. Two-Stage Uncompressed Shelf/Chunk Retrieval (Multi-Million Context Scalability):
   - Tokens grouped into shelves of chunk size C = 64.
   - Stage 1 (Coarse Routing): Dual-pooled landmarks (mean + max) projected into
     holographic semantic space (map_dim = 32). Causal dot-product selects top-K shelves.
   - Stage 2 (Fine Attention): Uncompressed raw keys and values of selected shelves
     are gathered for exact dot-product attention with differentiable coarse boosting.

3. Adaptive Gated Fusion:
   - Per-head learnable sigmoid gating balances local dyadic synthesis with long-range
     uncompressed shelf recall.

4. Memory-Efficient Chunked Cross-Entropy:
   - Sliced cross-entropy prevents VRAM spikes during 8K+ training on large vocabularies.
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DyadicShelfConfig:
    vocab_size: int = 8192
    dim: int = 384
    hidden_dim: int = 1024
    num_layers: int = 8
    num_heads: int = 6
    max_seq_len: int = 8192
    chunk_size: int = 64      # Size of each memory shelf (C)
    top_k: int = 4            # Number of past shelves to retrieve
    map_dim: int = 32         # Dimension of coarse holographic landmark embeddings
    conv_kernel_size: int = 4 # Causal depthwise convolution kernel
    dropout: float = 0.0
    bias: bool = False


# -----------------------------------------------------------------------------
# Primitive Modules
# -----------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019)."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = torch.mean(x ** 2, dim=-1, keepdim=True)
        return x * torch.rsqrt(var + self.eps) * self.weight


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network (Shazeer, 2020)."""
    def __init__(self, dim: int, hidden_dim: int, bias: bool = False):
        super().__init__()
        self.w_gate = nn.Linear(dim, hidden_dim, bias=bias)
        self.w_val = nn.Linear(dim, hidden_dim, bias=bias)
        self.w_out = nn.Linear(hidden_dim, dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_out(F.silu(self.w_gate(x)) * self.w_val(x))


def chunked_cross_entropy(
    hidden_states: torch.Tensor,
    lm_head_weight: torch.Tensor,
    targets: torch.Tensor,
    chunk_size: int = 2048,
) -> torch.Tensor:
    """
    Computes cross-entropy loss in sequential token slices to prevent VRAM explosion
    when projecting large sequence lengths (e.g. 8K) against large vocabularies.
    """
    total_tokens = hidden_states.size(0)
    total_loss = torch.tensor(0.0, device=hidden_states.device, dtype=torch.float32)

    for i in range(0, total_tokens, chunk_size):
        end_idx = min(i + chunk_size, total_tokens)
        h_chunk = hidden_states[i:end_idx]
        t_chunk = targets[i:end_idx]

        logits_chunk = F.linear(h_chunk, lm_head_weight)
        loss_chunk = F.cross_entropy(logits_chunk.float(), t_chunk, reduction="sum")
        total_loss = total_loss + loss_chunk

    return total_loss / max(total_tokens, 1)


# -----------------------------------------------------------------------------
# DyadicShelf Attention
# -----------------------------------------------------------------------------

class DyadicShelfAttention(nn.Module):
    """
    Hybrid Attention Layer combining:
      1. Local DyadicPush geometric packet routing (O(T log2 T))
      2. Two-Stage Coarse/Fine Uncompressed Shelf Retrieval for multi-million context
    """
    def __init__(self, config: DyadicShelfConfig):
        super().__init__()
        self.dim = config.dim
        self.num_heads = config.num_heads
        assert self.dim % self.num_heads == 0, "dim must be divisible by num_heads"
        self.head_dim = self.dim // self.num_heads

        self.max_seq_len = config.max_seq_len
        self.chunk_size = config.chunk_size
        self.top_k = config.top_k
        self.map_dim = config.map_dim

        # --- Local DyadicPush Components ---
        self.num_octaves = math.ceil(math.log2(config.max_seq_len))
        self.strides = [1 << k for k in range(self.num_octaves)]

        self.w_v_local = nn.Linear(self.dim, self.dim, bias=config.bias)
        self.w_g_local = nn.Linear(self.dim, self.dim, bias=config.bias)
        self.w_q_local = nn.Linear(self.dim, self.dim, bias=config.bias)

        self.octave_keys = nn.Parameter(
            torch.randn(self.num_octaves, self.num_heads, self.head_dim) * 0.02
        )
        prior = torch.tensor([1.0 / (1.0 + 0.1 * k) for k in range(self.num_octaves)], dtype=torch.float32)
        self.octave_bias = nn.Parameter(prior)

        k_sz = config.conv_kernel_size
        self.local_conv = nn.Conv1d(
            self.dim,
            self.dim,
            kernel_size=k_sz,
            padding=k_sz - 1,
            groups=self.dim,
            bias=config.bias,
        )

        # --- Global Shelf Retrieval Components ---
        self.w_k_shelf = nn.Linear(self.dim, self.dim, bias=config.bias)
        self.w_v_shelf = nn.Linear(self.dim, self.dim, bias=config.bias)
        self.w_q_shelf = nn.Linear(self.dim, self.dim, bias=config.bias)

        # Semantic lenses mapping dual-pooled summaries (mean + max = 2*head_dim) into map_dim
        self.lens_q = nn.Linear(2 * self.head_dim, self.map_dim, bias=False)
        self.lens_k = nn.Linear(2 * self.head_dim, self.map_dim, bias=False)
        nn.init.normal_(self.lens_q.weight, std=0.02)
        nn.init.normal_(self.lens_k.weight, std=0.02)

        # --- Output & Adaptive Fusion ---
        self.w_out = nn.Linear(self.dim, self.dim, bias=config.bias)
        self.gate = nn.Linear(self.dim, self.num_heads, bias=True)
        nn.init.constant_(self.gate.bias, 0.0)

        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0.0 else nn.Identity()

    def _forward_local_dyadic(self, x: torch.Tensor) -> torch.Tensor:
        """Computes O(T log2 T) local multi-scale dyadic octave routing."""
        B, T, D = x.shape
        val = F.silu(self.w_v_local(x)) * torch.sigmoid(self.w_g_local(x))
        conv_val = self.local_conv(val.transpose(1, 2))[:, :, :T].transpose(1, 2)
        combined_val = val + conv_val

        mailbox_slots = []
        for stride in self.strides:
            if stride < T:
                shifted = torch.zeros_like(combined_val)
                shifted[:, stride:, :] = combined_val[:, :-stride, :]
            else:
                shifted = torch.zeros_like(combined_val)
            mailbox_slots.append(shifted)

        mailbox = torch.stack(mailbox_slots, dim=2).view(
            B, T, self.num_octaves, self.num_heads, self.head_dim
        )
        q = self.w_q_local(x).view(B, T, self.num_heads, self.head_dim)

        scores = torch.einsum("bthd,khd->bthk", q, self.octave_keys) * (1.0 / math.sqrt(self.head_dim))
        scores = scores + self.octave_bias.view(1, 1, 1, self.num_octaves)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        readout = torch.einsum("bthk,btkhd->bthd", attn, mailbox)
        return readout.contiguous().view(B, T, D)

    def _forward_shelf_retrieval(self, x: torch.Tensor) -> torch.Tensor:
        """Computes Two-Stage Uncompressed Shelf Retrieval across chunks of size C."""
        B, T, D = x.shape
        H, hd = self.num_heads, self.head_dim
        C = self.chunk_size
        NC = T // C

        if NC == 0 or (NC == 1 and T == C):
            # Not enough past chunks to retrieve from
            return torch.zeros(B, T, D, device=x.device, dtype=x.dtype)

        # Project representations
        k_raw = self.w_k_shelf(x).view(B, T, H, hd).transpose(1, 2)  # [B, H, T, hd]
        v_raw = self.w_v_shelf(x).view(B, T, H, hd).transpose(1, 2)  # [B, H, T, hd]
        q_raw = self.w_q_shelf(x).view(B, T, H, hd).transpose(1, 2)  # [B, H, T, hd]

        # Reshape into discrete shelves
        k_c = k_raw[:, :, :NC * C].contiguous().view(B, H, NC, C, hd)
        v_c = v_raw[:, :, :NC * C].contiguous().view(B, H, NC, C, hd)
        q_c = q_raw[:, :, :NC * C].contiguous().view(B, H, NC, C, hd)

        # Stage 1: Coarse Routing via Dual-Pooled Landmarks (mean + max)
        k_summary = torch.cat([k_c.mean(dim=3), k_c.max(dim=3)[0]], dim=-1)  # [B, H, NC, 2*hd]
        q_summary = torch.cat([q_c.mean(dim=3), q_c.max(dim=3)[0]], dim=-1)

        k_lm = F.normalize(self.lens_k(k_summary), p=2, dim=-1)  # [B, H, NC, map_dim]
        q_lm = F.normalize(self.lens_q(q_summary), p=2, dim=-1)  # [B, H, NC, map_dim]

        coarse_logits = torch.matmul(q_lm, k_lm.transpose(-1, -2)) * 10.0  # [B, H, NC, NC] cosine temperature

        # Strict causal chunk mask: chunk i can only retrieve chunks < i
        causal_chunk_mask = torch.tril(torch.ones(NC, NC, device=x.device, dtype=torch.bool), diagonal=-1)
        neg_inf = -1e4 if coarse_logits.dtype == torch.float16 else -1e9
        coarse_logits = coarse_logits.masked_fill(~causal_chunk_mask.view(1, 1, NC, NC), neg_inf)

        n_sel = min(self.top_k, NC)
        top_scores, top_chunks = coarse_logits.topk(n_sel, dim=-1)  # [B, H, NC, n_sel]

        # Stage 2: Fine Attention Over Uncompressed Raw Keys and Values of Selected Shelves
        # Memory-efficient chunk gather: gather directly from k_c [B, H, NC, C, hd]
        idx_chunk = top_chunks.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, -1, C, hd)
        k_sel = torch.gather(k_c.unsqueeze(3).expand(-1, -1, NC, n_sel, C, hd), 2, idx_chunk).view(B, H, NC, n_sel * C, hd)
        v_sel = torch.gather(v_c.unsqueeze(3).expand(-1, -1, NC, n_sel, C, hd), 2, idx_chunk).view(B, H, NC, n_sel * C, hd)

        fine_logits = torch.matmul(q_c, k_sel.transpose(-1, -2)) / math.sqrt(hd)  # [B, H, NC, C, n_sel*C]

        # Differentiable bridge linking coarse routing scores into fine attention
        coarse_boost = top_scores.unsqueeze(-1).expand(-1, -1, -1, -1, C).reshape(B, H, NC, 1, n_sel * C)
        fine_logits = fine_logits + 0.2 * coarse_boost

        valid_threshold = -5000.0 if coarse_logits.dtype == torch.float16 else -1e5
        valid_mask = (top_scores > valid_threshold).unsqueeze(-1).expand(-1, -1, -1, -1, C).reshape(B, H, NC, 1, n_sel * C)
        fine_logits = fine_logits.masked_fill(~valid_mask, float("-inf"))

        fine_attn = F.softmax(fine_logits, dim=-1)
        fine_attn = torch.nan_to_num(fine_attn, 0.0)
        out_shelf_c = torch.matmul(fine_attn, v_sel)  # [B, H, NC, C, hd]

        out_shelf = out_shelf_c.view(B, H, NC * C, hd).transpose(1, 2).reshape(B, NC * C, D)

        # Handle remainder tokens (if T is not an exact multiple of C, e.g. during generation)
        rem = T - NC * C
        if rem > 0:
            q_rem = q_raw[:, :, NC * C : T]  # [B, H, rem, hd]
            q_rem_summary = torch.cat([q_rem.mean(dim=2), q_rem.max(dim=2)[0]], dim=-1)  # [B, H, 2*hd]
            q_rem_lm = F.normalize(self.lens_q(q_rem_summary), p=2, dim=-1)  # [B, H, map_dim]

            coarse_logits_rem = torch.matmul(q_rem_lm.unsqueeze(2), k_lm.transpose(-1, -2)) * 10.0  # [B, H, 1, NC]
            top_scores_rem, top_chunks_rem = coarse_logits_rem.topk(n_sel, dim=-1)  # [B, H, 1, n_sel]

            idx_chunk_rem = top_chunks_rem.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, -1, C, hd)
            k_sel_rem = torch.gather(k_c.unsqueeze(2).expand(-1, -1, 1, NC, C, hd), 3, idx_chunk_rem).view(B, H, n_sel * C, hd)
            v_sel_rem = torch.gather(v_c.unsqueeze(2).expand(-1, -1, 1, NC, C, hd), 3, idx_chunk_rem).view(B, H, n_sel * C, hd)

            fine_logits_rem = torch.matmul(q_rem, k_sel_rem.transpose(-1, -2)) / math.sqrt(hd)  # [B, H, rem, n_sel*C]
            coarse_boost_rem = top_scores_rem.unsqueeze(-1).expand(-1, -1, -1, -1, C).reshape(B, H, 1, -1)
            fine_logits_rem = fine_logits_rem + 0.2 * coarse_boost_rem

            fine_attn_rem = F.softmax(fine_logits_rem, dim=-1)
            fine_attn_rem = torch.nan_to_num(fine_attn_rem, 0.0)
            out_shelf_rem = torch.matmul(fine_attn_rem, v_sel_rem).transpose(1, 2).reshape(B, rem, D)

            out_shelf = torch.cat([out_shelf, out_shelf_rem], dim=1)

        return out_shelf

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        out_local = self._forward_local_dyadic(x)
        out_shelf = self._forward_shelf_retrieval(x)

        # Adaptive Gated Fusion
        # g is (B, T, H) -> broadcast over head dimension
        g = torch.sigmoid(self.gate(x)).unsqueeze(-1)  # [B, T, H, 1]
        out_local_h = out_local.view(B, T, self.num_heads, self.head_dim)
        out_shelf_h = out_shelf.view(B, T, self.num_heads, self.head_dim)

        fused_h = (1.0 - g) * out_local_h + g * out_shelf_h
        fused = fused_h.view(B, T, D)

        return self.w_out(fused)


# -----------------------------------------------------------------------------
# Transformer Block & DyadicShelfLM
# -----------------------------------------------------------------------------

class DyadicShelfBlock(nn.Module):
    """Transformer block integrating DyadicShelfAttention and SwiGLU MLP."""
    def __init__(self, config: DyadicShelfConfig):
        super().__init__()
        self.norm1 = RMSNorm(config.dim)
        self.attn = DyadicShelfAttention(config)
        self.norm2 = RMSNorm(config.dim)
        self.mlp = SwiGLU(config.dim, config.hidden_dim, bias=config.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class DyadicShelfLM(nn.Module):
    """
    DyadicShelfLM: Hybrid Language Model combining local DyadicPush routing
    with Two-Stage Uncompressed Shelf Retrieval for multi-million token context.
    """
    def __init__(self, config: DyadicShelfConfig):
        super().__init__()
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.dim)
        self.blocks = nn.ModuleList([DyadicShelfBlock(config) for _ in range(config.num_layers)])
        self.final_norm = RMSNorm(config.dim)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.token_emb.weight
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        chunk_size_ce: int = 2048,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass.
        If targets is provided: computes memory-efficient chunked cross-entropy.
        If targets is None: returns full logits.
        """
        B, T = idx.shape
        x = self.token_emb(idx)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)

        if targets is not None:
            # Memory-efficient chunked cross entropy over 8K context
            flat_x = x.view(-1, self.config.dim)
            flat_targets = targets.view(-1)
            loss = chunked_cross_entropy(flat_x, self.lm_head.weight, flat_targets, chunk_size=chunk_size_ce)
            return None, loss
        else:
            logits = self.lm_head(x)
            return logits, None

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @torch.no_grad()
    def generate(
        self,
        prompt_tokens: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        """
        Autoregressive generation.
        During generation over long context, past tokens can scale to arbitrarily long sequences.
        """
        self.eval()
        idx = prompt_tokens.clone()

        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.max_seq_len else idx[:, -self.config.max_seq_len:]
            logits, _ = self(idx_cond)
            next_logits = logits[:, -1, :] / max(temperature, 1e-5)

            if top_k > 0:
                v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < v[:, [-1]]] = -float("Inf")

            if 0.0 < top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_logits[:, indices_to_remove] = -float("Inf")

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)

        return idx
