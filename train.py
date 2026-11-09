"""
train.py - Training Script for DyadicShelf LM (DyadicPush V2)
============================================================
Trains a hybrid DyadicPush + Two-Stage Shelf Retrieval language model.
Designed for 8K context training on single GPUs and scaling to multi-million
token context during inference.

Features:
- Sub-quadratic O(T log2 T) local dyadic geometric routing.
- Two-stage uncompressed shelf retrieval across chunks of size C = 64.
- Memory-efficient chunked cross-entropy (prevents VRAM spikes at 8K context).
- Automatic BPE tokenizer training/loading with ByteLevel decoding.
- Cosine learning rate decay with linear warmup.
- Mixed precision training (AMP autocast) and gradient clipping.
"""

import argparse
import math
import os
import random
import time
from dataclasses import asdict

import numpy as np
import torch
import torch.nn as nn
from model import DyadicShelfConfig, DyadicShelfLM
from tokenizer import (
    BPETokenizerWrapper,
    CharTokenizer,
    load_or_create_tokenizer,
)


def set_seed(seed: int = 1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_batch(data: torch.Tensor, batch_size: int, seq_len: int, device: str):
    max_idx = len(data) - seq_len - 1
    if max_idx <= 0:
        raise ValueError(
            f"Dataset length ({len(data)}) is too short for seq_len ({seq_len}). "
            "Please provide a larger text corpus or reduce --seq_len."
        )
    ix = torch.randint(0, max_idx, (batch_size,))
    x = torch.stack([data[i : i + seq_len] for i in ix]).to(device).long()
    y = torch.stack([data[i + 1 : i + seq_len + 1] for i in ix]).to(device).long()
    return x, y


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    data: torch.Tensor,
    batch_size: int,
    seq_len: int,
    device: str,
    num_batches: int = 15,
    use_amp: bool = True,
):
    model.eval()
    losses = []
    stride = (len(data) - seq_len - 1) // max(num_batches, 1)
    if use_amp and device.startswith("cuda"):
        try:
            amp_ctx = torch.amp.autocast("cuda", dtype=torch.float16)
        except Exception:
            amp_ctx = torch.cuda.amp.autocast(dtype=torch.float16)
    else:
        amp_ctx = torch.no_grad()

    for b in range(num_batches):
        idx = min(b * stride, len(data) - seq_len - 1)
        x = data[idx : idx + seq_len].unsqueeze(0).to(device).long()
        y = data[idx + 1 : idx + seq_len + 1].unsqueeze(0).to(device).long()
        with amp_ctx:
            _, loss = model(x, y)
        losses.append(loss.item())

    model.train()
    mean_loss = float(np.mean(losses))
    perplexity = math.exp(min(mean_loss, 20.0))
    return mean_loss, perplexity


def get_lr(step: int, warmup_steps: int, max_steps: int, lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return lr * (step + 1) / (warmup_steps + 1)
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (lr - min_lr)


def main():
    parser = argparse.ArgumentParser(description="Train DyadicShelf LM (DyadicPush V2) on input.txt")
    parser.add_argument("--data", type=str, default="input.txt", help="Path to input text file")
    parser.add_argument("--dim", type=int, default=384, help="Embedding dimension")
    parser.add_argument("--hidden_dim", type=int, default=1024, help="SwiGLU hidden dimension")
    parser.add_argument("--num_layers", type=int, default=8, help="Number of layers")
    parser.add_argument("--num_heads", type=int, default=6, help="Number of attention heads")
    parser.add_argument("--seq_len", type=int, default=4096, help="Context sequence length (e.g. 4096 or 8192)")
    parser.add_argument("--chunk_size", type=int, default=64, help="Shelf chunk size C")
    parser.add_argument("--top_k", type=int, default=4, help="Top-K shelves to retrieve")
    parser.add_argument("--map_dim", type=int, default=32, help="Landmark projection dimension")
    parser.add_argument("--batch_size", type=int, default=2, help="Micro-batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Peak learning rate")
    parser.add_argument("--min_lr", type=float, default=5e-5, help="Minimum learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="AdamW weight decay")
    parser.add_argument("--max_steps", type=int, default=3000, help="Total training steps")
    parser.add_argument("--warmup_steps", type=int, default=150, help="Linear warmup steps")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--eval_interval", type=int, default=250, help="Evaluation interval in steps")
    parser.add_argument("--generate_interval", type=int, default=500, help="Sample generation interval")
    parser.add_argument("--sample_prompt", type=str, default="To be, or not to be, that is the question:", help="Sample prompt")
    parser.add_argument("--tokenizer_type", type=str, default="auto", choices=["auto", "bpe", "char"], help="Tokenizer type")
    parser.add_argument("--vocab_size", type=int, default=8192, help="BPE vocabulary size")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda/cpu)")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed")
    parser.add_argument("--no_amp", action="store_true", help="Disable mixed precision")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print("=" * 70)
    print("DyadicPush V2 (DyadicShelf LM): Training Engine")
    print("=" * 70)

    if not os.path.exists(args.data):
        print(f"[!] Error: Data file '{args.data}' not found.")
        return

    # 1. Tokenizer
    tokenizer = load_or_create_tokenizer(
        args.data,
        tokenizer_type=args.tokenizer_type,
        vocab_size=args.vocab_size,
        save_dir=args.checkpoint_dir,
    )

    print(f"[*] Encoding '{args.data}' into token IDs...")
    with open(args.data, "r", encoding="utf-8", errors="ignore") as f:
        raw_text = f.read()

    all_tokens = torch.tensor(tokenizer.encode(raw_text), dtype=torch.long)
    total_tokens = len(all_tokens)
    print(f"[*] Total dataset tokens: {total_tokens:,}")

    n_train = int(0.9 * total_tokens)
    train_tokens = all_tokens[:n_train]
    val_tokens = all_tokens[n_train:]
    print(f"[*] Train tokens: {len(train_tokens):,} | Val tokens: {len(val_tokens):,}")

    # 2. Model
    config = DyadicShelfConfig(
        vocab_size=tokenizer.vocab_size,
        dim=args.dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        max_seq_len=args.seq_len,
        chunk_size=args.chunk_size,
        top_k=args.top_k,
        map_dim=args.map_dim,
    )
    model = DyadicShelfLM(config).to(args.device)
    num_params = model.get_num_params()

    num_octaves = math.ceil(math.log2(args.seq_len))
    num_chunks = args.seq_len // args.chunk_size
    print("-" * 70)
    print(f"[*] Architecture:   DyadicShelfLM (DyadicPush V2)")
    print(f"[*] Parameters:     {num_params:,} (~{num_params / 1e6:.2f}M)")
    print(f"[*] Context Length: {args.seq_len} tokens")
    print(f"[*] Local Engine:   DyadicPush with {num_octaves} octaves (1 to {1 << (num_octaves - 1)})")
    print(f"[*] Shelf Memory:   {num_chunks} chunks of size {args.chunk_size} | Top-{args.top_k} retrieval")
    print(f"[*] Landmark Space: {args.map_dim}-dim holographic dual-pooling (mean + max)")
    print(f"[*] Device:         {args.device}")
    print("-" * 70)

    # 3. Optimizer
    decay_params = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay_params = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": args.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=args.lr, betas=(0.9, 0.95), eps=1e-8)

    use_amp = (not args.no_amp) and args.device.startswith("cuda")
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    if isinstance(tokenizer, BPETokenizerWrapper):
        tok_data = {"type": "bpe", "data": tokenizer.to_json_str()}
    elif isinstance(tokenizer, CharTokenizer):
        tok_data = tokenizer.to_dict()
    else:
        tok_data = {"type": "unknown"}

    best_val_loss = float("inf")
    t0 = time.time()
    print(f"[*] Starting training for {args.max_steps} steps...")

    for step in range(1, args.max_steps + 1):
        cur_lr = get_lr(step, args.warmup_steps, args.max_steps, args.lr, args.min_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = cur_lr

        x, y = get_batch(train_tokens, args.batch_size, args.seq_len, args.device)

        step_t0 = time.time()
        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            try:
                autocast_ctx = torch.amp.autocast("cuda", dtype=torch.float16)
            except Exception:
                autocast_ctx = torch.cuda.amp.autocast(dtype=torch.float16)
            with autocast_ctx:
                _, loss = model(x, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            _, loss = model(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

        step_dt = (time.time() - step_t0) * 1000.0  # ms
        tokens_per_sec = (args.batch_size * args.seq_len) / (step_dt / 1000.0)

        if step % 50 == 0 or step == 1:
            vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if args.device.startswith("cuda") else 0.0
            print(
                f"Step {step:5d}/{args.max_steps} | "
                f"Loss: {loss.item():.4f} | "
                f"LR: {cur_lr:.2e} | "
                f"Latency: {step_dt:6.1f} ms | "
                f"Throughput: {tokens_per_sec:7.1f} tok/s | "
                f"VRAM: {vram_mb:.0f} MB"
            )

        if step % args.eval_interval == 0 or step == args.max_steps:
            val_loss, perplexity = evaluate_loss(
                model,
                val_tokens,
                args.batch_size,
                args.seq_len,
                args.device,
                num_batches=15,
                use_amp=use_amp,
            )
            print("*" * 70)
            print(f"[EVAL @ Step {step}] Val Loss: {val_loss:.4f} nats/token | Perplexity: {perplexity:.2f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_ckpt_path = os.path.join(args.checkpoint_dir, "best_model.pt")
                torch.save(
                    {
                        "step": step,
                        "model_state_dict": model.state_dict(),
                        "config": asdict(config),
                        "val_loss": val_loss,
                        "tokenizer": tok_data,
                    },
                    best_ckpt_path,
                )
                print(f"[*] Saved new best checkpoint to '{best_ckpt_path}'")
            print("*" * 70)

        if step % args.generate_interval == 0 or step == args.max_steps:
            model.eval()
            prompt_tokens = torch.tensor([tokenizer.encode(args.sample_prompt)], dtype=torch.long, device=args.device)
            gen_tokens = model.generate(prompt_tokens, max_new_tokens=64, temperature=0.8, top_p=0.9)[0].tolist()
            generated_text = tokenizer.decode(gen_tokens)
            print("-" * 70)
            print(f"[SAMPLE GENERATION @ Step {step}]")
            print(generated_text)
            print("-" * 70)
            model.train()

    total_time = time.time() - t0
    print("=" * 70)
    print(f"Training Complete in {total_time:.1f}s ({total_time / 60:.2f} mins).")
    print(f"Best Validation Loss: {best_val_loss:.4f}")

    final_ckpt_path = os.path.join(args.checkpoint_dir, "final_model.pt")
    torch.save(
        {
            "step": args.max_steps,
            "model_state_dict": model.state_dict(),
            "config": asdict(config),
            "val_loss": best_val_loss,
            "tokenizer": tok_data,
        },
        final_ckpt_path,
    )
    print(f"[*] Saved final checkpoint to '{final_ckpt_path}'")
    print("=" * 70)


if __name__ == "__main__":
    main()
