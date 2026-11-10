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
- Multi-dataset Parquet streaming with lookahead buffer and exact consumption ledger.
- Temperature-adjusted weighted mixing: p_i ~ w_i^(1/T).
- Micro-batch gradient accumulation with optional Gradient Sign Chaos Injection (GSCI).
- Automatic BPE tokenizer training/loading with ByteLevel decoding.
- Cosine learning rate decay with linear warmup (or pinned forced LR).
- Mixed precision training (AMP autocast) and gradient clipping.
- Atomic checkpoint saving and crash-resilient completed-files logging.
"""

import argparse
import glob
import math
import os
import random
import sys
import time
from dataclasses import asdict
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from dataset_mixer import (
    DatasetMixer,
    Ledger,
    parse_dataset_specs,
    parse_text_columns,
    print_mix_stats,
    reconcile_completed,
    write_completed_log,
)
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


def get_lr(step: int, warmup_steps: int, max_steps: int, lr: float, min_lr: float, forced_lr: float = 0.0) -> float:
    if forced_lr > 0:
        return forced_lr
    if step < warmup_steps:
        return lr * (step + 1) / max(1, warmup_steps + 1)
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (lr - min_lr)


def gsci_combine(grads: List[torch.Tensor], lam: float) -> torch.Tensor:
    """Gradient Sign Chaos Injection over per-group gradients: mean - lam * var * sign(mean)."""
    s = torch.stack([g.detach() for g in grads])
    mean = s.mean(dim=0)
    var = s.var(dim=0, unbiased=False)
    return mean - lam * var * torch.sign(mean)


def gsci_groups(n_micro: int, shards: int) -> List[List[int]]:
    """Contiguous groups of micro-batch indices."""
    k = min(shards, n_micro)
    return [list(g) for g in np.array_split(np.arange(n_micro), k) if len(g)]


def save_checkpoint_atomic(path: str, data: dict):
    """Atomic checkpoint write (.tmp + os.replace)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(data, tmp)
    os.replace(tmp, path)


def train_tokenizer_from_parquet(
    specs: List[Tuple[str, str, float]],
    default_col: str,
    col_map: dict,
    vocab_size: int,
    save_path: str,
    samples_per_dataset: int = 10000,
) -> BPETokenizerWrapper:
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    samples = []
    for name, path, _ in specs:
        ds_samples = 0
        col = col_map.get(name, default_col)
        found = sorted(glob.glob(os.path.join(glob.escape(path), "**", "*.parquet"), recursive=True))
        for fp in found:
            try:
                pf = pq.ParquetFile(fp)
                for batch in pf.iter_batches(batch_size=200, columns=[col]):
                    for text in batch.column(0).to_pylist():
                        if text:
                            samples.append(str(text))
                            ds_samples += 1
                            if ds_samples >= samples_per_dataset:
                                break
                    if ds_samples >= samples_per_dataset:
                        break
                if hasattr(pf, "close"):
                    pf.close()
            except Exception:
                pass
            if ds_samples >= samples_per_dataset:
                break
        print(f"[*] Collected {ds_samples:,} samples from '{name}' for tokenizer training.")

    if not samples:
        raise RuntimeError("No text could be extracted from parquet files to train a tokenizer.")

    print(f"[*] Training BPE tokenizer on {len(samples):,} total text samples from {len(specs)} dataset(s)...")
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<unk>", "<s>", "</s>", "<pad>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tokenizer.train_from_iterator(samples, trainer)
    actual_vocab = tokenizer.get_vocab_size()
    print(f"[*] Tokenizer training complete. Vocab size: {actual_vocab:,} (target: {vocab_size:,})")
    if save_path:
        tokenizer.save(save_path)
    return BPETokenizerWrapper(tokenizer)


def accumulate_grads(
    model: nn.Module,
    micro: List[Tuple[torch.Tensor, torch.Tensor]],
    args,
    device: str,
    use_amp: bool,
    scaler=None,
    use_gsci: bool = False,
) -> float:
    params = [p for p in model.parameters() if p.requires_grad]
    n = len(micro)
    tot_loss = 0.0

    if not use_gsci:
        for p in params:
            p.grad = None
        for x, y in micro:
            if use_amp and device.startswith("cuda"):
                try:
                    autocast_ctx = torch.amp.autocast("cuda", dtype=torch.float16)
                except Exception:
                    autocast_ctx = torch.cuda.amp.autocast(dtype=torch.float16)
                with autocast_ctx:
                    _, loss = model(x, y)
                scaled_loss = loss * (1.0 / n)
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
            else:
                _, loss = model(x, y)
                (loss * (1.0 / n)).backward()
            tot_loss += loss.item() / n
        return tot_loss

    # GSCI enabled
    group_grads = []
    for grp in gsci_groups(n, args.gsci_shards):
        for p in params:
            p.grad = None
        for i in grp:
            x, y = micro[i]
            scale = 1.0 / len(grp)
            if use_amp and device.startswith("cuda"):
                try:
                    autocast_ctx = torch.amp.autocast("cuda", dtype=torch.float16)
                except Exception:
                    autocast_ctx = torch.cuda.amp.autocast(dtype=torch.float16)
                with autocast_ctx:
                    _, loss = model(x, y)
                scaled_loss = loss * scale
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
            else:
                _, loss = model(x, y)
                (loss * scale).backward()
            tot_loss += loss.item() / n

        # Capture group gradients
        if use_amp and scaler is not None and device.startswith("cuda") and scaler.is_enabled():
            inv_scale = 1.0 / max(scaler.get_scale(), 1e-8)
            group_grads.append([
                (p.grad.detach().clone() * inv_scale) if p.grad is not None else None
                for p in params
            ])
        else:
            group_grads.append([
                p.grad.detach().clone() if p.grad is not None else None
                for p in params
            ])

    # Recombine via GSCI into p.grad
    for i, p in enumerate(params):
        gs = [g[i] for g in group_grads]
        if all(g is None for g in gs):
            p.grad = None
        else:
            combined = gsci_combine([g if g is not None else torch.zeros_like(p) for g in gs], args.gsci_lambda)
            if scaler is not None and scaler.is_enabled() and device.startswith("cuda"):
                combined = combined * scaler.get_scale()
            p.grad = combined

    return tot_loss


def run_mixed_training(
    args,
    specs: List[Tuple[str, str, float]],
    model: DyadicShelfLM,
    optimizer: torch.optim.Optimizer,
    tokenizer,
    device: str,
    config: DyadicShelfConfig,
    tok_data: dict,
    ckpt: Optional[dict],
):
    log_path = args.completed_log or os.path.join(args.checkpoint_dir, "completed_files.json")
    ledger = Ledger(reconcile_completed(args, ckpt, log_path))
    default_col, col_map = parse_text_columns(args.text_column)
    mixer = DatasetMixer(
        specs,
        tokenizer,
        args.seq_len,
        args.mix_chunk,
        args.mix_temperature,
        default_col,
        col_map,
        ledger,
        seed=args.seed,
    )
    if ckpt is not None:
        mixer.load_stats(ckpt.get("mixer_stats"))

    if not mixer.active:
        print("[*] Nothing to train: every dataset is fully completed (or has no files).")
        print("    Use --reset_completed to train them again. No checkpoint written.")
        return

    accum = max(1, args.grad_accum)
    use_gsci = bool(args.gsci)
    if use_gsci and min(args.gsci_shards, accum) < 2:
        print(f"[!] GSCI disabled: needs >= 2 gradient groups (gsci_shards={args.gsci_shards}, grad_accum={accum}).")
        use_gsci = False

    use_amp = (not args.no_amp) and device.startswith("cuda")
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    start_step = int(ckpt.get("step", ckpt.get("global_step", 0))) if ckpt else 0
    if start_step >= args.max_steps:
        print(f"[*] Step {start_step} >= --max_steps {args.max_steps}: nothing to do. Raise --max_steps to continue.")
        return

    print("-" * 75)
    print(f"[*] Mixed Parquet Training: {len(mixer.active)} dataset(s) | chunk {mixer.chunk_len} tokens")
    print(f"[*] Micro-batch: {args.batch_size} x Accum {accum} = {args.batch_size * accum} seqs/step")
    print(f"[*] Context: {args.seq_len} tokens | Tokens/step: {args.batch_size * accum * args.seq_len:,}")
    print(f"[*] GSCI: {'on (shards=%d, lambda=%g)' % (min(args.gsci_shards, accum), args.gsci_lambda) if use_gsci else 'off'}")
    print(
        f"[*] LR: "
        + (f"FORCED {args.forced_lr:.2e}" if args.forced_lr > 0 else f"{args.lr:.0e} -> {args.min_lr:.0e} cosine over {args.max_steps:,} steps")
        + f" | Steps {start_step + 1}..{args.max_steps} | Save every {args.save_interval} | Log '{log_path}'"
    )
    print("-" * 75)

    def do_save(current_step: int, tag: str, loss_val: Optional[float] = None):
        snap = ledger.snapshot()
        ckpt_data = {
            "step": current_step,
            "global_step": current_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": asdict(config),
            "val_loss": loss_val,
            "tokenizer": tok_data,
            "completed_files": snap,
            "mixer_stats": mixer.stats_state(),
            "data_mode": "mixed",
            "args": vars(args),
        }
        ckpt_file = os.path.join(args.checkpoint_dir, f"{tag}_model.pt")
        save_checkpoint_atomic(ckpt_file, ckpt_data)
        write_completed_log(log_path, snap, current_step)
        ledger.commit()
        print(f"[*] Checkpoint saved ({tag}) at step {current_step}: '{ckpt_file}' | completed files: {len(snap)}")

    global_step = start_step
    interrupted = False
    stop_reason = f"reached --max_steps {args.max_steps}"
    tot_loss = float("nan")
    t0 = time.time()
    model.train()

    try:
        while global_step < args.max_steps:
            micro = []
            for _ in range(accum):
                b = mixer.get_batch(args.batch_size, device)
                if b is None:
                    break
                micro.append(b)

            if not micro:
                stop_reason = "all datasets exhausted"
                ledger.apply()
                break

            step_no = global_step + 1
            cur_lr = get_lr(step_no, args.warmup_steps, args.max_steps, args.lr, args.min_lr, args.forced_lr)
            for g in optimizer.param_groups:
                g["lr"] = cur_lr

            step_t0 = time.time()
            tot_loss = accumulate_grads(model, micro, args, device, use_amp, scaler, use_gsci)

            if not math.isfinite(tot_loss):
                raise RuntimeError(f"Non-finite loss at step {step_no}: {tot_loss}")

            if scaler is not None and scaler.is_enabled() and device.startswith("cuda"):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)
            global_step = step_no
            ledger.apply()

            step_dt = (time.time() - step_t0) * 1000.0  # ms
            tokens_in_step = sum(len(x_i) * args.seq_len for x_i, _ in micro)
            tokens_per_sec = tokens_in_step / max(step_dt / 1000.0, 1e-6)

            if global_step % 50 == 0 or global_step == 1 or global_step == start_step + 1:
                vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if device.startswith("cuda") else 0.0
                print(
                    f"Step {global_step:5d}/{args.max_steps} | "
                    f"Loss: {tot_loss:.4f} | "
                    f"LR: {cur_lr:.2e} | "
                    f"Latency: {step_dt:6.1f} ms | "
                    f"Throughput: {tokens_per_sec:7.1f} tok/s | "
                    f"VRAM: {vram_mb:.0f} MB"
                )

            if global_step % args.generate_interval == 0 or global_step == args.max_steps:
                model.eval()
                try:
                    prompt_tokens = torch.tensor(
                        [tokenizer.encode(args.sample_prompt)], dtype=torch.long, device=device
                    )
                    gen_tokens = model.generate(prompt_tokens, max_new_tokens=64, temperature=0.8, top_p=0.9)[0].tolist()
                    generated_text = tokenizer.decode(gen_tokens)
                    print("-" * 75)
                    print(f"[SAMPLE GENERATION @ Step {global_step}]")
                    print(generated_text)
                    print("-" * 75)
                except Exception as e:
                    print(f"[!] Sample generation error: {e}")
                model.train()

            if global_step % args.stats_interval == 0:
                print_mix_stats(mixer, ledger, global_step)

            if global_step % args.save_interval == 0:
                do_save(global_step, "checkpoint", tot_loss)

    except KeyboardInterrupt:
        interrupted = True
        dropped = ledger.discard_unapplied()
        print(
            f"\n[!] Interrupted: partial accumulation discarded ({dropped} unapplied file(s) not logged); "
            f"saving at last applied step {global_step}."
        )

    total_time = time.time() - t0
    if global_step > start_step or ledger.applied:
        print(f"\n[*] Training stopped: {stop_reason} (step {global_step}) in {total_time:.1f}s | last loss: {tot_loss:.4f}")
        print_mix_stats(mixer, ledger, global_step)
        do_save(global_step, "final" if not interrupted else "interrupt", tot_loss)
    else:
        print("[*] No optimizer step was run; no checkpoint written.")


def main():
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass

    parser = argparse.ArgumentParser(
        description="Train DyadicShelf LM (DyadicPush V2) on input text or streaming Parquet files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Model architecture
    parser.add_argument("--dim", type=int, default=384, help="Embedding dimension")
    parser.add_argument("--hidden_dim", type=int, default=1024, help="SwiGLU hidden dimension")
    parser.add_argument("--num_layers", type=int, default=8, help="Number of layers")
    parser.add_argument("--num_heads", type=int, default=6, help="Number of attention heads")
    parser.add_argument("--seq_len", type=int, default=4096, help="Context sequence length (e.g. 4096 or 8192)")
    parser.add_argument("--chunk_size", type=int, default=64, help="Shelf chunk size C")
    parser.add_argument("--top_k", type=int, default=4, help="Top-K shelves to retrieve")
    parser.add_argument("--map_dim", type=int, default=32, help="Landmark projection dimension")

    # Training & Optimization
    parser.add_argument("--batch_size", type=int, default=2, help="Micro-batch size")
    parser.add_argument("--grad_accum", type=int, default=1, help="Micro-batches per optimizer step")
    parser.add_argument("--lr", type=float, default=5e-4, help="Peak learning rate")
    parser.add_argument("--min_lr", type=float, default=5e-5, help="Minimum learning rate")
    parser.add_argument("--forced_lr", type=float, default=0.0, help="If > 0, pin LR to this value every step")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="AdamW weight decay")
    parser.add_argument("--max_steps", type=int, default=3000, help="Total training steps")
    parser.add_argument("--warmup_steps", type=int, default=150, help="Linear warmup steps")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--eval_interval", type=int, default=250, help="Evaluation interval in steps (text mode)")
    parser.add_argument("--generate_interval", type=int, default=500, help="Sample generation interval")
    parser.add_argument("--save_interval", type=int, default=1000, help="Checkpoint save interval in steps")
    parser.add_argument("--sample_prompt", type=str, default="To be, or not to be, that is the question:", help="Sample prompt")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda/cpu)")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--resume", type=str, default=None, help="Resume training from a specific checkpoint file")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed")
    parser.add_argument("--no_amp", action="store_true", help="Disable mixed precision")

    # GSCI
    parser.add_argument("--gsci", action=argparse.BooleanOptionalAction, default=False, help="Gradient Sign Chaos Injection")
    parser.add_argument("--gsci_shards", type=int, default=4, help="GSCI gradient groups (clamped to --grad_accum; <2 disables)")
    parser.add_argument("--gsci_lambda", type=float, default=0.1, help="GSCI chaos factor: g = mean - lambda * var * sign(mean)")

    # Legacy text mode
    parser.add_argument("--data", type=str, default="input.txt", help="Path to input text file (legacy mode)")
    parser.add_argument("--tokenizer_type", type=str, default="auto", choices=["auto", "bpe", "char"], help="Tokenizer type")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to existing tokenizer JSON file")
    parser.add_argument("--vocab_size", type=int, default=32000, help="BPE vocabulary size")

    # Mixed Parquet streaming mode
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        metavar="NAME=PATH[:WEIGHT]",
        help="Mixed parquet training: one or more NAME=DIR[:WEIGHT] specs (weight default 1.0). "
             "Without this flag, the legacy text mode runs.",
    )
    parser.add_argument("--mix_temperature", type=float, default=1.0, help="Mixing temperature: probs ~ weight^(1/T)")
    parser.add_argument("--mix_chunk", type=int, default=4096, help="Target tokens per mixer chunk (rounded to whole windows)")
    parser.add_argument("--text_column", type=str, default="text", help="Parquet text column: 'col' or 'name=col,name2=col2'")
    parser.add_argument("--completed_log", type=str, default=None, help="Path to completed-parquet-files log JSON")
    parser.add_argument("--reset_completed", action="store_true", help="Ignore completed-files log and retrain all files")
    parser.add_argument("--retrain_tokenizer", action="store_true", help="Force retraining the tokenizer even if tokenizer.json exists")
    parser.add_argument("--stats_interval", type=int, default=500, help="Print mixer stats every N optimizer steps")

    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print("=" * 75)
    print("DyadicPush V2 (DyadicShelf LM): Training Engine")
    print("=" * 75)

    mixed = args.datasets is not None
    specs = None
    if mixed:
        try:
            specs = parse_dataset_specs(args.datasets, args.mix_temperature)
            if args.mix_chunk < 1 or args.save_interval < 1 or args.stats_interval < 1 or args.max_steps < 1:
                raise ValueError("--mix_chunk, --save_interval, --stats_interval and --max_steps must be >= 1")
            parse_text_columns(args.text_column)
        except ValueError as e:
            print(f"[!] Error in dataset configuration: {e}")
            sys.exit(2)
        print("[*] Mode: Mixed-Dataset Parquet Streaming")
        print("    Datasets: " + ", ".join(f"{n} (w={w:g})" for n, _, w in specs) + f" | T={args.mix_temperature:g}")
    else:
        print("[*] Mode: Legacy Single Text File")
        if not os.path.exists(args.data):
            print(f"[!] Error: Data file '{args.data}' not found.")
            sys.exit(1)

    # Checkpoint to resume from
    ckpt = None
    if args.resume and os.path.exists(args.resume):
        print(f"[*] Resuming from checkpoint: '{args.resume}'")
        ckpt = torch.load(args.resume, map_location=args.device, weights_only=False)

    # 1. Tokenizer initialization
    tokenizer = None
    tok_json_path = os.path.join(args.checkpoint_dir, "tokenizer.json")
    if args.tokenizer_path and os.path.exists(args.tokenizer_path):
        tokenizer = load_or_create_tokenizer(
            data_path=None,
            tokenizer_type=args.tokenizer_type,
            vocab_size=args.vocab_size,
            save_dir=args.checkpoint_dir,
            tokenizer_path=args.tokenizer_path,
        )
    elif ckpt is not None and "tokenizer" in ckpt:
        tok_info = ckpt["tokenizer"]
        tok_type = tok_info.get("type", "unknown")
        if tok_type == "bpe":
            tokenizer = BPETokenizerWrapper.from_json_str(tok_info["data"])
        elif tok_type == "char":
            tokenizer = CharTokenizer.from_dict(tok_info)
    elif os.path.exists(tok_json_path) and not args.retrain_tokenizer:
        tokenizer = load_or_create_tokenizer(
            data_path=None,
            tokenizer_type=args.tokenizer_type,
            vocab_size=args.vocab_size,
            save_dir=args.checkpoint_dir,
            tokenizer_path=tok_json_path,
        )
        if tokenizer.vocab_size != args.vocab_size:
            print(
                f"[!] Notice: Loaded existing tokenizer '{tok_json_path}' with vocab_size={tokenizer.vocab_size:,}, "
                f"which differs from requested --vocab_size={args.vocab_size:,}."
            )
            print(
                f"    To retrain with vocab_size={args.vocab_size:,}, pass --retrain_tokenizer or specify a new --checkpoint_dir."
            )
    elif not mixed and os.path.exists(args.data):
        tokenizer = load_or_create_tokenizer(
            data_path=args.data,
            tokenizer_type=args.tokenizer_type,
            vocab_size=args.vocab_size,
            save_dir=args.checkpoint_dir,
        )
    elif mixed and specs:
        default_col, col_map = parse_text_columns(args.text_column)
        tokenizer = train_tokenizer_from_parquet(
            specs, default_col, col_map, args.vocab_size, tok_json_path
        )

    if tokenizer is None:
        print("[!] Error: Could not load or initialize a tokenizer.")
        sys.exit(1)

    if isinstance(tokenizer, BPETokenizerWrapper):
        tok_data = {"type": "bpe", "data": tokenizer.to_json_str()}
    elif isinstance(tokenizer, CharTokenizer):
        tok_data = tokenizer.to_dict()
    else:
        tok_data = {"type": "unknown"}

    # 2. Model configuration
    if ckpt and "config" in ckpt:
        config_dict = ckpt["config"]
        config = DyadicShelfConfig(**config_dict)
    else:
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
    if ckpt and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        print("[*] Loaded model weights from checkpoint.")

    num_params = model.get_num_params()
    num_octaves = math.ceil(math.log2(args.seq_len))
    num_chunks = args.seq_len // args.chunk_size

    print("-" * 75)
    print(f"[*] Architecture:   DyadicShelfLM (DyadicPush V2)")
    print(f"[*] Parameters:     {num_params:,} (~{num_params / 1e6:.2f}M)")
    print(f"[*] Vocab Size:     {tokenizer.vocab_size:,}")
    print(f"[*] Context Length: {args.seq_len} tokens")
    print(f"[*] Local Engine:   DyadicPush with {num_octaves} octaves (1 to {1 << (num_octaves - 1)})")
    print(f"[*] Shelf Memory:   {num_chunks} chunks of size {args.chunk_size} | Top-{args.top_k} retrieval")
    print(f"[*] Landmark Space: {args.map_dim}-dim holographic dual-pooling (mean + max)")
    print(f"[*] Device:         {args.device}")
    print("-" * 75)

    # 3. Optimizer
    decay_params = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay_params = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": args.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=args.lr, betas=(0.9, 0.95), eps=1e-8)
    if ckpt and "optimizer_state_dict" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            print("[*] Loaded optimizer state from checkpoint.")
        except Exception as e:
            print(f"[!] Warning: Could not restore optimizer state: {e}")

    # 4. Dispatch training mode
    if mixed:
        run_mixed_training(
            args=args,
            specs=specs,
            model=model,
            optimizer=optimizer,
            tokenizer=tokenizer,
            device=args.device,
            config=config,
            tok_data=tok_data,
            ckpt=ckpt,
        )
    else:
        # Legacy single text file mode
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

        use_amp = (not args.no_amp) and args.device.startswith("cuda")
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        except Exception:
            scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        best_val_loss = float("inf")
        start_step = int(ckpt.get("step", ckpt.get("global_step", 0))) if ckpt else 0
        t0 = time.time()
        print(f"[*] Starting training for steps {start_step + 1}..{args.max_steps}...")

        for step in range(start_step + 1, args.max_steps + 1):
            cur_lr = get_lr(step, args.warmup_steps, args.max_steps, args.lr, args.min_lr, args.forced_lr)
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
            tokens_per_sec = (args.batch_size * args.seq_len) / max(step_dt / 1000.0, 1e-6)

            if step % 50 == 0 or step == 1 or step == start_step + 1:
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
                print("*" * 75)
                print(f"[EVAL @ Step {step}] Val Loss: {val_loss:.4f} nats/token | Perplexity: {perplexity:.2f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_ckpt_path = os.path.join(args.checkpoint_dir, "best_model.pt")
                    save_checkpoint_atomic(
                        best_ckpt_path,
                        {
                            "step": step,
                            "global_step": step,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "config": asdict(config),
                            "val_loss": val_loss,
                            "tokenizer": tok_data,
                            "data_mode": "text",
                            "args": vars(args),
                        },
                    )
                    print(f"[*] Saved new best checkpoint to '{best_ckpt_path}'")
                print("*" * 75)

            if step % args.generate_interval == 0 or step == args.max_steps:
                model.eval()
                try:
                    prompt_tokens = torch.tensor(
                        [tokenizer.encode(args.sample_prompt)], dtype=torch.long, device=args.device
                    )
                    gen_tokens = model.generate(prompt_tokens, max_new_tokens=64, temperature=0.8, top_p=0.9)[0].tolist()
                    generated_text = tokenizer.decode(gen_tokens)
                    print("-" * 75)
                    print(f"[SAMPLE GENERATION @ Step {step}]")
                    print(generated_text)
                    print("-" * 75)
                except Exception as e:
                    print(f"[!] Sample generation error: {e}")
                model.train()

        total_time = time.time() - t0
        print("=" * 75)
        print(f"Training Complete in {total_time:.1f}s ({total_time / 60:.2f} mins).")
        if best_val_loss < float("inf"):
            print(f"Best Validation Loss: {best_val_loss:.4f}")

        final_ckpt_path = os.path.join(args.checkpoint_dir, "final_model.pt")
        save_checkpoint_atomic(
            final_ckpt_path,
            {
                "step": args.max_steps,
                "global_step": args.max_steps,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": asdict(config),
                "val_loss": best_val_loss,
                "tokenizer": tok_data,
                "data_mode": "text",
                "args": vars(args),
            },
        )
        print(f"[*] Saved final checkpoint to '{final_ckpt_path}'")
        print("=" * 75)


if __name__ == "__main__":
    main()

