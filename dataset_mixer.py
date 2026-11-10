"""
dataset_mixer.py - Streaming Multi-Dataset Parquet Mixer with Consumption Ledger
================================================================================
Adapted from BetterAttention folder.py for DyadicShelf LM (DyadicPush V2).

Features:
- Stream multi-dataset Parquet files without loading whole datasets into RAM.
- Temperature-adjusted weighted mixing: p_i ~ w_i^(1/T).
- Exact 4-stage ledger state machine: read -> consumed -> applied -> committed.
- Reconciles completed file logs with checkpoints across restarts.
- Auto-drops exhausted datasets and renormalizes sampling probabilities.
- Windows path-safe spec parsing (handles drive letters correctly).
"""

import glob
import json
import math
import os
import random
from collections import deque
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch

MIX_LOOKAHEAD = 8
LOG_VERSION = 1


def parse_dataset_specs(
    specs: List[str], temperature: float = 1.0, check_dirs: bool = True
) -> List[Tuple[str, str, float]]:
    """
    Parse NAME=PATH[:WEIGHT] specs -> [(name, path, weight)].
    The weight is split off with rsplit(':', 1) only if numeric, so Windows
    drive-letter paths (C:\\data\\x) parse correctly.
    """
    if not (isinstance(temperature, (int, float)) and math.isfinite(temperature) and temperature > 0):
        raise ValueError(f"--mix_temperature must be > 0 (got {temperature})")
    out, seen = [], set()
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"bad dataset spec '{spec}' (expected NAME=PATH[:WEIGHT])")
        name, rest = spec.split("=", 1)
        name, rest = name.strip(), rest.strip()
        if not name or not rest:
            raise ValueError(f"bad dataset spec '{spec}' (empty name or path)")
        path, weight = rest, 1.0
        if ":" in rest:
            head, tail = rest.rsplit(":", 1)
            try:
                w = float(tail)
            except ValueError:
                w = None
            if w is not None:
                path, weight = head, w
        if not path:
            raise ValueError(f"bad dataset spec '{spec}' (empty path)")
        if not (math.isfinite(weight) and weight > 0):
            raise ValueError(f"dataset '{name}': weight must be a finite number > 0 (got {weight})")
        if name in seen:
            raise ValueError(f"duplicate dataset name '{name}'")
        if check_dirs and not os.path.isdir(path):
            raise ValueError(f"dataset '{name}': directory not found: {path}")
        seen.add(name)
        out.append((name, path, weight))
    if not out:
        raise ValueError("no datasets given")
    return out


def parse_text_columns(arg: str) -> Tuple[str, Dict[str, str]]:
    """'col' -> (col, {}); 'a=c1,b=c2' -> ('text', {'a': 'c1', 'b': 'c2'})."""
    if not arg or "=" not in arg:
        return (arg.strip() if arg else "text") or "text", {}
    col_map = {}
    for item in arg.split(","):
        if not item.strip():
            continue
        if "=" not in item:
            raise ValueError(f"bad --text_column item '{item}' (expected name=col)")
        n, c = item.split("=", 1)
        col_map[n.strip()] = c.strip()
    return "text", col_map


def mix_probs(weights: List[float], temperature: float) -> np.ndarray:
    """Normalized sampling probabilities: w^(1/T) / sum."""
    p = np.array([w ** (1.0 / temperature) for w in weights], dtype=np.float64)
    return p / p.sum()


def file_key(path: str) -> Optional[Tuple[str, int, int]]:
    """Identity of a file version: (normalized abs path, size, mtime_ns); None if unreadable."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    return (os.path.normcase(os.path.abspath(path)), int(st.st_size), int(st.st_mtime_ns))


class Ledger:
    """
    Per-file state machine: read -> consumed -> applied -> committed.
    read      : first chunk of the file entered the mixer
    consumed  : last window of the file was emitted to a (not yet applied) micro-batch
    applied   : the optimizer step containing that window completed
    committed : a checkpoint holding that step (and the log) was written
    """

    def __init__(self, entries=()):
        self.completed: Dict[Tuple[str, int, int], dict] = {}
        for e in entries:
            self.completed[(e["path"], e["size"], e["mtime_ns"])] = dict(e)
        self.consumed: Dict[Tuple[str, int, int], dict] = {}
        self.applied: Dict[Tuple[str, int, int], dict] = {}
        self.read: Set[Tuple[str, int, int]] = set()
        self.failed: Set[Tuple[str, int, int]] = set()

    def is_done(self, key: Tuple[str, int, int]) -> bool:
        return key in self.completed

    def consume(self, key: Tuple[str, int, int], dataset: str, tokens: int):
        self.consumed[key] = {
            "path": key[0],
            "size": key[1],
            "mtime_ns": key[2],
            "dataset": dataset,
            "tokens": int(tokens),
        }

    def apply(self):
        for key in self.consumed:
            for store in (self.completed, self.applied):
                for old in [k for k in store if k[0] == key[0] and k != key]:
                    del store[old]
        self.applied.update(self.consumed)
        self.consumed.clear()

    def discard_unapplied(self) -> int:
        n = len(self.consumed)
        self.consumed.clear()
        return n

    def snapshot(self) -> List[dict]:
        return list(self.completed.values()) + list(self.applied.values())

    def commit(self):
        self.completed.update(self.applied)
        self.applied.clear()

    def done_count(self, dataset: str) -> int:
        return sum(1 for e in self.snapshot() if e["dataset"] == dataset)


def _valid_entry(e) -> bool:
    return (
        isinstance(e, dict)
        and isinstance(e.get("path"), str)
        and isinstance(e.get("size"), int)
        and isinstance(e.get("mtime_ns"), int)
        and isinstance(e.get("dataset"), str)
        and isinstance(e.get("tokens"), int)
    )


def load_completed_log(path: str) -> Optional[Tuple[int, List[dict]]]:
    """-> (committed_step, entries) or None (missing/corrupt; corrupt file is renamed *.corrupt)."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if (
            not isinstance(obj, dict)
            or obj.get("version") != LOG_VERSION
            or not isinstance(obj.get("committed_step"), int)
            or not isinstance(obj.get("files"), list)
            or not all(_valid_entry(e) for e in obj["files"])
        ):
            raise ValueError("schema mismatch")
        return obj["committed_step"], obj["files"]
    except (OSError, ValueError) as e:
        print(f"[!] Completed-files log '{path}' is unreadable ({e}); treating as empty.")
        try:
            os.replace(path, path + ".corrupt")
            print(f"    (moved to '{path}.corrupt')")
        except OSError:
            pass
        return None


def write_completed_log(path: str, entries: List[dict], step: int):
    """Atomic write (tmp + fsync + os.replace)."""
    obj = {
        "version": LOG_VERSION,
        "committed_step": int(step),
        "files": sorted((dict(e) for e in entries), key=lambda e: e["path"]),
    }
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def reconcile_completed(args, ckpt: Optional[dict], log_path: str) -> List[dict]:
    """
    Decide the completed-files list at startup. Checkpoint is authoritative;
    the log mirrors it (and is repaired from it). Returns a list of entries.
    """
    if getattr(args, "reset_completed", False):
        print("[*] --reset_completed: ignoring completed-files list (rewritten at next save).")
        return []
    log = load_completed_log(log_path)
    if ckpt is None:
        if log is not None:
            print(
                f"[!] Stale completed-files log '{log_path}' ignored: no checkpoint to resume from "
                "(it will be overwritten at the first save)."
            )
        return []
    ck_step = int(ckpt.get("step", ckpt.get("global_step", 0)))
    ck_files = ckpt.get("completed_files")
    if isinstance(ck_files, list) and all(_valid_entry(e) for e in ck_files):
        ck_keys = {(e["path"], e["size"], e["mtime_ns"]) for e in ck_files}
        log_keys = {(e["path"], e["size"], e["mtime_ns"]) for e in log[1]} if log else None
        if log is None or log[0] != ck_step or log_keys != ck_keys:
            print(
                f"[!] Completed-files log disagrees with checkpoint (step {ck_step}); "
                f"repairing log from checkpoint ({len(ck_files)} files)."
            )
            write_completed_log(log_path, ck_files, ck_step)
        return list(ck_files)
    if log is not None and log[0] == ck_step:
        print(f"[*] Checkpoint has no embedded file list; using log '{log_path}' (step {ck_step}).")
        return list(log[1])
    if log is not None:
        print(f"[!] Log step {log[0]} != checkpoint step {ck_step} and checkpoint has no file list; log ignored.")
    return []


class _Chunk:
    __slots__ = ("dataset", "key", "arr", "last", "file_tokens")

    def __init__(self, dataset: str, key: Tuple[str, int, int], arr: np.ndarray, last: bool, file_tokens: int):
        self.dataset = dataset
        self.key = key
        self.arr = arr
        self.last = last
        self.file_tokens = file_tokens


def iter_file_chunks(path: str, column: str, enc, chunk_len: int):
    """
    Stream one parquet file -> (uint16/uint32 array, is_last).
    Chunks never span files; the final chunk is held back to be flagged last.
    Documents are separated by enc.eot_token.
    """
    import pyarrow.parquet as pq

    eot = getattr(enc, "eot_token", 0)
    dtype = np.uint32 if getattr(enc, "vocab_size", 0) > 65535 else np.uint16
    pf = pq.ParquetFile(path)
    try:
        names = pf.schema_arrow.names
        if column not in names:
            raise KeyError(f"column '{column}' not found (available: {names})")
        buf = np.empty(0, dtype=dtype)
        held = None
        for batch in pf.iter_batches(batch_size=500, columns=[column]):
            for text in batch.column(0).to_pylist():
                if text is None:
                    continue
                if hasattr(enc, "encode_ordinary"):
                    encoded = enc.encode_ordinary(str(text))
                else:
                    encoded = enc.encode(str(text))
                ids = np.asarray(encoded + [eot], dtype=dtype)
                buf = ids if buf.size == 0 else np.concatenate([buf, ids])
                pos = 0
                while buf.size - pos >= chunk_len:
                    arr = buf[pos : pos + chunk_len].copy()
                    pos += chunk_len
                    if held is not None:
                        yield held, False
                    held = arr
                if pos:
                    buf = buf[pos:].copy()
        if buf.size:
            if held is not None:
                yield held, False
            held = buf
        if held is not None:
            yield held, True
    finally:
        close = getattr(pf, "close", None)
        if close:
            close()


class DatasetMixer:
    """
    Weighted multi-dataset chunk mixer with an exact consumption ledger.
    prob_i = w_i^(1/T) / sum_j w_j^(1/T) over datasets that still have data.
    """

    def __init__(
        self,
        specs: List[Tuple[str, str, float]],
        enc,
        seq_len: int,
        mix_chunk: int,
        temperature: float,
        default_col: str,
        col_map: Dict[str, str],
        ledger: Ledger,
        seed: int = 1337,
        lookahead: int = MIX_LOOKAHEAD,
    ):
        self.enc = enc
        self.seq_len = seq_len
        self.ledger = ledger
        self.chunk_len = max(1, mix_chunk // seq_len) * seq_len + 1
        self.temperature = temperature
        self.lookahead = lookahead
        self.rng = np.random.default_rng(seed)
        self.weights, self.queues, self.files, self.gens, self.cur = {}, {}, {}, {}, {}
        self.cols, self.files_total, self.files_skipped = {}, {}, {}
        self.tokens_sampled, self.chunks_sampled, self.batches_sampled = {}, {}, {}
        self.active, self.dropped = [], []
        self._cur = None

        shuf = random.Random(seed)
        for name, path, w in specs:
            self.weights[name] = w
            self.cols[name] = col_map.get(name, default_col)
            self.tokens_sampled[name] = self.chunks_sampled[name] = self.batches_sampled[name] = 0
            found = sorted(glob.glob(os.path.join(glob.escape(path), "**", "*.parquet"), recursive=True))
            shuf.shuffle(found)
            todo, skipped = deque(), 0
            for fp in found:
                key = file_key(fp)
                if key is None:
                    print(f"[!] {name}: cannot stat '{fp}', skipped.")
                elif ledger.is_done(key):
                    skipped += 1
                else:
                    todo.append((fp, key))
            self.files_total[name], self.files_skipped[name] = len(found), skipped
            if not todo:
                why = (
                    "no parquet files found"
                    if not found
                    else f"all {len(found)} files already completed"
                )
                print(f"[*] Dataset '{name}' dropped: {why}.")
                self.dropped.append(name)
                continue
            if skipped:
                print(f"[*] Dataset '{name}': skipping {skipped} completed file(s); {len(todo)} to train.")
            self.files[name], self.queues[name] = todo, deque()
            self.gens[name], self.cur[name] = None, None
            self.active.append(name)
        self.target = (
            dict(zip(self.active, mix_probs([self.weights[n] for n in self.active], temperature)))
            if self.active
            else {}
        )

    def _fill(self, ds: str):
        q = self.queues[ds]
        while len(q) < self.lookahead:
            if self.gens[ds] is None:
                if not self.files[ds]:
                    return
                path, key = self.files[ds].popleft()
                self.cur[ds] = {"path": path, "key": key, "tokens": 0, "any": False}
                self.gens[ds] = iter_file_chunks(path, self.cols[ds], self.enc, self.chunk_len)
            info = self.cur[ds]
            try:
                arr, last = next(self.gens[ds])
            except StopIteration:
                self.gens[ds] = None
                if not info["any"]:
                    print(f"[!] {ds}: '{info['path']}' has no text; marked complete (0 tokens).")
                    self.ledger.consume(info["key"], ds, 0)
                continue
            except Exception as e:
                print(f"[!] {ds}: skipping '{info['path']}': {type(e).__name__}: {e}")
                self.ledger.failed.add(info["key"])
                self.gens[ds] = None
                continue
            if not info["any"]:
                self.ledger.read.add(info["key"])
            info["any"] = True
            info["tokens"] += len(arr)
            q.append(_Chunk(ds, info["key"], arr, last, info["tokens"] if last else 0))
            if last:
                self.gens[ds] = None

    def _active_with_data(self) -> List[str]:
        for ds in list(self.active):
            self._fill(ds)
            if not self.queues[ds]:
                print(f"[*] Dataset '{ds}' exhausted; dropping it and renormalizing mixing probabilities.")
                self.active.remove(ds)
                self.dropped.append(ds)
        return self.active

    def _next_chunk(self) -> Optional[_Chunk]:
        names = self._active_with_data()
        if not names:
            return None
        probs = mix_probs([self.weights[n] for n in names], self.temperature)
        ds = names[int(self.rng.choice(len(names), p=probs))]
        ch = self.queues[ds].popleft()
        self.tokens_sampled[ds] += len(ch.arr)
        self.chunks_sampled[ds] += 1
        return ch

    def _finish(self, ch: _Chunk):
        self._cur = None
        if ch.last:
            self.ledger.consume(ch.key, ch.dataset, ch.file_tokens)

    def next_window(self) -> Optional[Tuple[np.ndarray, str]]:
        """-> (array of seq_len+1 tokens, dataset) or None when all data is consumed."""
        while True:
            if self._cur is not None:
                ch, off = self._cur
                if off + self.seq_len + 1 <= len(ch.arr):
                    w = ch.arr[off : off + self.seq_len + 1]
                    off += self.seq_len
                    self._cur = (ch, off)
                    if off + self.seq_len + 1 > len(ch.arr):
                        self._finish(ch)
                    return w, ch.dataset
                self._finish(ch)
            ch = self._next_chunk()
            if ch is None:
                return None
            self._cur = (ch, 0)

    def get_batch(self, batch_size: int, device: str) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """-> (x, y) int64 tensors, or None if exhausted."""
        ws = []
        for _ in range(batch_size):
            r = self.next_window()
            if r is None:
                break
            ws.append(r[0])
            self.batches_sampled[r[1]] += 1
        if not ws:
            return None
        a = torch.from_numpy(np.stack(ws).astype(np.int64))
        return a[:, :-1].contiguous().to(device), a[:, 1:].contiguous().to(device)

    def stats_state(self) -> dict:
        return {
            "tokens_sampled": dict(self.tokens_sampled),
            "chunks_sampled": dict(self.chunks_sampled),
            "batches_sampled": dict(self.batches_sampled),
        }

    def load_stats(self, st: Optional[dict]):
        if not isinstance(st, dict):
            return
        for k in ("tokens_sampled", "chunks_sampled", "batches_sampled"):
            for n, v in (st.get(k) or {}).items():
                if n in getattr(self, k) and isinstance(v, int):
                    getattr(self, k)[n] += v


def print_mix_stats(mixer: DatasetMixer, ledger: Ledger, step: int):
    tot = max(1, sum(mixer.tokens_sampled.values()))
    print(f"\n[Mixer stats @ step {step}] (target = normalized w^(1/T) over datasets present at start)")
    print(f"  {'dataset':<14}{'files done':>12}{'tokens':>14}{'share%':>9}{'target%':>9}{'drift':>8}{'windows':>10}")
    for n in mixer.weights:
        share = 100.0 * mixer.tokens_sampled[n] / tot
        tgt = 100.0 * mixer.target.get(n, 0.0)
        print(
            f"  {n:<14}{ledger.done_count(n):>6}/{mixer.files_total[n]:<5}{mixer.tokens_sampled[n]:>14,}"
            f"{share:>9.1f}{tgt:>9.1f}{share - tgt:>+8.1f}{mixer.batches_sampled[n]:>10,}"
        )
    print(f"  failed files: {len(ledger.failed)} | active datasets: {', '.join(mixer.active) or '-'}\n")
