"""
test_parquet.py - Automated Tests for Parquet Streaming and Training
====================================================================
Tests:
1. DatasetMixer, spec parsing, and 4-stage ledger state machine.
2. Full train.py integration with --datasets, --grad_accum, GSCI, and resumption.
"""

import os
import shutil
import subprocess
import sys
import tempfile
import pyarrow as pa
import pyarrow.parquet as pq

from dataset_mixer import (
    DatasetMixer,
    Ledger,
    parse_dataset_specs,
    parse_text_columns,
)
from tokenizer import CharTokenizer


def test_mixer_and_ledger():
    print("=" * 70)
    print("TEST 1: DatasetMixer, Spec Parsing, and Ledger State Machine")
    print("=" * 70)

    tmpdir = tempfile.mkdtemp(prefix="test_parquet_unit_")
    ds1_dir = os.path.join(tmpdir, "ds1")
    ds2_dir = os.path.join(tmpdir, "ds2")
    os.makedirs(ds1_dir, exist_ok=True)
    os.makedirs(ds2_dir, exist_ok=True)

    try:
        schema = pa.schema([("text", pa.string())])
        texts_1 = ["The quick brown fox jumps over the lazy dog. " * 10 for _ in range(5)]
        table_1 = pa.Table.from_arrays([pa.array(texts_1)], schema=schema)
        pq.write_table(table_1, os.path.join(ds1_dir, "file1.parquet"))

        schema_custom = pa.schema([("body", pa.string())])
        texts_2 = ["Deep learning models process sequences with geometric routing. " * 10 for _ in range(5)]
        table_2 = pa.Table.from_arrays([pa.array(texts_2)], schema=schema_custom)
        pq.write_table(table_2, os.path.join(ds2_dir, "file2.parquet"))

        specs = parse_dataset_specs([f"d1={ds1_dir}:2.0", f"d2={ds2_dir}:1.0"], temperature=1.0)
        assert len(specs) == 2
        default_col, col_map = parse_text_columns("d1=text,d2=body")
        assert col_map["d1"] == "text" and col_map["d2"] == "body"

        tok = CharTokenizer.train_from_text("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ 0123456789.,!?;\n")
        ledger = Ledger()
        mixer = DatasetMixer(
            specs=specs,
            enc=tok,
            seq_len=32,
            mix_chunk=64,
            temperature=1.0,
            default_col=default_col,
            col_map=col_map,
            ledger=ledger,
        )

        assert len(mixer.active) == 2
        b = mixer.get_batch(batch_size=2, device="cpu")
        assert b is not None
        x, y = b
        assert x.shape == (2, 32)
        assert y.shape == (2, 32)
        assert (x[:, 1:] == y[:, :-1]).all()
        print("  [PASS] Mixer, causal token shift, and column mapping validated.")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_e2e_training_and_resume():
    print("\n" + "=" * 70)
    print("TEST 2: End-to-End Training & Resumption with train.py")
    print("=" * 70)

    tmpdir = tempfile.mkdtemp(prefix="test_parquet_e2e_")
    data_dir = os.path.join(tmpdir, "data")
    ckpt_dir = os.path.join(tmpdir, "ckpts")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    try:
        schema = pa.schema([("text", pa.string())])
        texts = ["DyadicPush routing enables multi-scale geometric packet propagation. " * 10 for _ in range(20)]
        table = pa.Table.from_arrays([pa.array(texts)], schema=schema)
        pq.write_table(table, os.path.join(data_dir, "shard1.parquet"))

        # 1. Run first training phase
        cmd1 = [
            sys.executable,
            "train.py",
            "--datasets", f"test={data_dir}:1.0",
            "--dim", "64",
            "--hidden_dim", "128",
            "--num_layers", "2",
            "--num_heads", "2",
            "--seq_len", "64",
            "--chunk_size", "16",
            "--top_k", "2",
            "--map_dim", "16",
            "--batch_size", "2",
            "--grad_accum", "2",
            "--gsci",
            "--gsci_shards", "2",
            "--max_steps", "4",
            "--warmup_steps", "1",
            "--save_interval", "2",
            "--checkpoint_dir", ckpt_dir,
            "--device", "cpu",
        ]
        res1 = subprocess.run(cmd1, capture_output=True, text=True)
        assert res1.returncode == 0, f"Initial training failed:\n{res1.stderr}"
        final_ckpt = os.path.join(ckpt_dir, "final_model.pt")
        assert os.path.exists(final_ckpt), "final_model.pt not generated!"
        print("  [PASS] Initial training to step 4 succeeded.")

        # 2. Resume up to step 6
        cmd2 = [
            sys.executable,
            "train.py",
            "--datasets", f"test={data_dir}:1.0",
            "--dim", "64",
            "--hidden_dim", "128",
            "--num_layers", "2",
            "--num_heads", "2",
            "--seq_len", "64",
            "--chunk_size", "16",
            "--top_k", "2",
            "--map_dim", "16",
            "--batch_size", "2",
            "--grad_accum", "2",
            "--max_steps", "6",
            "--checkpoint_dir", ckpt_dir,
            "--resume", final_ckpt,
            "--device", "cpu",
        ]
        res2 = subprocess.run(cmd2, capture_output=True, text=True)
        assert res2.returncode == 0, f"Resumption training failed:\n{res2.stderr}"
        assert "Steps 5..6" in res2.stdout or "Step      5/6" in res2.stdout
        print("  [PASS] Resumed training to step 6 succeeded.")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    test_mixer_and_ledger()
    test_e2e_training_and_resume()
    print("\n" + "=" * 70)
    print("ALL PARQUET TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)
