#!/usr/bin/env python3
"""Single-process precompute of a WING train dataset_stats.json.

Pinning the result via `pretrained_norm_stats` lets multi-GPU training skip the
runtime stats computation and avoids a multi-rank read/write race on
dataset_stats.json that can crash distributed launches with a JSONDecodeError.

Runs on CPU (only reads action/state from the parquet to compute min/max/q01/q99/
mean/std). Writes <out_dir>/dataset_stats.json where out_dir is the parent of --out.

Usage:
    python scripts/data_process/precompute_dataset_stats.py \
        --data robotwin \
        --out  data/robotwin2.0/adjust_bottle_aloha_randomized/dataset_stats.json
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # scripts/data_process/<file> -> wing
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


def statistics_dataset_config(config):
    from omegaconf import open_dict

    dataset_config = config.data.train
    while "base_dataset" in dataset_config:
        dataset_config = dataset_config.base_dataset
    with open_dict(dataset_config):
        dataset_config.pretrained_norm_stats = None
        dataset_config.text_embedding_cache_dir = None
        dataset_config.vae_latent_cache_dir = None
        dataset_config.vae_latent_cache_required = False
    return dataset_config


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="data config name under configs/data (no .yaml).")
    ap.add_argument("--out", required=True, help="Target dataset_stats.json path (its parent dir becomes work_dir).")
    args = ap.parse_args()

    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate

    from wing.utils import misc
    from wing.utils.config_resolvers import register_default_resolvers

    register_default_resolvers()

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # RobotVideoDataset saves stats to misc.get_work_dir()/dataset_stats.json (rank0).
    misc.register_work_dir(str(out_path.parent))

    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        # Force a fresh computation: the data config may already pin
        # pretrained_norm_stats to the file this script is about to create
        # (chicken-and-egg on first run).
        cfg = compose(
            config_name="train",
            overrides=[f"data={args.data}"],
        )

    print(f"[stats] instantiating train dataset (data={args.data}); this computes stats...", flush=True)
    ds = instantiate(statistics_dataset_config(cfg))
    print(f"[stats] train dataset len={len(ds)}", flush=True)

    saved = out_path.parent / "dataset_stats.json"
    if not saved.is_file():
        raise SystemExit(f"[stats] FAILED: expected {saved} to be written but it is missing")
    if saved != out_path:
        saved.replace(out_path)
    print(f"[stats] OK wrote {out_path} ({out_path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
