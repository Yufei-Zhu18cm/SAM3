import argparse
import json
import os
import shutil
from copy import deepcopy
from datetime import datetime
from pathlib import Path

from eventbook.event_pipeline import load_yaml, train_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="eventbook/config_h2o2101.yaml")
    parser.add_argument("--tag", default="", help="Optional suffix added to checkpoint filename")
    parser.add_argument("--checkpoint_dir", default="", help="Override checkpoint output directory")
    return parser.parse_args()


def normalize_name(s: str) -> str:
    s = str(s).strip()
    keep = []
    for ch in s:
        if ch.isalnum() or ch in ["-", "_", "."]:
            keep.append(ch)
        else:
            keep.append("_")
    out = "".join(keep).strip("_")
    return out or "model"


def build_paths(cfg: dict, tag: str = "", checkpoint_dir: str = ""):
    base_save_path = cfg["model"]["save_path"]
    base_dir = checkpoint_dir if checkpoint_dir else os.path.dirname(base_save_path)
    base_dir = base_dir or "."
    os.makedirs(base_dir, exist_ok=True)

    stem = Path(base_save_path).stem
    suffix = Path(base_save_path).suffix or ".joblib"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = normalize_name(tag) if tag else ""

    version_name = f"{stem}_{timestamp}{'_' + tag if tag else ''}{suffix}"
    version_path = os.path.join(base_dir, version_name)
    latest_path = os.path.join(base_dir, "latest.joblib")
    meta_path = os.path.join(base_dir, f"{stem}_{timestamp}{'_' + tag if tag else ''}.meta.json")
    log_path = os.path.join(base_dir, "train_log.jsonl")
    return version_path, latest_path, meta_path, log_path, timestamp, stem, suffix


def extract_dataset_list(cfg: dict):
    if "train_videos" in cfg and isinstance(cfg["train_videos"], list):
        return [str(x) for x in cfg["train_videos"]]
    if "train_datasets" in cfg and isinstance(cfg["train_datasets"], list):
        names = []
        for item in cfg["train_datasets"]:
            if isinstance(item, dict):
                names.append(str(item.get("name", item)))
            else:
                names.append(str(item))
        return names
    return []


def update_latest(version_path: str, latest_path: str):
    latest = Path(latest_path)
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    try:
        latest.symlink_to(Path(version_path).resolve())
        return "symlink"
    except Exception:
        shutil.copy2(version_path, latest_path)
        return "copy"


def main():
    args = parse_args()
    print("[1/5] Loading config...")
    cfg = load_yaml(args.config)
    cfg_run = deepcopy(cfg)

    version_path, latest_path, meta_path, log_path, timestamp, stem, suffix = build_paths(
        cfg_run, tag=args.tag, checkpoint_dir=args.checkpoint_dir
    )
    cfg_run["model"]["save_path"] = version_path

    progress_cfg = cfg_run.setdefault("progress", {})
    progress_cfg.setdefault("enabled", True)
    progress_cfg.setdefault("every_n_frames", 50)

    datasets = extract_dataset_list(cfg_run)
    print(f"[2/5] Training will use {len(datasets)} dataset(s): {datasets}")
    print(f"[3/5] Checkpoint will be saved to: {version_path}")
    print("[4/5] Building training table and fitting model...")
    model, df = train_model(cfg_run)

    print("[5/5] Writing metadata and updating latest checkpoint...")
    latest_mode = update_latest(version_path, latest_path)
    positive = int(df["label"].sum()) if "label" in df.columns else None
    total = int(len(df))
    negative = int(total - positive) if positive is not None else None

    meta = {
        "timestamp": timestamp,
        "config_path": os.path.abspath(args.config),
        "checkpoint_path": os.path.abspath(version_path),
        "latest_path": os.path.abspath(latest_path),
        "latest_mode": latest_mode,
        "datasets": datasets,
        "num_datasets": len(datasets),
        "num_samples": total,
        "num_positive": positive,
        "num_negative": negative,
        "tag": args.tag,
        "model_base_name": stem,
        "model_suffix": suffix,
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(meta, ensure_ascii=False) + "")

    print("Training done.")
    print(f"Samples: {total}, positive: {positive}, negative: {negative}")
    print(f"Version checkpoint: {version_path}")
    print(f"Latest checkpoint: {latest_path} ({latest_mode})")
    print(f"Meta saved: {meta_path}")
    print(f"Train log: {log_path}")


if __name__ == "__main__":
    main()
