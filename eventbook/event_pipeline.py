from __future__ import annotations
import os
import re
import glob
import math
import joblib
import numpy as np
import pandas as pd
from PIL import Image
from typing import Dict, List, Tuple, Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

try:
    import yaml
except ImportError:
    yaml = None


FEATURE_COLUMNS = [
    "kind_split",
    "kind_merge",
    "n_src",
    "n_dst",
    "src_area_sum",
    "dst_area_sum",
    "area_ratio",
    "max_intersection",
    "mean_intersection",
    "mean_move_dist",
    "max_move_dist",
    "src_bbox_area",
    "dst_bbox_area_sum",
]


def load_yaml(path: str) -> dict:
    if yaml is None:
        raise ImportError("Please install pyyaml: pip install pyyaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def progress_enabled(cfg: dict) -> bool:
    return bool(cfg.get("progress", {}).get("enabled", False))


def progress_every_n_frames(cfg: dict) -> int:
    return max(1, int(cfg.get("progress", {}).get("every_n_frames", 50)))


def pmsg(cfg: dict, msg: str):
    if progress_enabled(cfg):
        print(msg, flush=True)


def parse_frame_range(x: Any) -> Tuple[int, int]:
    s = str(x).strip()
    nums = re.findall(r"\d+", s)
    if len(nums) >= 2:
        return int(nums[0]), int(nums[1])
    if len(nums) == 1:
        n = int(nums[0])
        return n, n
    raise ValueError(f"Cannot parse frame range from: {x}")


def parse_bracket_ids(x: Any) -> List[str]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    s = str(x).strip()
    if not s:
        return []
    ids = re.findall(r"\[([^\[\]]+)\]", s)
    if ids:
        return [i.strip() for i in ids if i.strip()]
    return [s]


def expand_id_token(token: str) -> set:
    token = str(token).strip()
    out = {token}
    m = re.match(r"^(\d+)-(\d+)$", token)
    if m:
        out.add(m.group(1))
    m2 = re.match(r"^(\d+)\.0$", token)
    if m2:
        out.add(m2.group(1))
    return out


def ids_overlap(a: List[str], b: List[str]) -> bool:
    if not a or not b:
        return True
    sa = set()
    sb = set()
    for x in a:
        sa |= expand_id_token(x)
    for x in b:
        sb |= expand_id_token(x)
    return len(sa & sb) > 0


def normalize_event_type(x: Any) -> str:
    s = str(x).strip().lower()
    if s in ["merge", "sinter", "烧结", "合并"]:
        return "merge"
    if s in ["split", "裂分", "分裂"]:
        return "split"
    return s


def load_annotations_for_video(ann_path: str) -> pd.DataFrame:
    df = pd.read_excel(ann_path)
    required = ["Frame", "Event Type", "Parent IDs", "Child IDs", "Note"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"{ann_path} missing required column: {c}")

    rows = []
    for _, r in df.iterrows():
        start_frame, end_frame = parse_frame_range(r["Frame"])
        rows.append({
            "Frame": str(r["Frame"]).strip(),
            "Event Type": normalize_event_type(r["Event Type"]),
            "Parent IDs": parse_bracket_ids(r["Parent IDs"]),
            "Child IDs": parse_bracket_ids(r["Child IDs"]),
            "Note": "" if pd.isna(r["Note"]) else str(r["Note"]),
            "start_frame": start_frame,
            "end_frame": end_frame,
        })
    return pd.DataFrame(rows)


def parse_frame_num_from_name(path: str) -> int:
    name = os.path.basename(path)
    m = re.match(r"^(\d+)", name)
    if not m:
        raise ValueError(f"Cannot parse frame number from filename: {name}")
    return int(m.group(1))


def load_mask_sequence(mask_dir: str) -> Tuple[List[np.ndarray], List[str], List[int]]:
    files = glob.glob(os.path.join(mask_dir, "*_tracks.tif"))
    if not files:
        raise FileNotFoundError(f"No *_tracks.tif found in {mask_dir}")
    files = sorted(files, key=lambda p: parse_frame_num_from_name(p))

    masks = []
    frame_nums = []
    for f in files:
        arr = np.array(Image.open(f))
        if arr.ndim == 3:
            arr = arr[..., 0]
        masks.append(arr.astype(np.int32))
        frame_nums.append(parse_frame_num_from_name(f))

    return masks, files, frame_nums


def region_stats(mask: np.ndarray, min_area: int = 1) -> Dict[int, dict]:
    ids, counts = np.unique(mask, return_counts=True)
    out = {}
    for obj_id, area in zip(ids, counts):
        if int(obj_id) == 0 or int(area) < min_area:
            continue
        ys, xs = np.where(mask == obj_id)
        if len(xs) == 0:
            continue
        y0, y1 = int(ys.min()), int(ys.max())
        x0, x1 = int(xs.min()), int(xs.max())
        out[int(obj_id)] = {
            "area": int(area),
            "centroid_y": float(ys.mean()),
            "centroid_x": float(xs.mean()),
            "bbox_area": int((y1 - y0 + 1) * (x1 - x0 + 1)),
        }
    return out


def pair_overlap(prev_mask: np.ndarray, next_mask: np.ndarray) -> Dict[Tuple[int, int], int]:
    valid = (prev_mask > 0) & (next_mask > 0)
    if valid.sum() == 0:
        return {}
    pairs = np.stack([prev_mask[valid], next_mask[valid]], axis=1)
    uniq, cnts = np.unique(pairs, axis=0, return_counts=True)
    return {(int(a), int(b)): int(c) for (a, b), c in zip(uniq, cnts)}


def center_distance(a: dict, b: dict) -> float:
    return float(math.sqrt((a["centroid_x"] - b["centroid_x"]) ** 2 + (a["centroid_y"] - b["centroid_y"]) ** 2))


def generate_candidates_for_transition(prev_mask, next_mask, prev_abs_frame, next_abs_frame, feat_cfg):
    prev_stats = region_stats(prev_mask, min_area=feat_cfg["min_region_area"])
    next_stats = region_stats(next_mask, min_area=feat_cfg["min_region_area"])
    overlaps = pair_overlap(prev_mask, next_mask)

    parent_to_children = {}
    child_to_parents = {}

    for (p, c), inter in overlaps.items():
        if inter < feat_cfg["min_overlap_pixels"]:
            continue
        parent_to_children.setdefault(p, []).append((c, inter))
        child_to_parents.setdefault(c, []).append((p, inter))

    candidates = []

    for p, lst in parent_to_children.items():
        if p not in prev_stats or len(lst) < feat_cfg["split_min_children"]:
            continue
        children = [c for c, inter in sorted(lst, key=lambda x: x[1], reverse=True) if c in next_stats]
        if len(children) < feat_cfg["split_min_children"]:
            continue

        src_area = prev_stats[p]["area"]
        dst_areas = [next_stats[c]["area"] for c in children]
        total_dst_area = sum(dst_areas)
        area_ratio = total_dst_area / max(src_area, 1)
        if not (feat_cfg["area_ratio_low"] <= area_ratio <= feat_cfg["area_ratio_high"]):
            continue

        inters = [inter for c, inter in lst if c in next_stats]
        dists = [center_distance(prev_stats[p], next_stats[c]) for c in children]

        candidates.append({
            "prev_abs_frame": int(prev_abs_frame),
            "next_abs_frame": int(next_abs_frame),
            "event_type": "split",
            "src_ids": [str(p)],
            "dst_ids": [str(c) for c in children],
            "n_src": 1,
            "n_dst": len(children),
            "src_area_sum": float(src_area),
            "dst_area_sum": float(total_dst_area),
            "area_ratio": float(area_ratio),
            "max_intersection": float(max(inters) if inters else 0),
            "mean_intersection": float(np.mean(inters) if inters else 0),
            "mean_move_dist": float(np.mean(dists) if dists else 0),
            "max_move_dist": float(np.max(dists) if dists else 0),
            "src_bbox_area": float(prev_stats[p]["bbox_area"]),
            "dst_bbox_area_sum": float(sum(next_stats[c]["bbox_area"] for c in children)),
        })

    for c, lst in child_to_parents.items():
        if c not in next_stats or len(lst) < feat_cfg["merge_min_parents"]:
            continue
        parents = [p for p, inter in sorted(lst, key=lambda x: x[1], reverse=True) if p in prev_stats]
        if len(parents) < feat_cfg["merge_min_parents"]:
            continue

        dst_area = next_stats[c]["area"]
        src_areas = [prev_stats[p]["area"] for p in parents]
        total_src_area = sum(src_areas)
        area_ratio = dst_area / max(total_src_area, 1)
        if not (1.0 / feat_cfg["area_ratio_high"] <= area_ratio <= 1.0 / feat_cfg["area_ratio_low"]):
            continue

        inters = [inter for p, inter in lst if p in prev_stats]
        dists = [center_distance(prev_stats[p], next_stats[c]) for p in parents]

        candidates.append({
            "prev_abs_frame": int(prev_abs_frame),
            "next_abs_frame": int(next_abs_frame),
            "event_type": "merge",
            "src_ids": [str(p) for p in parents],
            "dst_ids": [str(c)],
            "n_src": len(parents),
            "n_dst": 1,
            "src_area_sum": float(total_src_area),
            "dst_area_sum": float(dst_area),
            "area_ratio": float(area_ratio),
            "max_intersection": float(max(inters) if inters else 0),
            "mean_intersection": float(np.mean(inters) if inters else 0),
            "mean_move_dist": float(np.mean(dists) if dists else 0),
            "max_move_dist": float(np.max(dists) if dists else 0),
            "src_bbox_area": float(sum(prev_stats[p]["bbox_area"] for p in parents)),
            "dst_bbox_area_sum": float(next_stats[c]["bbox_area"]),
        })

    return candidates


def candidate_to_feature_row(cand: dict) -> dict:
    return {
        "kind_split": 1 if cand["event_type"] == "split" else 0,
        "kind_merge": 1 if cand["event_type"] == "merge" else 0,
        "n_src": cand["n_src"],
        "n_dst": cand["n_dst"],
        "src_area_sum": cand["src_area_sum"],
        "dst_area_sum": cand["dst_area_sum"],
        "area_ratio": cand["area_ratio"],
        "max_intersection": cand["max_intersection"],
        "mean_intersection": cand["mean_intersection"],
        "mean_move_dist": cand["mean_move_dist"],
        "max_move_dist": cand["max_move_dist"],
        "src_bbox_area": cand["src_bbox_area"],
        "dst_bbox_area_sum": cand["dst_bbox_area_sum"],
    }


def candidate_to_sheet_columns(cand: dict) -> Tuple[List[str], List[str]]:
    if cand["event_type"] == "split":
        return cand["src_ids"], cand["dst_ids"]
    return cand["dst_ids"], cand["src_ids"]


def candidate_frame_range(cand: dict) -> Tuple[int, int, str]:
    start_frame = int(cand["prev_abs_frame"])
    end_frame = int(cand["next_abs_frame"])
    return start_frame, end_frame, f"{start_frame}--{end_frame}"


def candidate_note(cand: dict) -> str:
    if cand["event_type"] == "split":
        n = len(cand["dst_ids"])
        if n == 2:
            return "A particle splits into two"
        return f"A particle splits into {n}"
    n = len(cand["src_ids"])
    if n == 2:
        return "Two particles merge into one"
    return f"{n} particles merge into one"


def match_candidate_to_annotation(cand: dict, ann_df: pd.DataFrame) -> int:
    st, ed, _ = candidate_frame_range(cand)
    cand_parent_ids, cand_child_ids = candidate_to_sheet_columns(cand)

    same_type = ann_df["Event Type"] == cand["event_type"]
    frame_hit = (ann_df["start_frame"] <= ed) & (ann_df["end_frame"] >= st)
    sub = ann_df[same_type & frame_hit]
    if len(sub) == 0:
        return 0

    for _, row in sub.iterrows():
        ann_parent = row["Parent IDs"]
        ann_child = row["Child IDs"]
        if ids_overlap(cand_parent_ids, ann_parent) and ids_overlap(cand_child_ids, ann_child):
            return 1
    return 1


def build_training_table(cfg: dict) -> pd.DataFrame:
    ann_dir = cfg["data"]["annotation_dir"]
    mask_root = cfg["data"]["mask_root"]
    rows = []
    videos = cfg["train_videos"]
    total_videos = len(videos)
    every_n = progress_every_n_frames(cfg)

    pmsg(cfg, f"[build] Start building training table for {total_videos} video(s)")

    for video_idx, video_name in enumerate(videos, start=1):
        ann_path = os.path.join(ann_dir, f"{video_name}.xlsx")
        mask_dir = os.path.join(mask_root, video_name)
        pmsg(cfg, f"[build][{video_idx}/{total_videos}] Loading annotations: {ann_path}")
        ann_df = load_annotations_for_video(ann_path)
        pmsg(cfg, f"[build][{video_idx}/{total_videos}] Loading masks: {mask_dir}")
        masks, tif_files, frame_nums = load_mask_sequence(mask_dir)
        total_steps = max(0, len(masks) - 1)
        pmsg(cfg, f"[build][{video_idx}/{total_videos}] {video_name}: {len(ann_df)} annotations, {len(masks)} masks, {total_steps} transitions")

        before_rows = len(rows)
        for i in range(len(masks) - 1):
            if i == 0 or (i + 1) % every_n == 0 or i == len(masks) - 2:
                pmsg(cfg, f"[build][{video_idx}/{total_videos}] {video_name}: processing transition {i + 1}/{total_steps} ({frame_nums[i]} -> {frame_nums[i + 1]})")
            cands = generate_candidates_for_transition(
                masks[i], masks[i + 1], frame_nums[i], frame_nums[i + 1], cfg["feature"]
            )
            for cand in cands:
                feat = candidate_to_feature_row(cand)
                parent_ids, child_ids = candidate_to_sheet_columns(cand)
                st, ed, frame_text = candidate_frame_range(cand)
                rows.append({
                    "Video": video_name,
                    "Frame": frame_text,
                    "Event Type": cand["event_type"],
                    "Parent IDs": "".join([f"[{x}]" for x in parent_ids]),
                    "Child IDs": "".join([f"[{x}]" for x in child_ids]),
                    "Note": candidate_note(cand),
                    "start_frame": st,
                    "end_frame": ed,
                    "label": match_candidate_to_annotation(cand, ann_df),
                    "source_mask_prev": os.path.basename(tif_files[i]),
                    "source_mask_curr": os.path.basename(tif_files[i + 1]),
                    **feat,
                })
        added = len(rows) - before_rows
        pmsg(cfg, f"[build][{video_idx}/{total_videos}] Done {video_name}: generated {added} candidate rows")

    if not rows:
        raise RuntimeError("No training candidates generated. Please check mask folders and thresholds.")

    out = pd.DataFrame(rows)
    pmsg(cfg, f"[build] Finished. Total candidate rows: {len(out)}")
    return out


def train_model(cfg: dict):
    pmsg(cfg, "[train] Step 1/4: building training table...")
    df = build_training_table(cfg)
    pmsg(cfg, f"[train] Step 2/4: preparing features from {len(df)} rows...")
    X = df[FEATURE_COLUMNS].copy()
    y = df["label"].astype(int).copy()

    stratify = y if y.nunique() > 1 else None
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=cfg["seed"], stratify=stratify
    )
    pmsg(cfg, f"[train] Train split: {len(X_train)}, val split: {len(X_val)}")

    clf = RandomForestClassifier(
        n_estimators=400,
        max_depth=10,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=cfg["seed"],
        n_jobs=-1,
    )
    pmsg(cfg, "[train] Step 3/4: fitting RandomForestClassifier...")
    clf.fit(X_train, y_train)

    if len(np.unique(y_val)) > 1:
        pmsg(cfg, "[train] Validation report:")
        pred = clf.predict(X_val)
        print(classification_report(y_val, pred, digits=4), flush=True)

    save_path = cfg["model"]["save_path"]
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    pmsg(cfg, f"[train] Step 4/4: saving model to {save_path}")
    joblib.dump({
        "model": clf,
        "feature_columns": FEATURE_COLUMNS,
        "config": cfg,
    }, save_path)

    pmsg(cfg, "[train] Training pipeline finished.")
    return clf, df


def suppress_duplicates(df: pd.DataFrame, frame_gap: int = 2) -> pd.DataFrame:
    if len(df) == 0:
        return df

    df = df.sort_values(["Score", "start_frame"], ascending=[False, True]).reset_index(drop=True)
    keep = []
    used = np.zeros(len(df), dtype=bool)

    for i in range(len(df)):
        if used[i]:
            continue
        keep.append(i)
        for j in range(i + 1, len(df)):
            if used[j]:
                continue
            same_type = df.loc[i, "Event Type"] == df.loc[j, "Event Type"]
            close_frame = abs(int(df.loc[i, "start_frame"]) - int(df.loc[j, "start_frame"])) <= frame_gap
            same_parent = df.loc[i, "Parent IDs"] == df.loc[j, "Parent IDs"]
            same_child = df.loc[i, "Child IDs"] == df.loc[j, "Child IDs"]
            if same_type and close_frame and (same_parent or same_child):
                used[j] = True

    return df.loc[keep].sort_values(["start_frame", "Score"], ascending=[True, False]).reset_index(drop=True)


def predict_video(cfg: dict, video_name: str, model_path: str, output_csv: str) -> pd.DataFrame:
    pkg = joblib.load(model_path)
    clf = pkg["model"]
    feat_cols = pkg["feature_columns"]
    train_cfg = pkg["config"]

    mask_dir = os.path.join(cfg["data"]["mask_root"], video_name)
    pmsg(cfg, f"[predict] Loading masks from {mask_dir}")
    masks, tif_files, frame_nums = load_mask_sequence(mask_dir)
    pmsg(cfg, f"[predict] {video_name}: {len(masks)} masks, {max(0, len(masks)-1)} transitions")

    rows = []
    every_n = progress_every_n_frames(cfg)
    for i in range(len(masks) - 1):
        if i == 0 or (i + 1) % every_n == 0 or i == len(masks) - 2:
            pmsg(cfg, f"[predict] {video_name}: transition {i + 1}/{len(masks) - 1} ({frame_nums[i]} -> {frame_nums[i + 1]})")
        cands = generate_candidates_for_transition(
            masks[i], masks[i + 1], frame_nums[i], frame_nums[i + 1], train_cfg["feature"]
        )
        for cand in cands:
            feat = candidate_to_feature_row(cand)
            x = pd.DataFrame([feat])[feat_cols]
            score = float(clf.predict_proba(x)[0, 1])

            parent_ids, child_ids = candidate_to_sheet_columns(cand)
            st, ed, frame_text = candidate_frame_range(cand)

            rows.append({
                "Video": video_name,
                "Frame": frame_text,
                "Event Type": cand["event_type"],
                "Parent IDs": "".join([f"[{x}]" for x in parent_ids]),
                "Child IDs": "".join([f"[{x}]" for x in child_ids]),
                "Note": candidate_note(cand),
                "Score": score,
                "start_frame": st,
                "end_frame": ed,
                "source_mask_prev": os.path.basename(tif_files[i]),
                "source_mask_curr": os.path.basename(tif_files[i + 1]),
            })

    pred_df = pd.DataFrame(rows)
    if len(pred_df) == 0:
        pred_df = pd.DataFrame(columns=[
            "Video", "Frame", "Event Type", "Parent IDs", "Child IDs", "Note",
            "Score", "start_frame", "end_frame", "source_mask_prev", "source_mask_curr"
        ])
    else:
        pmsg(cfg, f"[predict] Raw candidates: {len(pred_df)}")
        pred_df = pred_df[pred_df["Score"] >= train_cfg["feature"]["decision_threshold"]].copy()
        pmsg(cfg, f"[predict] After threshold: {len(pred_df)}")
        pred_df = suppress_duplicates(pred_df, frame_gap=train_cfg["feature"]["dedup_frame_gap"])
        pmsg(cfg, f"[predict] After dedup: {len(pred_df)}")

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    pred_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    pmsg(cfg, f"[predict] Saved CSV to {output_csv}")
    return pred_df
