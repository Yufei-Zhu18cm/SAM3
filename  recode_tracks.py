import argparse
import os
import re
from pathlib import Path
from collections import defaultdict

import numpy as np

try:
    import tifffile as tiff
except ImportError:
    raise ImportError("Please install tifffile: pip install tifffile")


FRAME_RE = re.compile(r"(?P<frame>\d+)")


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def find_frame_id(name: str) -> str:
    """
    从文件名里提取 frame id（默认取第一个连续数字串）。
    例如 0000_xxx_tracks.tif -> '0000'
    """
    m = FRAME_RE.search(name)
    if not m:
        raise ValueError(f"Cannot parse frame id from filename: {name}")
    return m.group("frame")


def list_tifs(d: str):
    p = Path(d)
    files = sorted([x for x in p.glob("*.tif")] + [x for x in p.glob("*.tiff")])
    return files


def parse_split_child(child_id: int):
    """
    split 编码：child = parent*10000 + 1110 + k
    返回 (parent_id, k) 或 (None, None) 表示不是split子代编码
    """
    if child_id <= 0:
        return (None, None)
    parent = child_id // 10000
    tail = child_id % 10000
    # tail 应该类似 1111, 1112, 1113 ... (即 1110+k)
    if tail < 1111:
        return (None, None)
    if (tail // 10) != 111:
        return (None, None)
    k = tail % 10
    if k <= 0:
        return (None, None)
    return (parent, k)


def unique_ids(lbl: np.ndarray):
    ids = np.unique(lbl.astype(np.int64))
    ids = ids[ids != 0]
    return ids


def overlap_counts(raw: np.ndarray, edited_mask: np.ndarray):
    """
    统计 edited_mask 区域内 raw 的 id 像素数量，用于推断 merge 父 id。
    返回 dict raw_id -> count
    """
    r = raw[edited_mask]
    if r.size == 0:
        return {}
    ids, cnt = np.unique(r.astype(np.int64), return_counts=True)
    out = {}
    for i, c in zip(ids, cnt):
        if int(i) != 0:
            out[int(i)] = int(c)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True, help="tracks_raw 目录，每帧一个 tif")
    ap.add_argument("--edited_dir", required=True, help="tracks_edited 目录（napari保存），每帧一个 tif")
    ap.add_argument("--out_dir", required=True, help="输出目录：tracks_final + csv")
    ap.add_argument("--min_parent_pixels", type=int, default=200, help="推断merge父id时的最小像素重叠阈值")
    ap.add_argument("--min_parent_frac", type=float, default=0.05, help="父id像素占该edited对象面积的最小比例")
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    out_tracks_dir = os.path.join(args.out_dir, "tracks_final")
    ensure_dir(out_tracks_dir)

    raw_files = list_tifs(args.raw_dir)
    edt_files = list_tifs(args.edited_dir)

    raw_map = {find_frame_id(f.name): f for f in raw_files}
    edt_map = {find_frame_id(f.name): f for f in edt_files}

    frames = sorted(set(raw_map.keys()) & set(edt_map.keys()), key=lambda x: int(x))
    if not frames:
        raise ValueError("No matching frames between raw_dir and edited_dir")

    # 事件与meta收集
    events = []  # list of dicts
    # track_id -> meta
    track_meta = defaultdict(lambda: {
        "first_frame": None,
        "last_frame": None,
        "kind": "unknown",   # raw / split_child / merge_child / manual_new
        "parents": set(),
        "children": set(),
    })

    def update_presence(track_ids, frame_idx):
        for tid in track_ids:
            m = track_meta[int(tid)]
            if m["first_frame"] is None:
                m["first_frame"] = frame_idx
            m["last_frame"] = frame_idx

    for fidx, fid in enumerate(frames):
        raw = tiff.imread(str(raw_map[fid])).astype(np.int64)
        edt = tiff.imread(str(edt_map[fid])).astype(np.int64)

        if raw.shape != edt.shape:
            raise ValueError(f"Shape mismatch frame {fid}: raw{raw.shape} edited{edt.shape}")

        # 输出：最终 track label 就等于你 napari 编辑后的 edt（保持你所有像素修正）
        out_path = os.path.join(out_tracks_dir, f"{fid}__tracks_final.tif")
        tiff.imwrite(out_path, edt.astype(np.int32))

        edt_ids = unique_ids(edt)
        raw_ids = unique_ids(raw)

        update_presence(edt_ids, int(fid))

        # 1) 标注 raw 里的 id（若仍存在）为 raw
        for tid in edt_ids:
            if int(tid) in set(map(int, raw_ids)):
                if track_meta[int(tid)]["kind"] == "unknown":
                    track_meta[int(tid)]["kind"] = "raw"

        # 2) 识别 split 子代（按编码解析），并建立 parent->child 关系
        split_children_by_parent = defaultdict(list)
        for tid in edt_ids:
            parent, k = parse_split_child(int(tid))
            if parent is not None:
                split_children_by_parent[parent].append(int(tid))
                track_meta[int(tid)]["kind"] = "split_child"
                track_meta[int(tid)]["parents"].add(int(parent))
                track_meta[int(parent)]["children"].add(int(tid))

        # 如果某个 parent 在本帧出现 split 子代，记录 split 事件
        for parent, childs in split_children_by_parent.items():
            # 仅当 parent 在 raw 中确实存在过/或上一帧存在过时才记录更合理，但这里先宽松
            events.append({
                "frame": int(fid),
                "event_type": "split",
                "parents": str(parent),
                "children": ";".join(map(str, sorted(childs))),
            })

        # 3) 识别 merge：出现“新 id”，且不是 split 子代编码
        #    通过该新 id 区域在 raw 上的重叠，找出父 id 集合
        raw_id_set = set(map(int, raw_ids))
        for tid in edt_ids:
            tid = int(tid)
            parent, k = parse_split_child(tid)
            if parent is not None:
                continue  # split子代不当成merge
            if tid in raw_id_set:
                continue  # 仍是原id

            # 这是“新出现的id”（可能merge，也可能手动新增粒子）
            obj_mask = (edt == tid)
            area = int(obj_mask.sum())
            ov = overlap_counts(raw, obj_mask)  # raw_id -> px_count

            # 根据像素占比筛父id
            parents = []
            for rid, cnt in sorted(ov.items(), key=lambda x: x[1], reverse=True):
                if cnt < args.min_parent_pixels:
                    continue
                if (cnt / max(area, 1)) < args.min_parent_frac:
                    continue
                parents.append(int(rid))

            if len(parents) >= 2:
                # 认为是 merge 事件
                events.append({
                    "frame": int(fid),
                    "event_type": "merge",
                    "parents": ";".join(map(str, sorted(parents))),
                    "children": str(tid),
                })
                track_meta[tid]["kind"] = "merge_child"
                for p in parents:
                    track_meta[tid]["parents"].add(int(p))
                    track_meta[int(p)]["children"].add(int(tid))
            elif len(parents) == 1:
                # 可能是：某个父id在napari里被你“重命名/重分配”成新的id
                # 先记为 manual_new，并保留单亲关系（后续你可用事件表再细分）
                p = parents[0]
                track_meta[tid]["kind"] = "manual_new"
                track_meta[tid]["parents"].add(int(p))
                track_meta[int(p)]["children"].add(int(tid))
            else:
                # 完全没法从raw解释：认为是你手工新增的粒子
                track_meta[tid]["kind"] = "manual_new"

    # 写 events_final.csv
    events_path = os.path.join(args.out_dir, "events_final.csv")
    import csv
    with open(events_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["frame", "event_type", "parents", "children"])
        w.writeheader()
        for e in events:
            w.writerow(e)

    # 写 track_meta.csv
    meta_path = os.path.join(args.out_dir, "track_meta.csv")
    with open(meta_path, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["track_id", "kind", "first_frame", "last_frame", "parents", "children"]
        )
        w.writeheader()
        for tid, m in sorted(track_meta.items(), key=lambda x: int(x[0])):
            w.writerow({
                "track_id": int(tid),
                "kind": m["kind"],
                "first_frame": m["first_frame"],
                "last_frame": m["last_frame"],
                "parents": ";".join(map(str, sorted(m["parents"]))) if m["parents"] else "",
                "children": ";".join(map(str, sorted(m["children"]))) if m["children"] else "",
            })

    print("Done.")
    print(" - tracks_final dir:", out_tracks_dir)
    print(" - events:", events_path)
    print(" - meta:", meta_path)


if __name__ == "__main__":
    main()
