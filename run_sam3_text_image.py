import argparse
import os
import re
from pathlib import Path

import numpy as np
import cv2
from PIL import Image
import torch

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def safe_name(s: str, max_len: int = 64) -> str:
    s = s.strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^0-9a-zA-Z_\-\.]+", "", s)
    return s[:max_len] if len(s) > max_len else s


def to_nhw_bool(masks_t: torch.Tensor) -> torch.Tensor:
    # Sam3Processor.state['masks'] 可能是 (N,1,H,W)
    if masks_t.ndim == 4 and masks_t.shape[1] == 1:
        masks_t = masks_t[:, 0]
    if masks_t.ndim == 2:
        masks_t = masks_t[None, ...]
    return masks_t.bool()


def iter_images(input_dir: str, exts=(".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")):
    p = Path(input_dir)
    files = []
    for ext in exts:
        files.extend(p.rglob(f"*{ext}"))
        files.extend(p.rglob(f"*{ext.upper()}"))
    return sorted(set(files))


def mask_bbox(m: np.ndarray):
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    return (x1, y1, x2, y2)  # inclusive


def bbox_area(b):
    x1, y1, x2, y2 = b
    return max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)


def bbox_iou(b1, b2) -> float:
    if b1 is None or b2 is None:
        return 0.0
    ax1, ay1, ax2, ay2 = b1
    bx1, by1, bx2, by2 = b2
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1 + 1), max(0, iy2 - iy1 + 1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = bbox_area(b1) + bbox_area(b2) - inter
    return float(inter) / float(ua + 1e-9)


def mask_iou_cropped(a: np.ndarray, b: np.ndarray, ba=None, bb=None) -> float:
    # a,b: (H,W) bool
    # 用 bbox 的 union 做裁剪，显著减少大图上的逻辑运算量
    if ba is None:
        ba = mask_bbox(a)
    if bb is None:
        bb = mask_bbox(b)
    if ba is None or bb is None:
        return 0.0
    x1 = min(ba[0], bb[0])
    y1 = min(ba[1], bb[1])
    x2 = max(ba[2], bb[2])
    y2 = max(ba[3], bb[3])
    aa = a[y1 : y2 + 1, x1 : x2 + 1]
    bbm = b[y1 : y2 + 1, x1 : x2 + 1]
    inter = np.logical_and(aa, bbm).sum()
    union = np.logical_or(aa, bbm).sum()
    return float(inter) / float(union + 1e-9)


def centroid_and_area(m: np.ndarray):
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        return (np.nan, np.nan, 0)
    cx = float(xs.mean())
    cy = float(ys.mean())
    area = int(len(xs))
    return (cx, cy, area)


def overlay_tracks(image_rgb_u8: np.ndarray, track_label: np.ndarray, alpha=0.45) -> np.ndarray:
    """
    track_label: (H,W) int32, 0背景，其余是track id
    """
    out = image_rgb_u8.copy()
    ids = np.unique(track_label)
    ids = ids[ids != 0]
    rng = np.random.default_rng(0)
    for tid in ids:
        m = (track_label == tid)
        color = rng.integers(0, 255, size=(3,), dtype=np.uint8)
        out[m] = (out[m] * (1 - alpha) + color * alpha).astype(np.uint8)
    return out


def run_sam3_on_image(image_path: str, processor: Sam3Processor, text: str):
    img = Image.open(image_path).convert("RGB")
    state = {}
    state = processor.set_image(img, state=state)
    state = processor.set_text_prompt(text, state=state)
    masks_t = to_nhw_bool(state["masks"])
    scores_t = state["scores"]
    return img, masks_t, scores_t


def track_sequence(
    files, processor, text, outdir, conf, save_vis,
    iou_th=0.10,
    max_center_dist=80.0,
    max_lost=5,
    lambda_dist=0.25,
    area_ratio_min=0.4,
    area_ratio_max=2.5,
    bbox_iou_th=0.05,
    use_bbox_fallback=True,
    use_hungarian=True,
):
    """
    Hungarian全局匹配 + 轨迹存活(max_lost) + 速度预测 + bbox轻接触容错。
    """
    ensure_dir(outdir)
    text_tag = safe_name(text)

    next_track_id = 1
    # tracks: tid -> dict(mask, bbox, cx, cy, area, vx, vy, lost)
    tracks = {}
    csv_rows = []

    hungarian_ok = False
    linear_sum_assignment = None
    if use_hungarian:
        try:
            from scipy.optimize import linear_sum_assignment as _lsa
            linear_sum_assignment = _lsa
            hungarian_ok = True
        except Exception:
            hungarian_ok = False

    BIG = 1e9

    for t, p in enumerate(files):
        img, masks_t, scores_t = run_sam3_on_image(str(p), processor, text)
        masks_np = masks_t.cpu().numpy().astype(bool)  # (N,H,W)
        scores = scores_t.detach().cpu().numpy()

        H, W = masks_np.shape[-2], masks_np.shape[-1]
        track_label = np.zeros((H, W), dtype=np.int32)

        # --- 当前帧 detections ---
        dets = []
        for i in range(masks_np.shape[0]):
            m = masks_np[i]
            cx, cy, area = centroid_and_area(m)
            if area <= 0 or np.isnan(cx) or np.isnan(cy):
                continue
            bb = mask_bbox(m)
            dets.append({
                "i": i,
                "mask": m,
                "bbox": bb,
                "cx": cx,
                "cy": cy,
                "area": area,
                "score": float(scores[i]) if i < len(scores) else float("nan"),
            })

        track_ids = list(tracks.keys())
        det_ids = [d["i"] for d in dets]

        # --- 预测位置（用于距离门控） ---
        pred = {}
        for tid in track_ids:
            tr = tracks[tid]
            px = tr["cx"] + tr.get("vx", 0.0)
            py = tr["cy"] + tr.get("vy", 0.0)
            pred[tid] = (px, py)

        # --- 代价矩阵 / 候选 ---
        det_idx_map = {det_ids[k]: k for k in range(len(det_ids))}
        cost = None
        if hungarian_ok and len(track_ids) > 0 and len(det_ids) > 0:
            cost = np.full((len(track_ids), len(det_ids)), BIG, dtype=np.float32)

        candidates = []  # greedy用

        # 为了快速索引 det
        det_by_id = {d["i"]: d for d in dets}

        for det in dets:
            for r, tid in enumerate(track_ids):
                tr = tracks[tid]

                # 1) 距离门控：用预测位置
                px, py = pred[tid]
                dist = float(np.hypot(det["cx"] - px, det["cy"] - py))
                if dist > max_center_dist:
                    continue

                # 2) 面积比门控
                area_ratio = det["area"] / (tr["area"] + 1e-9)
                if not (area_ratio_min <= area_ratio <= area_ratio_max):
                    continue

                # 3) bbox IoU（轻微接触/边界抖动时更稳定；也用于减少要算mask IoU的pair数量）
                bbiou = bbox_iou(det["bbox"], tr.get("bbox"))

                # 关键修改：只在没丢失时用 bbox_iou_th；丢失后放宽/关闭
                if tr.get("lost", 0) == 0:
                    if bbiou < bbox_iou_th:
                        continue
                else:
                    # 丢失状态下不做 bbox IoU 门控（或用更小阈值）
                    pass

                # 4) mask IoU（裁剪版）
                miou = mask_iou_cropped(det["mask"], tr["mask"], det["bbox"], tr.get("bbox"))
                if miou < iou_th and (not use_bbox_fallback):
                    continue

                # 轻接触容错：mask IoU 低但 bbox IoU 还可以时，给一个“有效IoU”
                eff_iou = miou
                if miou < iou_th and use_bbox_fallback:
                    # bbox更粗糙，权重给低一点，避免误配
                    eff_iou = max(miou, 0.5 * bbiou)

                # 分数/代价
                norm_dist = dist / (max_center_dist + 1e-9)
                greedy_score = eff_iou - lambda_dist * norm_dist
                candidates.append((greedy_score, tid, det["i"]))

                if cost is not None:
                    c = det_idx_map[det["i"]]
                    cost[r, c] = (1.0 - eff_iou) + lambda_dist * norm_dist

        det_to_tid = {}
        matched_tids = set()
        matched_dets = set()

        # --- 匹配：Hungarian优先，否则贪心 ---
        if cost is not None and np.isfinite(cost).any():
            rr, cc = linear_sum_assignment(cost)
            for r, c in zip(rr, cc):
                if cost[r, c] >= BIG:
                    continue
                tid = track_ids[r]
                det_i = det_ids[c]
                det_to_tid[det_i] = tid
                matched_tids.add(tid)
                matched_dets.add(det_i)
        else:
            candidates.sort(key=lambda x: x[0], reverse=True)
            for s, tid, det_i in candidates:
                if tid in matched_tids or det_i in matched_dets:
                    continue
                det_to_tid[det_i] = tid
                matched_tids.add(tid)
                matched_dets.add(det_i)

        # --- 未匹配 detection：新建轨迹 ---
        for det in dets:
            if det["i"] in det_to_tid:
                continue
            tid = next_track_id
            next_track_id += 1
            det_to_tid[det["i"]] = tid
            tracks[tid] = {
                "mask": det["mask"],
                "bbox": det["bbox"],
                "cx": det["cx"],
                "cy": det["cy"],
                "area": det["area"],
                "vx": 0.0,
                "vy": 0.0,
                "lost": 0,
            }

        # --- 更新匹配到的轨迹 ---
        updated = set()
        for det in dets:
            tid = det_to_tid[det["i"]]
            tr = tracks.get(tid)
            if tr is None:
                tracks[tid] = {
                    "mask": det["mask"],
                    "bbox": det["bbox"],
                    "cx": det["cx"],
                    "cy": det["cy"],
                    "area": det["area"],
                    "vx": 0.0,
                    "vy": 0.0,
                    "lost": 0,
                }
                tr = tracks[tid]

            vx = det["cx"] - tr["cx"]
            vy = det["cy"] - tr["cy"]

            tr["mask"] = det["mask"]
            tr["bbox"] = det["bbox"]
            tr["cx"] = det["cx"]
            tr["cy"] = det["cy"]
            tr["area"] = det["area"]
            tr["vx"] = float(vx)
            tr["vy"] = float(vy)
            tr["lost"] = 0

            updated.add(tid)
            track_label[det["mask"]] = tid

            csv_rows.append({
                "frame": t,
                "path": str(p),
                "track_id": tid,
                "det_idx": det["i"],
                "cx": det["cx"],
                "cy": det["cy"],
                "area": det["area"],
                "sam_score": det["score"],
                "lost": tr["lost"],
            })

        # --- 未匹配旧轨迹：lost++，用速度推进；超 max_lost 才删 ---
        to_delete = []
        for tid in list(tracks.keys()):
            if tid in updated:
                continue
            tr = tracks[tid]
            tr["lost"] = tr.get("lost", 0) + 1
            tr["cx"] = tr["cx"] + tr.get("vx", 0.0)
            tr["cy"] = tr["cy"] + tr.get("vy", 0.0)
            if tr["lost"] > max_lost:
                to_delete.append(tid)
        for tid in to_delete:
            del tracks[tid]

        base = os.path.splitext(os.path.basename(str(p)))[0]
        prefix = f"{base}__{text_tag}__conf{conf:.2f}"

        # 保存 track label（每帧一张，值=track id）
        try:
            import tifffile as tiff
            tiff.imwrite(os.path.join(outdir, f"{prefix}__tracks.tif"), track_label.astype(np.int32))
        except ImportError:
            cv2.imwrite(os.path.join(outdir, f"{prefix}__tracks.png"), track_label.astype(np.uint16))

        if save_vis:
            vis_rgb = overlay_tracks(np.array(img), track_label)
            cv2.imwrite(os.path.join(outdir, f"{prefix}__tracks_vis.png"), vis_rgb[:, :, ::-1].copy())

        print(f"[OK] frame={t:05d} dets={len(dets)} tracks_alive={len(tracks)} file={p}")

    # 写 tracks.csv
    csv_path = os.path.join(outdir, f"tracks__{text_tag}__conf{conf:.2f}.csv")
    import csv
    fieldnames = list(csv_rows[0].keys()) if csv_rows else [
        "frame", "path", "track_id", "det_idx", "cx", "cy", "area", "sam_score", "lost"
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in csv_rows:
            w.writerow(r)
    print("Saved CSV:", csv_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=None, help="单张图片路径（与 --input_dir 二选一）")
    ap.add_argument("--input_dir", default=None, help="输入文件夹（递归搜图，与 --image 二选一）")
    ap.add_argument("--text", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--outdir", default="out_sam3")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--conf", type=float, default=0.5)  # 阈值过滤
    ap.add_argument("--resolution", type=int, default=1008)
    ap.add_argument("--save_vis", action="store_true")

    # tracking
    ap.add_argument("--track", action="store_true", help="开启跨帧track id输出（需--input_dir）")
    ap.add_argument("--iou_th", type=float, default=0.10, help="mask IoU阈值")
    ap.add_argument("--max_center_dist", type=float, default=80.0, help="质心距离阈值(像素)")
    ap.add_argument("--max_lost", type=int, default=5, help="允许连续漏检的最大帧数")
    ap.add_argument("--lambda_dist", type=float, default=0.25, help="距离项权重(归一化后)")
    ap.add_argument("--area_ratio_min", type=float, default=0.4, help="面积比下限")
    ap.add_argument("--area_ratio_max", type=float, default=2.5, help="面积比上限")
    ap.add_argument("--bbox_iou_th", type=float, default=0.05, help="bbox IoU门控阈值(轻接触更稳)")
    ap.add_argument("--no_bbox_fallback", action="store_true", help="禁用bbox容错(更保守)")
    ap.add_argument("--no_hungarian", action="store_true", help="禁用Hungarian，退化为贪心(不建议)")

    args = ap.parse_args()
    ensure_dir(args.outdir)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    # 模型只初始化一次
    model = build_sam3_image_model(
        device=device,
        eval_mode=True,
        checkpoint_path=args.ckpt,
        load_from_HF=False,
        enable_segmentation=True,
        enable_inst_interactivity=False,
        compile=False,
    )
    processor = Sam3Processor(
        model=model,
        resolution=args.resolution,
        device=device,
        confidence_threshold=args.conf,
    )

    if args.track:
        if not args.input_dir:
            raise ValueError("--track 模式需要 --input_dir（按文件名排序视为时间顺序）")
        files = iter_images(args.input_dir)
        print(f"Found {len(files)} images under: {args.input_dir}")

        track_sequence(
            files, processor, args.text, args.outdir, args.conf, args.save_vis,
            iou_th=args.iou_th,
            max_center_dist=args.max_center_dist,
            max_lost=args.max_lost,
            lambda_dist=args.lambda_dist,
            area_ratio_min=args.area_ratio_min,
            area_ratio_max=args.area_ratio_max,
            bbox_iou_th=args.bbox_iou_th,
            use_bbox_fallback=(not args.no_bbox_fallback),
            use_hungarian=(not args.no_hungarian),
        )
        return

    # 非track：逐帧独立输出（每帧独立ID）
    if (args.image is None) == (args.input_dir is None):
        raise ValueError("请二选一：要么传 --image，要么传 --input_dir")

    if args.image:
        files = [Path(args.image)]
    else:
        files = iter_images(args.input_dir)

    for p in files:
        img, masks_t, scores_t = run_sam3_on_image(str(p), processor, args.text)
        masks_np = masks_t.cpu().numpy().astype(bool)

        base = os.path.splitext(os.path.basename(str(p)))[0]
        prefix = f"{base}__{safe_name(args.text)}__conf{args.conf:.2f}"

        # 保存 union
        union = masks_np.any(axis=0) if masks_np.size > 0 else np.zeros(np.array(img).shape[:2], dtype=bool)
        cv2.imwrite(os.path.join(args.outdir, f"{prefix}__mask_all.png"), (union.astype(np.uint8) * 255))

        # 保存 labels（每帧独立ID）
        label = np.zeros(union.shape, dtype=np.int32)
        for i in range(masks_np.shape[0]):
            label[masks_np[i]] = i + 1

        try:
            import tifffile as tiff
            tiff.imwrite(os.path.join(args.outdir, f"{prefix}__labels.tif"), label)
        except ImportError:
            cv2.imwrite(os.path.join(args.outdir, f"{prefix}__labels.png"), label.astype(np.uint16))

        if args.save_vis:
            vis = overlay_tracks(np.array(img), label)
            cv2.imwrite(os.path.join(args.outdir, f"{prefix}__vis.png"), vis[:, :, ::-1].copy())

        s = scores_t.detach().cpu().numpy()
        msg = f"[OK] {p} instances={masks_np.shape[0]}"
        if masks_np.shape[0] > 0:
            msg += f" score(min/mean/max)={s.min():.3f}/{s.mean():.3f}/{s.max():.3f}"
        print(msg)


if __name__ == "__main__":
    main()
