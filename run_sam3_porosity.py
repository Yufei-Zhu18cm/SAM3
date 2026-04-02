import argparse
import os
import re
import math
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


def run_sam3_on_image(image_path: str, processor: Sam3Processor, text: str):
    img = Image.open(image_path).convert("RGB")
    state = {}
    state = processor.set_image(img, state=state)
    state = processor.set_text_prompt(text, state=state)
    masks_t = to_nhw_bool(state["masks"])
    scores_t = state["scores"]
    return img, masks_t, scores_t


def detect_red_scale_bar_bbox_px(
    image_rgb_u8: np.ndarray,
    roi_frac=(0.60, 0.75, 1.00, 1.00),
    min_w_px=30,
    min_aspect=5.0,
):
    H, W = image_rgb_u8.shape[:2]
    x0 = int(W * roi_frac[0])
    y0 = int(H * roi_frac[1])
    x1 = int(W * roi_frac[2])
    y1 = int(H * roi_frac[3])
    roi = image_rgb_u8[y0:y1, x0:x1]

    roi_bgr = cv2.cvtColor(roi, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)

    lower1 = np.array([0, 120, 80], dtype=np.uint8)
    upper1 = np.array([10, 255, 255], dtype=np.uint8)
    lower2 = np.array([170, 120, 80], dtype=np.uint8)
    upper2 = np.array([180, 255, 255], dtype=np.uint8)

    mask = cv2.bitwise_or(
        cv2.inRange(hsv, lower1, upper1),
        cv2.inRange(hsv, lower2, upper2),
    )

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w < min_w_px or h <= 0:
            continue
        if (w / float(h)) < min_aspect:
            continue
        area = float(cv2.contourArea(c))
        score = w + 0.001 * area - 0.1 * h
        if best is None or score > best[0]:
            best = (score, (x, y, w, h))

    if best is None:
        return None
    _, (x, y, w, h) = best
    return (x + x0, y + y0, w, h)


def bbox_from_mask(mask_bool: np.ndarray):
    ys, xs = np.nonzero(mask_bool)
    if len(xs) == 0:
        return None
    x0 = int(xs.min())
    x1 = int(xs.max())
    y0 = int(ys.min())
    y1 = int(ys.max())
    return (x0, y0, x1, y1)


def overlay_instances(image_rgb_u8: np.ndarray, masks_bool: list, alpha=0.45):
    out = image_rgb_u8.copy()
    rng = np.random.default_rng(0)
    for idx, m in enumerate(masks_bool):
        color = rng.integers(0, 255, size=(3,), dtype=np.uint8)
        out[m] = (out[m] * (1 - alpha) + color * alpha).astype(np.uint8)
    return out


def write_csv(path, rows, fieldnames):
    import csv
    ensure_dir(os.path.dirname(path) if os.path.dirname(path) else ".")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=None, help="单张图片路径（与 --input_dir 二选一）")
    ap.add_argument("--input_dir", default=None, help="输入文件夹（递归搜图，与 --image 二选一）")
    ap.add_argument("--text", required=True, help="SAM3 文本提示词（比如 pore/hole/void）")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--outdir", default="out_porosity")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--conf", type=float, default=0.3, help="SAM3 confidence_threshold")
    ap.add_argument("--resolution", type=int, default=1008)

    # stats
    ap.add_argument("--min_area_px", type=int, default=1, help="最小孔洞面积(像素)，小于则忽略")
    ap.add_argument("--scale_um", type=float, default=100.0, help="红色比例尺对应真实长度(um)，默认100")
    ap.add_argument("--px_per_um", type=float, default=None, help="手动指定 像素/微米；给了就不自动识别比例尺")
    ap.add_argument("--auto_scale_bar", action="store_true", help="自动识别右下角红色比例尺")
    ap.add_argument("--exclude_scale_bar", action="store_true", help="把比例尺区域从统计面积里剔除")
    ap.add_argument("--scale_bar_margin_px", type=int, default=8, help="剔除比例尺bbox外扩margin(px)")

    # outputs
    ap.add_argument("--save_mask_all", action="store_true", help="额外保存 union mask（二值图）")
    ap.add_argument("--vis_alpha", type=float, default=0.45)

    args = ap.parse_args()
    ensure_dir(args.outdir)

    if (args.image is None) == (args.input_dir is None):
        raise ValueError("请二选一：要么传 --image，要么传 --input_dir")

    files = [Path(args.image)] if args.image else iter_images(args.input_dir)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

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

    text_tag = safe_name(args.text)

    pores_rows_all = []
    summary_rows = []

    for p in files:
        img_pil, masks_t, scores_t = run_sam3_on_image(str(p), processor, args.text)
        img_rgb = np.array(img_pil).astype(np.uint8)
        masks_np = masks_t.detach().cpu().numpy().astype(bool)  # (N,H,W)
        scores_np = scores_t.detach().cpu().numpy()
        H, W = img_rgb.shape[:2]

        # scale bar
        scale_bbox = None
        scale_bar_w_px = None
        px_per_um = args.px_per_um

        if (px_per_um is None) and args.auto_scale_bar:
            scale_bbox = detect_red_scale_bar_bbox_px(img_rgb)
            if scale_bbox is not None:
                x, y, w, h = scale_bbox
                scale_bar_w_px = int(w)
                if args.scale_um > 0:
                    px_per_um = float(scale_bar_w_px) / float(args.scale_um)

        um_per_px = (1.0 / px_per_um) if (px_per_um is not None and px_per_um > 0) else None

        # valid area
        valid = np.ones((H, W), dtype=bool)
        excluded_bbox_str = ""
        if args.exclude_scale_bar and (scale_bbox is not None):
            x, y, w, h = scale_bbox
            m = int(args.scale_bar_margin_px)
            x0 = max(0, x - m)
            y0 = max(0, y - m)
            x1 = min(W, x + w + m)
            y1 = min(H, y + h + m)
            valid[y0:y1, x0:x1] = False
            excluded_bbox_str = f"{x0},{y0},{x1-x0},{y1-y0}"

        valid_area_px = int(valid.sum())

        # filter + compute union area
        kept_masks = []
        kept_scores = []
        for i in range(masks_np.shape[0]):
            m = masks_np[i] & valid
            area_px = int(m.sum())
            if area_px < int(args.min_area_px):
                continue
            kept_masks.append(m)
            kept_scores.append(float(scores_np[i]) if i < len(scores_np) else float("nan"))

        pore_count = len(kept_masks)
        union = np.zeros((H, W), dtype=bool)
        for m in kept_masks:
            union |= m
        union_pore_area_px = int(union.sum())

        pore_area_ratio = (float(union_pore_area_px) / float(valid_area_px)) if valid_area_px > 0 else float("nan")
        porosity_percent = pore_area_ratio * 100.0
        solid_area_ratio = 1.0 - pore_area_ratio
        solid_percent = solid_area_ratio * 100.0

        # per-pore rows
        for j, m in enumerate(kept_masks):
            area_px = int(m.sum())
            bb = bbox_from_mask(m)
            if bb is None:
                continue
            x0, y0, x1, y1 = bb
            bw = int(x1 - x0 + 1)
            bh = int(y1 - y0 + 1)
            ys, xs = np.nonzero(m)
            cx = float(xs.mean()) if len(xs) else float("nan")
            cy = float(ys.mean()) if len(xs) else float("nan")
            eq_diam_px = float(2.0 * math.sqrt(float(area_px) / math.pi))

            row = {
                "image": str(p),
                "pore_id": j + 1,
                "area_px": int(area_px),
                "eq_diam_px": float(eq_diam_px),
                "bbox_w_px": int(bw),
                "bbox_h_px": int(bh),
                "cx_px": float(cx),
                "cy_px": float(cy),
                "sam_score": float(kept_scores[j]),
                "px_per_um": px_per_um if px_per_um is not None else "",
                "scale_um": float(args.scale_um),
                "scale_bar_w_px": scale_bar_w_px if scale_bar_w_px is not None else "",
            }
            if um_per_px is not None:
                row.update({
                    "area_um2": float(area_px) * (um_per_px ** 2),
                    "eq_diam_um": float(eq_diam_px) * um_per_px,
                    "bbox_w_um": float(bw) * um_per_px,
                    "bbox_h_um": float(bh) * um_per_px,
                    "cx_um": float(cx) * um_per_px,
                    "cy_um": float(cy) * um_per_px,
                })
            else:
                row.update({
                    "area_um2": "",
                    "eq_diam_um": "",
                    "bbox_w_um": "",
                    "bbox_h_um": "",
                    "cx_um": "",
                    "cy_um": "",
                })
            pores_rows_all.append(row)

        # summary row 
        sum_row = {
            "image": str(p),
            "H": int(H),
            "W": int(W),
            "valid_area_px": int(valid_area_px),
            "pore_count": int(pore_count),
            "union_pore_area_px": int(union_pore_area_px),
            "pore_area_ratio": float(pore_area_ratio),       # 0~1
            "porosity_percent": float(porosity_percent),     # 0~100
            "solid_area_ratio": float(solid_area_ratio),     # 0~1
            "solid_percent": float(solid_percent),           # 0~100
            "min_area_px": int(args.min_area_px),
            "px_per_um": px_per_um if px_per_um is not None else "",
            "scale_um": float(args.scale_um),
            "scale_bar_w_px": scale_bar_w_px if scale_bar_w_px is not None else "",
            "excluded_bbox_xywh": excluded_bbox_str,
        }
        if um_per_px is not None:
            sum_row.update({
                "valid_area_um2": float(valid_area_px) * (um_per_px ** 2),
                "union_pore_area_um2": float(union_pore_area_px) * (um_per_px ** 2),
            })
        else:
            sum_row.update({"valid_area_um2": "", "union_pore_area_um2": ""})
        summary_rows.append(sum_row)

        # outputs: vis + optional mask
        vis = overlay_instances(img_rgb, kept_masks, alpha=float(args.vis_alpha))
        # put text
        txt = f"pores={pore_count} porosity={porosity_percent:.4f}%"
        cv2.putText(vis, txt, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2, cv2.LINE_AA)

        out_vis = os.path.join(args.outdir, f"{Path(p).stem}__{text_tag}__vis.png")
        cv2.imwrite(out_vis, vis[:, :, ::-1].copy())

        if args.save_mask_all:
            out_mask = os.path.join(args.outdir, f"{Path(p).stem}__{text_tag}__mask_all.png")
            cv2.imwrite(out_mask, (union.astype(np.uint8) * 255))

        print(f"[OK] {p} pores={pore_count} porosity={porosity_percent:.6f}%")

    pores_csv = os.path.join(args.outdir, f"porosity_pores__{text_tag}__conf{args.conf:.2f}.csv")
    summary_csv = os.path.join(args.outdir, f"porosity_summary__{text_tag}__conf{args.conf:.2f}.csv")

    pore_fields = [
        "image","pore_id","area_px","area_um2","eq_diam_px","eq_diam_um",
        "bbox_w_px","bbox_h_px","bbox_w_um","bbox_h_um",
        "cx_px","cy_px","cx_um","cy_um",
        "sam_score","px_per_um","scale_um","scale_bar_w_px",
    ]
    sum_fields = [
        "image","H","W","valid_area_px","valid_area_um2",
        "pore_count","union_pore_area_px","union_pore_area_um2",
        "pore_area_ratio","porosity_percent","solid_area_ratio","solid_percent",
        "min_area_px","px_per_um","scale_um","scale_bar_w_px","excluded_bbox_xywh",
    ]

    write_csv(pores_csv, pores_rows_all, pore_fields)
    write_csv(summary_csv, summary_rows, sum_fields)

    print("Saved:", pores_csv)
    print("Saved:", summary_csv)


if __name__ == "__main__":
    main()
