#!/usr/bin/env bash
set -u  # 先别用 -e，方便你看到错误后不中断

export NVIDIA_DRIVER_CAPABILITIES="${NVIDIA_DRIVER_CAPABILITIES:-compute,utility}"

if [ -d /lib/x86_64-linux-gnu ]; then
  export LD_LIBRARY_PATH="/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

# conda 初始化（保留你原来的逻辑）
if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
fi

conda activate sam3   # 取消注释，确保用对环境

# python run_sam3_porosity.py \
#   --image /data/250010003/D1/0009.jpg \
#   --text "irregular shapes" \
#   --ckpt /data/250010003/sam3-checkpoint/sam3.pt \
#   --outdir /data/250010003/out/D1 \
#   --conf 0.01 \
#   --min_area_px 1 \
#   --auto_scale_bar \
#   --exclude_scale_bar

python run_sam3_porosity.py \
  --input_dir /data/250010003/D1 \
  --text "irregular shapes" \
  --ckpt /data/250010003/sam3-checkpoint/sam3.pt \
  --outdir /data/250010003/out/D1/conf0.151 \
  --conf 0.15 \
  --min_area_px 1 \
  --auto_scale_bar \
  --exclude_scale_bar