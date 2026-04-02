#!/usr/bin/env bash
set -u  

export NVIDIA_DRIVER_CAPABILITIES="${NVIDIA_DRIVER_CAPABILITIES:-compute,utility}"

if [ -d /lib/x86_64-linux-gnu ]; then
  export LD_LIBRARY_PATH="/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

# conda 初始化
if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
fi

conda activate sam3   
# conda install -c conda-forge einops
# python -m pip install -U decord
# pip install psutil 
# pip install tifffile   # 只为 labels.tif；不装也能跑，只是少一个输出
# python run_sam3_text_image.py \
#   --input_dir /data/250010003/H2O2101/500C-TEMmovie2-jpg \
#   --text "irregular shape" \
#   --ckpt /data/250010003/sam3-checkpoint/sam3.pt \
#   --outdir /data/250010003/out/H2O2101/500C-TEMmovie2-jpg \
#   --conf 0.4 \
#   --save_vis

# python run_sam3_text_image.py \
#   --image /data/250010003/D1/0009.jpg\
#   --text "irregular shapes" \
#   --ckpt /data/250010003/sam3-checkpoint/sam3.pt \
#   --outdir /data/250010003/out/D1 \
#   --conf 0.05 \
#   --save_vis


python run_sam3_text_image.py \
  --input_dir /data/250010003/H2O2101/500C-left \
  --text "small dark solid TEM particle,irregular shape,sharp edge,non-hollow,no bubble" \
  --ckpt /data/250010003/sam3-checkpoint/sam3.pt \
  --outdir /data/250010003/out/H2O2101/500C-left\
  --conf 0.25 \
  --track --save_vis \
  --max_center_dist 150 \
  --iou_th 0.1\
  --bbox_iou_th 0.01 \
  --max_lost 10

