#!/bin/bash
set -e

VIDEO_DIR="/home/code-server/NO7"
SAVE_DIR="/home/code-server/NO7_pred_result"

CUDA_VISIBLE_DEVICES=1 python predict.py --video_dir "$VIDEO_DIR" --tracknet_file exp/TrackNet_best.pt --inpaintnet_file exp/InpaintNet_best.pt --save_dir "$SAVE_DIR" --output_video --large_video

python speed_analysis/stroke_zone_analysis.py --video_root "$VIDEO_DIR" --save_root "$SAVE_DIR" --save_video

python speed_analysis/plot_speed.py --input "$SAVE_DIR"

echo "Done."