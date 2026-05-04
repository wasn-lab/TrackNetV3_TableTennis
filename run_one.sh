#!/bin/bash
set -e

START_TIME=$(date +%s)

echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"

VIDEO_FILE="/home/code-server/NO7/078/C0037.MP4"
SAVE_DIR="/home/code-server/NO7_pred_result/078"
NAME=$(basename "$VIDEO_FILE")
NAME="${NAME%.*}"

CUDA_VISIBLE_DEVICES=2 python predict.py --video_file "$VIDEO_FILE" --tracknet_file exp/TrackNet_best.pt --inpaintnet_file exp/InpaintNet_best.pt --save_dir "$SAVE_DIR" --large_video
python speed_analysis/stroke_zone_analysis.py --video_file "$VIDEO_FILE" --ball_csv "$SAVE_DIR/${NAME}_ball.csv" --save_dir "$SAVE_DIR"
python speed_analysis/plot_speed.py --input "$SAVE_DIR"

END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))

echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Total time: ${TOTAL_TIME} seconds"
echo "Done."