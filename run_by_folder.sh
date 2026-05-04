#!/bin/bash
set -e

ROOT_DIR="/home/code-server/NO7"
SAVE_ROOT="/home/code-server/NO7_pred_result"

for VIDEO_DIR in "$ROOT_DIR"/*; do
  if [ ! -d "$VIDEO_DIR" ]; then
    continue
  fi

  FOLDER_NAME=$(basename "$VIDEO_DIR")
  SAVE_DIR="$SAVE_ROOT/$FOLDER_NAME"
  FOLDER_START_TIME=$(date +%s)

  echo "========================================"
  echo "Processing folder: $VIDEO_DIR"
  echo "Folder start time: $(date '+%Y-%m-%d %H:%M:%S')"
  echo "Save dir: $SAVE_DIR"
  echo "========================================"

  CUDA_VISIBLE_DEVICES=1 python predict.py --video_dir "$VIDEO_DIR" --tracknet_file exp/TrackNet_best.pt --inpaintnet_file exp/InpaintNet_best.pt --save_dir "$SAVE_DIR" --large_video --output_video

  python speed_analysis/stroke_zone_analysis.py --video_root "$VIDEO_DIR" --save_root "$SAVE_DIR" --save_video

  python speed_analysis/plot_speed.py --input "$SAVE_DIR"

  FOLDER_END_TIME=$(date +%s)
  FOLDER_TOTAL_TIME=$((FOLDER_END_TIME - FOLDER_START_TIME))

  echo "========================================"
  echo "Finished folder: $VIDEO_DIR"
  echo "Folder end time: $(date '+%Y-%m-%d %H:%M:%S')"
  echo "Folder total time: ${FOLDER_TOTAL_TIME} seconds"
  echo "========================================"
done

echo "Done."