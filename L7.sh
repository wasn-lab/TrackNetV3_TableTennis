#!/bin/bash
set -e

START_TIME=$(date +%s)

echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"

VIDEO_FILE="/home/wasn/table_tennis_video/C0077.MP4"
SAVE_DIR="/home/wasn/table_tennis_video"
NAME=$(basename "$VIDEO_FILE")
NAME="${NAME%.*}"

#python speed_analysis/helper_table.py --video "$VIDEO_FILE" --frame 2000 --save "$SAVE_DIR/${NAME}_helper_table.png" --save_corners "$SAVE_DIR/${NAME}_helper_table.json"
#python predict.py --video_file "$VIDEO_FILE" --tracknet_file exp/TrackNet_best.pt --inpaintnet_file exp/InpaintNet_best.pt --save_dir "$SAVE_DIR" --large_video --output_video
python speed_analysis/stroke_zone_analysis.py --video_file "$VIDEO_FILE" --ball_csv "$SAVE_DIR/${NAME}_ball.csv" --save_dir "$SAVE_DIR" --helper_table_json "$SAVE_DIR/${NAME}_helper_table.json"
python speed_analysis/plot_speed_bounce.py --input "$SAVE_DIR/${NAME}_stroke_zone.csv" --target_mode r12_r34
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))

echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Total time: ${TOTAL_TIME} seconds"
echo "Done."