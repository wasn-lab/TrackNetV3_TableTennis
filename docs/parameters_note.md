# 重要參數說明

本文件整理 `TrackNetV3_TableTennis` 目前版本中比較重要的參數，包含球點預測、stroke 切分、速度計算、球高補償、落點分析與影片輸出相關設定。

參數分成兩類：

1. CLI 參數：執行程式時可以從 command line 傳入。
2. code 內固定值：目前寫在 `.py` 檔中，需要修改 code 才會改變。

---

## 1. 參數調整原則

在調整速度或落點參數前，建議先確認：

```text
1. 影片 fps 是否正確
2. ball.csv 球點是否正確
3. helper_table.json 是否正確
4. overlay 影片中 table / near-net box 是否正確
5. 是否有多球誤抓
```

不要一開始就直接調速度比例。  
如果球點或 helper table 錯，調速度參數只會把錯誤結果包裝成看起來比較合理的數字。

---

## 2. `predict.py` 主要 CLI 參數

## 2.1 輸入來源

| 參數 | 說明 |
|---|---|
| `--video_file` | 單支影片路徑 |
| `--video_dir` | 整個資料夾批次預測，會遞迴搜尋 `.mp4` / `.MP4` |
| `--save_dir` | 輸出資料夾 |

### 使用建議

如果只跑一支影片，用：

```text
--video_file
```

如果要跑整個資料夾，用：

```text
--video_dir
```

批次模式會保留相對資料夾結構，適合大量影片處理。

---

## 2.2 模型權重

| 參數 | 說明 |
|---|---|
| `--tracknet_file` | TrackNet 權重檔 |
| `--inpaintnet_file` | InpaintNet 權重檔，不填則只跑 TrackNet |

### 注意

如果沒有提供 `--inpaintnet_file`，系統仍可輸出 TrackNet 預測結果，但不會使用 InpaintNet 補軌跡。

---

## 2.3 推論模式

| 參數 | 說明 |
|---|---|
| `--batch_size` | inference batch size，預設通常為 16 |
| `--eval_mode` | temporal ensemble 模式，可選 `nonoverlap`、`average`、`weight` |
| `--large_video` | 大影片模式，使用串流讀取，降低記憶體使用 |

### `eval_mode` 建議

| 模式 | 特性 |
|---|---|
| `weight` | 較穩定，但速度較慢 |
| `average` | 穩定度與速度折衷 |
| `nonoverlap` | 較快，但穩定度可能較低 |

一般建議使用：

```text
--eval_mode weight
```

因為球速分析對球點穩定度很敏感。

---

## 2.4 median background 相關參數

| 參數 | 說明 |
|---|---|
| `--max_sample_num` | 大影片產生 median background 時最多取樣幾個 frame |
| `--video_range` | 指定用影片某段秒數產生 median background |

### `--video_range` 注意

```text
--video_range 10,20
```

代表使用影片第 10 秒到第 20 秒的 frame 來產生 median background。  
它不是只預測第 10 到 20 秒，預測仍會跑完整支影片。

---

## 2.5 輸出影片參數

| 參數 | 說明 |
|---|---|
| `--output_video` | 是否輸出畫上球軌跡的影片 |
| `--traj_len` | 輸出影片中顯示幾個 frame 的歷史軌跡，預設 8 |
| `--video_codec` | 輸出影片編碼器，可選 `h264_nvenc` 或 `libx264` |

### codec 選擇

| codec | 說明 |
|---|---|
| `h264_nvenc` | NVIDIA GPU 硬體編碼，快但環境可能不支援 |
| `libx264` | CPU 編碼，較慢但相容性最高 |

如果遇到：

```text
OpenEncodeSessionEx failed
No capable devices found
ffmpeg pipe broken
```

通常可以改用：

```text
--video_codec libx264
```

---

## 3. `generate_inpaint_mask()` 相關參數

`generate_inpaint_mask()` 用來決定哪些缺失片段要交給 InpaintNet 補。

目前 README 中整理的 predict 實際呼叫值包含：

| 參數 | 目前值 | 說明 |
|---|---:|---|
| `max_gap` | `14` | 最多允許連續幾個 frame 消失仍可補 |
| `border_margin_x` | `160` | 右邊界保護範圍 |
| `max_angle_diff` | `100.0` | 長 gap 前後方向允許最大角度差 |
| `min_valid_run` | `1` | gap 前後至少需要幾個連續可見點 |
| `angle_check_min_gap` | `14` | gap 長度達到此值才做方向檢查 |
| `max_reverse_dx` | `40.0` | 長 gap 前後 x 方向反轉門檻 |

### 調整建議

| 問題 | 建議 |
|---|---|
| 很多短暫 miss 沒補到 | 調大 `max_gap` |
| 補太多不該補的洞 | 調小 `max_gap` |
| 球出畫面還被補回來 | 調大 `border_margin_x` |
| 邊界附近合理球點補不到 | 調小 `border_margin_x` |
| 長 gap 不是同一顆球卻被補 | 調小 `max_angle_diff` 或 `max_reverse_dx` |
| 合理長 gap 沒補 | 調大 `max_angle_diff` 或 `max_reverse_dx` |

---

## 4. `select_best_candidate()` 相關參數

`select_best_candidate()` 用來從 heatmap 候選點中選出最合理的球點。

目前 README 中整理的 function 參數包含：

| 參數 | 目前值 / 預設 | 說明 |
|---|---:|---|
| `min_area_no_history` | `6.0` | 沒有 history 時的候選點最小面積 |
| `min_area_with_history` | `2.0` | 有 history 時的候選點最小面積 |
| `min_y` | `150` | 候選點 y 座標下限 |
| `max_y` | `900` | 候選點 y 座標上限 |
| `HISTORY_SIZE` | `8` | history 最多保留最近 8 筆 |

### 調整方向

| 問題 | 可能調整 |
|---|---|
| 背景小白點常被抓 | 提高 `min_area_*` |
| 真球很小常被過濾 | 降低 `min_area_*` |
| 球在畫面上方被忽略 | 降低 `min_y` |
| 球在畫面下方被忽略 | 提高 `max_y` |
| history 影響太久 | 降低 `HISTORY_SIZE` |
| 軌跡不夠穩 | 提高 `HISTORY_SIZE`，但要小心錯誤延續 |

---

## 5. tracking reset 相關參數

`should_reset_track()` 用來判斷是否要重置追蹤狀態。

目前 README 中整理的 function 預設值包含：

| 參數 | 預設值 | 說明 |
|---|---:|---|
| `border_margin` | `40` | 邊界判斷範圍 |
| `stale_frames` | `6` | 判斷 stale 需要連續幾 frame |
| `stale_avg_step_thresh` | `6.5` | 平均移動過小的門檻 |
| `stale_y_span_thresh` | `12.0` | y 方向變化過小門檻 |
| `stale_x_span_thresh` | `35.0` | x 方向變化過小門檻 |

### stale reset 後

目前 `predict.py` 會記錄：

```text
ignore_stale_until = current_frame + 80
ignore_stale_pos = last_valid_position
```

這表示 stale reset 後，短時間內會避免再抓回同一個舊位置附近的候選點。

---

## 6. `stroke_zone_analysis.py` CLI 參數

## 6.1 輸入輸出

| 參數 | 說明 |
|---|---|
| `--video_file` | 單支影片路徑 |
| `--ball_csv` | TrackNet 輸出的球點 CSV |
| `--save_dir` | 輸出資料夾 |
| `--helper_table_json` | helper table JSON |
| `--video_root` | 批次模式下原始影片根目錄 |
| `--save_root` | 批次模式下預測結果根目錄 |

---

## 6.2 CSV-only 模式參數

| 參數 | 預設值 | 說明 |
|---|---:|---|
| `--fps` | `120.0` | 沒有影片時使用的 fps |
| `--frame_w` | `1920` | 沒有影片時使用的畫面寬度 |
| `--frame_h` | `1080` | 沒有影片時使用的畫面高度 |

### 注意

如果有提供 `--video_file`，程式會優先使用影片 fps 與解析度。  
如果沒有影片，這些參數一定要設定正確。

---

## 6.3 stroke 切分參數

| 參數 | 預設值 | 說明 |
|---|---:|---|
| `--min_left_segments` | `5` | stroke start 前至少需要幾段穩定左移 |
| `--min_candidate_frames` | `50` 或依 code 版本 | 有效 stroke 至少需要的 frame 長度 |
| `--min_no_hit_candidate_frames` | `20` | no_hit 候選至少需要的 frame 長度 |
| `--max_step_th` | `300.0` 或依 code 版本 | 相鄰 frame 最大允許位移 |
| `--max_abs_dy_th` | `45.0` | 找左移段時 y 方向允許最大變化 |
| `--left_half_ratio` | `0.35` | no_hit 左側過濾比例 |
| `--right_side_ratio` | `0.5` | 有效 stroke 需要到達的右側畫面比例 |

### 調整建議

| 問題 | 建議 |
|---|---|
| stroke 被切太碎 | 調大 `max_step_th`，或檢查球點是否中斷 |
| 錯誤點被接進同一段 | 調小 `max_step_th` |
| 有效球被漏掉 | 降低 `min_candidate_frames` 或檢查方向條件 |
| no_hit 太多 | 提高 no_hit 條件或檢查誤抓 |
| 有些球還沒到右側就結束 | 降低 `right_side_ratio` 或檢查影片方向 |

---

## 6.4 near-net box 相關參數

| 參數 | 說明 |
|---|---|
| `--near_dist` | near-net box 距離設定 |
| `--box_height` | near-net box 高度設定 |

這兩個參數會影響 near-net speed 的判斷區域。  
如果 near-net box 太大，可能包含太多不相關球點。  
如果太小，可能導致球沒有進入 net zone，summary 中出現 `no_ball_in_net_zone`。

---

## 6.5 影片輸出參數

| 參數 | 說明 |
|---|---|
| `--save_video` | 是否輸出 speed analysis overlay video |
| `--video_codec` | 影片編碼器，預設通常為 `h264_nvenc` |

若 GPU 編碼失敗，使用：

```text
--video_codec libx264
```

---

## 7. 速度計算固定參數

以下是目前 `stroke_zone_analysis.py` 中重要固定值。

| 參數 | 目前值 | 說明 |
|---|---:|---|
| `TABLE_W` | `274.0` | 桌球桌長度，單位 cm |
| `TABLE_H` | `152.5` | 桌球桌寬度，單位 cm |
| `MAX_SPEED_KMH` | `130.0` | 速度上限，超過會視為異常候選 |
| `SPEED_GT_SCALE_FACTOR` | `0.75` | 最後速度校正比例 |
| `ROBUST_NET_TOP_N` | `1` | near-net speed 直接取最大有效候選 |

---

## 8. 球高補償相關參數

目前 `stroke_zone_analysis.py` 有球高近似補償設計。

| 參數 | 目前值 | 說明 |
|---|---:|---|
| `BALL_HEIGHT_Y_OFFSET_PX` | `110.0` | 將球中心 y 位置修正到較接近桌面投影 |
| `DEFAULT_PLANE_HEIGHT_CM` | `26.0` | 預設球高平面高度 |
| `DEFAULT_CAMERA_FOCAL_SCALE` | `1.0` | 相機高度補償比例 |
| `DEFAULT_PLANE_DEBUG_MAX_HEIGHT_CM` | `50.0` | debug 平面高度上限 |
| `ORANGE_RECT_SCALE_PX_PER_CM` | `4.0` | orange plane debug rectified image 比例 |
| `ORANGE_SY_PROBE_PX` | `20.0` | orange plane y 方向 probe pixel |

### 參數概念

`BALL_HEIGHT_Y_OFFSET_PX`：

```text
球中心在影像中通常位於桌面投影點上方或下方，因此用 y offset 修正查表位置。
```

`DEFAULT_PLANE_HEIGHT_CM`：

```text
假設球在桌面上方某個平行平面，用來估計高度造成的比例差。
```

`DEFAULT_CAMERA_FOCAL_SCALE`：

```text
控制高度補償強度。
```

### 注意

這些參數是單相機近似補償，不是真正 3D 量測。  
換相機角度後需要重新驗證。

---

## 9. GT Scale 參數

目前最後速度會乘上：

```text
SPEED_GT_SCALE_FACTOR = 0.75
```

也就是：

```text
final_speed = raw_speed * 0.75
```

這個值會直接影響所有輸出速度。

### 使用注意

1. 這是最後校正比例。
2. 它不會改變 ball.csv。
3. 它會改變 detail / summary 中的 final speed。
4. 如果換場地、相機、fps、helper table，應重新驗證。
5. 如果有測速儀或發球機 ground truth，應用新的資料重新估計。

### 不建議

不要看到某支影片速度偏高就直接改這個值。  
應該先檢查：

```text
球點是否抓錯
fps 是否正確
helper table 是否正確
near-net box 是否合理
height correction 是否異常
```

---

## 10. `bounce_landing_analysis.py` 固定參數

目前落點分析中重要參數如下。

## 10.1 桌面與格線

| 參數 | 目前值 | 說明 |
|---|---:|---|
| `TABLE_W` | `274.0` | 桌面長度 cm |
| `TABLE_H` | `152.5` | 桌面寬度 cm |
| `GRID_COLS` | `8` | x 方向切成 8 欄 |
| `GRID_ROWS` | `4` | y 方向切成 4 列 |

因此目前落點區域概念上是：

```text
8 × 4 = 32 zones
```

---

## 10.2 table margin

| 參數 | 目前值 | 說明 |
|---|---:|---|
| `TABLE_MARGIN_X_CM` | `10.0` | x 方向放寬桌面邊界 |
| `TABLE_MARGIN_Y_CM` | `15.0` | y 方向放寬桌面邊界 |
| `TERMINAL_TABLE_MARGIN_X_CM` | `20.0` | terminal bounce 額外放寬 x 邊界 |
| `TERMINAL_TABLE_MARGIN_Y_CM` | `15.0` | terminal bounce y 邊界 |

### 為什麼需要 margin？

因為球在空中，直接用桌面 homography 投影時，可能會落在桌面邊界外一點點。  
margin 是為了避免合理落點被過度排除。

---

## 10.3 右半桌限制

| 參數 | 目前值 | 說明 |
|---|---:|---|
| `RIGHT_HALF_ONLY` | `True` | 只保留右半桌落點 |
| `RIGHT_HALF_MARGIN_CM` | `0.0` | 右半桌判定 margin |

目前判斷概念：

```text
x_cm >= TABLE_W / 2
```

如果要分析左半桌或雙方向影片，需要修改這個限制。

---

## 10.4 bounce detection 參數

| 參數 | 目前值 | 說明 |
|---|---:|---|
| `SMOOTH_IMAGE_Y_WINDOW` | `3` | image Y 平滑窗口 |
| `FIT_WINDOW` | `6` | bounce 前後局部 fitting 最大窗口 |
| `MIN_SEGMENT_POINTS` | `3` | 局部線段最少點數 |
| `MIN_PRE_POINTS` | `3` | bounce 前至少點數 |
| `MIN_POST_POINTS` | `2` | bounce 後至少點數 |
| `POST_EXTRA_FRAMES` | `8` | stroke end 後額外支援 frame |
| `MAX_FRAME_GAP_IN_FIT` | `4` | 局部 fitting 最大 frame gap |
| `MAX_LOCAL_STEP_PX` | `220.0` | 局部同一軌跡最大 pixel jump |

### 概念

bounce detection 會看局部軌跡是否像：

```text
落下 → 觸桌 → 彈起
```

在影像座標中通常是：

```text
Y 增加 → 局部最大 → Y 減少
```

---

## 10.5 normal bounce 門檻

| 參數 | 目前值 | 說明 |
|---|---:|---|
| `MIN_DROP_BEFORE_PX_FLOOR` | `3.0` | bounce 前最小下降量 |
| `MIN_RISE_AFTER_PX_FLOOR` | `6.0` | bounce 後最小上升量 |
| `MIN_DROP_BEFORE_RATIO` | `0.08` | 依局部 y range 計算下降比例 |
| `MIN_RISE_AFTER_RATIO` | `0.10` | 依局部 y range 計算上升比例 |
| `MIN_SLOPE_PX_PER_FRAME` | `0.6` | 最小斜率 |
| `LOCAL_PEAK_TOL_PX` | `2.0` | 局部 peak 容許誤差 |
| `MIN_RIGHTWARD_MOVE_PX` | `5.0` | bounce 附近需有右移趨勢 |

---

## 10.6 terminal bounce 參數

terminal bounce 是處理球在 stroke 尾端才落桌，後面沒有足夠反彈 frame 的情況。

| 參數 | 目前值 | 說明 |
|---|---:|---|
| `TERMINAL_LOOKBACK_POINTS` | `10` | 尾端往回看的點數 |
| `TERMINAL_MIN_DROP_BEFORE_PX` | `18.0` | terminal bounce 前最小下降量 |
| `TERMINAL_MIN_PRE_SLOPE` | `1.2` | terminal bounce 前最小下降斜率 |
| `TERMINAL_PLATEAU_TOL_PX` | `8.0` | terminal plateau 容許值 |
| `TERMINAL_POST_RISE_TOL_PX` | `8.0` | post rise 容許值 |
| `TERMINAL_LAST_RAW_DY_MAX` | `8.0` | 最後 raw dy 最大容許值 |
| `TERMINAL_MAX_POST_DOWN_SLOPE` | `2.0` | 避免候選點後仍明顯下落 |

---

## 10.7 flat rebound 參數

flat rebound 是處理落桌前 y 變化不明顯，但落桌後上升明顯的情況。

| 參數 | 目前值 | 說明 |
|---|---:|---|
| `FLAT_REBOUND_LOOKBACK_POINTS` | `12` | flat rebound 往回看的點數 |
| `FLAT_PRE_SLOPE_ABS_MAX` | `1.0` | bounce 前斜率絕對值上限 |
| `FLAT_MAX_DROP_BEFORE_PX` | `6.0` | bounce 前最大下降 |
| `FLAT_MIN_RISE_AFTER_PX` | `18.0` | bounce 後最小上升 |
| `FLAT_POST_SLOPE_MAX` | `-1.0` | bounce 後上升斜率條件 |
| `FLAT_FUTURE_REDESCEND_TOL_PX` | `10.0` | 防止未來又明顯下降 |

---

## 11. 參數調整流程建議

如果結果不合理，建議按照以下順序：

```text
1. 看 *_predict.mp4，確認球點是否正確
2. 看 *_ball.csv，確認 X/Y/Visibility 是否異常
3. 看 helper table，確認四角與 near-net box 是否正確
4. 看 speed overlay，確認速度線段是否連到正確球點
5. 看 speed detail，確認 raw speed、plane ratio、GT scale
6. 看 landing overlay / landing_detail，確認 bounce frame
7. 最後才調參數
```

---

## 12. 常見問題與對應參數

| 問題 | 優先檢查 | 可能調整 |
|---|---|---|
| 球點常漏 | TrackNet / Inpaint_Mask | `max_gap` |
| 補到出界球 | Inpaint_Mask | `border_margin_x` |
| 常抓錯背景球 | candidate selection | `min_area_*`、`min_y`、`max_y` |
| 追蹤停在舊位置 | stale reset | `stale_frames`、`ignore_stale_until` |
| stroke 被切太碎 | ball.csv / jump threshold | `max_step_th` |
| 有效 stroke 太少 | stroke 條件 | `min_candidate_frames`、`right_side_ratio` |
| 速度太高 | 球點 / fps / helper table | 先不要直接改 `SPEED_GT_SCALE_FACTOR` |
| 沒有 net speed | near-net box | `near_dist`、`box_height` |
| bounce 抓不到 | 軌跡品質 / bounce 門檻 | `FIT_WINDOW`、`MIN_*` |
| 落點跑到桌外 | helper table / margin | `TABLE_MARGIN_*` |
