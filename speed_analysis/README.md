# Speed Analysis

`speed_analysis/` 負責把 TrackNetV3 輸出的球軌跡 CSV 進一步轉換成桌球分析結果，包含：

- stroke 切分
- near-net box / net-zone speed 計算
- bounce frame 偵測
- landing zone 分析
- 速度折線圖
- 落點熱力圖
- 視覺化影片

目前這套流程主要針對固定相機視角的桌球影片設計。每次更換相機角度、球桌位置、影片解析度或擊球方向時，都應該重新建立 helper table，並檢查速度與落點結果是否合理。

---

## 1. 整體流程

```text
TrackNetV3 輸出的 *_ball.csv
        +
helper_table.py 輸出的 *_helper_table.json
        +
原始影片 MP4，可選
        ↓
stroke_zone_analysis.py
        ↓
*_stroke_zone.csv
*_net_zone_speed_detail.csv
landing_detail.csv
zone_stats.csv
landing_heatmap.png
landing_zones.png
*_stroke_zone_visualize.mp4，可選
```

---

## 2. 主要檔案

| 檔案 | 功能 |
|---|---|
| `helper_table.py` | 手動標記球桌四點，建立 table geometry 與 near-net box |
| `stroke_zone_analysis.py` | 主流程：切分 stroke、計算 net-zone speed、合併 bounce / landing 結果、輸出視覺化影片 |
| `stroke_analysis.py` | stroke 切分與基本軌跡判斷 |
| `bounce_landing_analysis.py` | bounce frame 偵測、桌面座標轉換、落點區域分析 |
| `net_zone_speed.py` | 獨立測試 net-zone speed 的工具，適合 GT / ball CSV 對齊驗證 |
| `plot_speed.py` | 依 `stroke_id` 畫速度折線圖 |
| `plot_speed_bounce.py` | 依球種 / target mode 輸出速度與 bounce 相關圖表 |
| `table_analysis.py` | 球桌相關分析工具 |
| `table_tracker.py` | 球桌追蹤 / table 相關工具 |

---

## 3. 輸入資料

### 3.1 `*_ball.csv`

由 `predict.py` 產生，至少需要以下欄位：

| 欄位 | 說明 |
|---|---|
| `Frame` | frame 編號 |
| `Visibility` | 是否偵測到球，`1` 代表有球 |
| `X` | 球心 x 座標 |
| `Y` | 球心 y 座標 |

可接受副檔名：

```text
*_ball.csv
*_bass.csv
```

### 3.2 `*_helper_table.json`

由 `helper_table.py` 產生，記錄球桌四個角點。主流程會固定讀取這個檔案，不再依賴自動偵測球桌，避免 table geometry 不穩造成速度與落點偏移。

### 3.3 原始影片 MP4，可選

若有提供 `--video_file`，程式會自動讀取：

- FPS
- frame width
- frame height
- 原始 frame，用於輸出視覺化影片

若沒有提供影片，也可以用 CSV-only 模式執行，但需要自行指定：

```bash
--fps 120 --frame_w 1920 --frame_h 1080
```

---

## 4. 建立 helper table

```bash
python speed_analysis/helper_table.py --video /path/to/C0081.MP4 --frame 2000 --save /path/to/C0081_helper_table.png --save_corners /path/to/C0081_helper_table.json
```

點選順序：

```text
LF -> RF -> RB -> LB
左前 -> 右前 -> 右後 -> 左後
```

程式內部會轉換成桌面座標使用的順序：

```text
LB -> RB -> RF -> LF
左後 -> 右後 -> 右前 -> 左前
```

桌面實際尺寸：

```text
TABLE_W = 274.0 cm
TABLE_H = 152.5 cm
```

桌面座標定義：

```text
LB = (0, 0)
RB = (274, 0)
RF = (274, 152.5)
LF = (0, 152.5)
```

---

## 5. 單支影片分析

### 5.1 建議版本：啟用 height-plane correction

```bash
python speed_analysis/stroke_zone_analysis.py --video_file /path/to/C0081.MP4 --ball_csv /path/to/C0081_ball.csv --save_dir /path/to/output --helper_table_json /path/to/C0081_helper_table.json --save_video --use_height_plane_scale --plane_height_cm 26 --camera_focal_scale 1.0
```

### 5.2 CPU 編碼版本

如果 `h264_nvenc` 失敗，可以改用 `libx264`：

```bash
python speed_analysis/stroke_zone_analysis.py --video_file /path/to/C0081.MP4 --ball_csv /path/to/C0081_ball.csv --save_dir /path/to/output --helper_table_json /path/to/C0081_helper_table.json --save_video --video_codec libx264 --use_height_plane_scale
```

### 5.3 CSV-only 模式

```bash
python speed_analysis/stroke_zone_analysis.py --ball_csv /path/to/C0081_ball.csv --save_dir /path/to/output --helper_table_json /path/to/C0081_helper_table.json --fps 120 --frame_w 1920 --frame_h 1080 --use_height_plane_scale
```

---

## 6. 批次分析

```bash
python speed_analysis/stroke_zone_analysis.py --video_root /path/to/original_video_root --save_root /path/to/pred_result_root --save_video --use_height_plane_scale
```

批次模式會在 `save_root` 底下遞迴尋找：

```text
*_ball.csv
*_bass.csv
```

並在相同資料夾尋找：

```text
*_helper_table.json
```

若找得到對應影片，會輸出視覺化影片；若找不到影片，仍可輸出 CSV 與落點圖。

---

## 7. Stroke 切分邏輯

`stroke_analysis.py` 會先找出連續 `Visibility = 1` 的球軌跡區段，再判斷是否形成有效 stroke。

目前主要判斷方向以左到右為主，也就是 x 座標增加的球路。大致流程如下：

```text
讀取 ball CSV
    ↓
找出連續可見球點區段
    ↓
排除過大跳點
    ↓
尋找可能的 stroke start / stroke end
    ↓
判斷是否為有效 stroke
    ↓
輸出 stroke_id、frame_start、frame_end、valid、note
```

常見欄位：

| 欄位 | 說明 |
|---|---|
| `stroke_id` | stroke 編號 |
| `frame_start` | stroke 起始 frame |
| `frame_end` | stroke 結束 frame |
| `bounce_frame` | bounce frame，若沒有偵測到則為 `0` |
| `valid` | 是否為有效 stroke |
| `note` | 額外註記，例如 `net_stop`、`no_hit`、`no_clean_speed_segment` |

---

## 8. Net Zone Speed 計算

目前主要速度指標是：

```text
net_zone_max_speed_kmh
```

它代表球通過 near-net box 附近時的最大合理速度。相較於整段 stroke 的最大速度，net-zone speed 比較不容易被遠端錯誤追蹤點影響。

---

## 9. 目前速度計算版本

### 9.1 基本概念

每一個速度段會從兩個球點計算影像位移：

```text
dx_px = x2 - x1
dy_px = y2 - y1
```

接著根據球桌 geometry 把 pixel 位移換成公分距離，再除以時間：

```text
time_sec = frame_gap / fps
speed_kmh = distance_cm / time_sec * 0.036
```

其中 `0.036` 是從 `cm/s` 轉成 `km/h` 的係數。

### 9.2 速度候選段

每個 frame 會嘗試三種速度段：

| 類型 | 說明 |
|---|---|
| `1f` | 相鄰 1 frame 的位移 |
| `2f` | 間隔 2 frame 的位移 |
| `c2f` | 前後各 1 frame 的 centered 2-frame 位移 |

每個 frame 會從合理候選中選最大的速度。最後 `net_zone_max_speed_kmh` 則是在 near-net box 附近的候選速度中取最大值。

### 9.3 方向限制

目前速度段主要保留左到右移動：

```text
x2 > x1
```

也就是 x 座標增加的速度段。若要分析右到左影片，需要修改方向判斷。

### 9.4 局部球桌寬度比例

速度換算不是只用全域固定比例，而是根據球當下位置估計 local table width。

因為桌球桌在影像中有透視變形：

- 靠近鏡頭的桌邊看起來較寬
- 遠離鏡頭的桌邊看起來較窄

所以程式會用球的 y 位置去估計當下對應的桌面寬度，再得到該位置的 cm / pixel。

### 9.5 球高度 y-offset

因為 TrackNet 偵測到的是球心，不是球在桌面上的投影點，所以程式會先將球心 y 位置往桌面方向修正：

```text
BALL_HEIGHT_Y_OFFSET_PX = 110.0
```

這個修正後的位置會用來查詢 local table width。

### 9.6 Height-plane correction

若加上：

```bash
--use_height_plane_scale
```

程式會用球桌四點建立簡化 camera model，並建立一個離桌面固定高度的 raised plane。

預設高度：

```text
plane_height_cm = 26.0
camera_focal_scale = 1.0
```

概念上會比較同一段影像位移在：

```text
blue plane  : z = 0，桌面平面
orange plane: z = h，球高度平面
```

上的距離比例，得到：

```text
plane_scale_ratio = distance(z=h) / distance(z=0)
```

最後速度會根據這個 ratio 做修正。

### 9.7 GT scale factor

目前程式最後會再乘上一個整體校正比例：

```text
SPEED_GT_SCALE_FACTOR = 0.75
```

這個值是用來把系統速度對齊發球機或人工 GT 速度。若後續有新的 GT 校正結果，可以在 `stroke_zone_analysis.py` 中調整此值。

### 9.8 速度上限

目前主流程使用：

```text
MAX_SPEED_KMH = 130.0
```

超過此值的速度段會被排除，避免錯誤追蹤點造成速度爆高。

---

## 10. 輸出檔案

執行 `stroke_zone_analysis.py` 後，常見輸出如下：

| 檔案 | 說明 |
|---|---|
| `xxx_stroke_zone.csv` | 最主要的 stroke 結果，包含 frame、速度、落點、valid、note |
| `xxx_zone_detail.csv` | 每個 stroke 使用的 table corners、near-net box 與 height-plane debug geometry |
| `xxx_net_zone_speed_detail.csv` | 每個 frame / segment 的速度細節 |
| `landing_detail.csv` | bounce / landing 詳細資料 |
| `zone_stats.csv` | 各落點區域統計 |
| `landing_heatmap.png` | 落點熱力圖 |
| `landing_zones.png` | 落點散佈圖 |
| `xxx_stroke_zone_visualize.mp4` | 視覺化影片，只有加 `--save_video` 且找到影片時才會輸出 |

---

## 11. `xxx_stroke_zone.csv` 主要欄位

| 欄位 | 說明 |
|---|---|
| `stroke_id` | stroke 編號 |
| `frame_start` | stroke 起始 frame |
| `frame_end` | stroke 結束 frame |
| `bounce_frame` | 偵測到的 bounce frame，沒有則為 `0` |
| `net_zone_max_speed_kmh` | near-net box 附近最大速度 |
| `net_zone_max_speed_type` | 最大速度來自 `1f`、`2f` 或 `c2f` |
| `net_zone_max_speed_start_frame` | 最大速度段起始 frame |
| `net_zone_max_speed_end_frame` | 最大速度段結束 frame |
| `zone_label` | 落點區域，例如 `C7R2` |
| `in_table` | 落點是否在桌面範圍內 |
| `valid` | 是否為有效 stroke |
| `note` | 額外註記 |

---

## 12. `xxx_net_zone_speed_detail.csv` 主要欄位

| 欄位 | 說明 |
|---|---|
| `Frame` | frame 編號 |
| `X`, `Y` | 球心座標 |
| `Visibility` | 是否有偵測到球 |
| `in_net` | 此 frame 是否在 near-net box 中 |
| `speed_1f_kmh` | 1-frame 速度 |
| `speed_2f_kmh` | 2-frame 速度 |
| `speed_c2f_kmh` | centered 2-frame 速度 |
| `best_speed_kmh` | 此 frame 可用候選中的最佳速度 |
| `best_speed_table_kmh` | 只使用桌面比例的速度 |
| `best_speed_plane_kmh` | height-plane correction 後的速度 |
| `best_speed_type` | 最佳速度類型 |
| `best_speed_start_frame` | 最佳速度段起始 frame |
| `best_speed_end_frame` | 最佳速度段結束 frame |
| `sx_cm_per_px_used` | 實際使用的 x 方向 cm / px |
| `sy_cm_per_px_used` | 實際使用的 y 方向 cm / px |
| `plane_scale_ratio` | height-plane correction ratio |
| `blue_depth_ratio` | blue plane rectification 得到的 depth ratio |
| `blue_local_width_px` | blue plane 當下 local table width |
| `orange_local_width_px` | orange plane 當下 local table width |
| `speed_mid_y` | 速度段中點的原始 y |
| `speed_table_y` | y-offset 修正後用來查比例的位置 |
| `use_for_net_max` | 是否被納入 net-zone 最大速度候選 |

---

## 13. Bounce / Landing 分析

`bounce_landing_analysis.py` 會針對每個有效 stroke 尋找 bounce frame。

目前 bounce 判斷不是單純找最低點，而是看影像座標中的局部 y 變化與桌面投影位置：

1. 先把 stroke 內的球點投影到桌面平面。
2. 檢查是否接近桌面範圍。
3. 使用影像中的 y 軌跡判斷是否有下降後上升的 V 形變化。
4. 若球在 stroke 尾端才落地，會使用 terminal / flat rebound 規則補足不明顯的反彈。

落點會轉換成桌面公分座標：

| 欄位 | 說明 |
|---|---|
| `ball_px` | bounce frame 的球心 x 像素座標 |
| `ball_py` | bounce frame 的球心 y 像素座標 |
| `x_cm` | 投影到桌面後的 x 公分座標 |
| `y_cm` | 投影到桌面後的 y 公分座標 |
| `zone_label` | 落點區域 |

---

## 14. 落點區域

目前落點區域會依照桌面座標切成 columns × rows 的 zone label。

範例：

```text
C7R2
```

代表第 7 欄、第 2 列。

輸出圖：

| 圖檔 | 說明 |
|---|---|
| `landing_heatmap.png` | 每個區域的落點數量熱力圖 |
| `landing_zones.png` | 落點散佈與 zone label |

---

## 15. 視覺化影片

若加上：

```bash
--save_video
```

會輸出：

```text
xxx_stroke_zone_visualize.mp4
```

影片中會畫出：

- frame index
- 球的位置與軌跡
- stroke start / bounce / end
- table polygon
- near-net box
- net-zone max speed segment
- local table width debug line
- valid / note 註記

若同時加上：

```bash
--save_height_debug
```

會額外輸出 height-plane correction 的 debug 圖，方便確認 blue plane / orange plane / scale ratio 是否合理。

---

## 16. 速度折線圖

### 16.1 一般速度圖

```bash
python speed_analysis/plot_speed.py --input /path/to/xxx_stroke_zone.csv --speed net_zone_max_speed_kmh
```

也可以輸入整個資料夾：

```bash
python speed_analysis/plot_speed.py --input /path/to/folder --speed net_zone_max_speed_kmh
```

### 16.2 速度與 bounce 圖

```bash
python speed_analysis/plot_speed_bounce.py --input /path/to/xxx_stroke_zone.csv --target_mode r12
```

`target_mode` 可依目前資料命名規則設定，例如右側 / 左側球種代號。

---

## 17. CLI 參數整理

### `stroke_zone_analysis.py`

| 參數 | 預設值 | 說明 |
|---|---:|---|
| `--video_file` | `None` | 單支影片路徑，可選 |
| `--ball_csv` | `None` | 單支 ball CSV 路徑 |
| `--save_dir` | `None` | 單支分析輸出資料夾 |
| `--video_root` | `None` | 批次模式原始影片資料夾 |
| `--save_root` | `None` | 批次模式預測結果資料夾 |
| `--helper_table_json` | `None` | helper table JSON 路徑 |
| `--save_video` | 關閉 | 是否輸出視覺化影片 |
| `--video_codec` | `h264_nvenc` | 可選 `h264_nvenc` 或 `libx264` |
| `--fps` | `120.0` | CSV-only 模式使用的 FPS |
| `--frame_w` | `1920` | CSV-only 模式畫面寬度 |
| `--frame_h` | `1080` | CSV-only 模式畫面高度 |
| `--min_left_segments` | `5` | stroke start 前需要的穩定段數 |
| `--min_candidate_frames` | `50` | 有效 stroke 最少 frame 數 |
| `--min_no_hit_candidate_frames` | `20` | no-hit 候選最少 frame 數 |
| `--max_step_th` | `300.0` | 相鄰球點最大允許位移 |
| `--max_abs_dy_th` | `45.0` | 判斷穩定段時 y 方向最大變化 |
| `--left_half_ratio` | `0.35` | no-hit 過濾用畫面比例 |
| `--right_side_ratio` | `0.5` | 有效 stroke 需要到達的右側比例 |
| `--near_dist` | helper table 預設 | near-net box 距離 |
| `--box_height` | helper table 預設 | near-net box 高度 |
| `--use_height_plane_scale` | 關閉 | 啟用 height-plane correction |
| `--plane_height_cm` | `26.0` | raised plane 高度 |
| `--camera_focal_scale` | `1.0` | 簡化 camera intrinsic focal scale |
| `--save_height_debug` | 關閉 | 輸出 height correction debug 圖 |

---

## 18. 常見 note 說明

| note | 說明 |
|---|---|
| `no_hit` | 該段軌跡沒有形成有效擊球 |
| `net_stop` | stroke 結束點落在 near-net box 內，可能是打到網或停在網前 |
| `no_clean_speed_segment` | 找不到可用的合理速度段 |
| `no_ball_in_net_zone` | 有效 stroke 沒有進入 near-net box，因此無 net-zone speed |
| `no_video_or_table_geometry` | 缺少影片資訊或 table / net geometry |

---

## 19. 目前限制

1. **主要適用固定視角**  
   換角度後必須重新建立 helper table，並檢查 near-net box。

2. **主要以左到右球路為主**  
   速度段目前主要保留 `x2 > x1`。若要分析右到左球路，需要修改方向條件或新增方向參數。

3. **不是完整 3D 測速**  
   height-plane correction 是用單相機與簡化 camera model 做比例補償，不等於雙相機 3D 重建。

4. **速度依賴球軌跡品質**  
   如果 TrackNetV3 偵測錯球、漏球或 InpaintNet 補點錯誤，速度與 bounce 都可能受到影響。

5. **GT scale factor 需要依場景校正**  
   `SPEED_GT_SCALE_FACTOR` 是整體速度校正值，不同相機、FPS、球桌位置或影片資料可能需要重新校正。

---

## 20. 建議檢查流程

每次分析新影片時，建議依序檢查：

1. `*_helper_table.png`：四個桌角是否點準。
2. `*_ball.csv` / `*_predict.mp4`：球軌跡是否追到真球。
3. `*_stroke_zone_visualize.mp4`：stroke 切分、near-net box、最大速度段是否合理。
4. `*_net_zone_speed_detail.csv`：確認 `use_for_net_max=True` 的 frame 是否在網前附近。
5. `*_stroke_zone.csv`：檢查 `net_zone_max_speed_kmh`、`net_zone_max_speed_type`、`start_frame`、`end_frame`。
6. `landing_detail.csv`：確認 bounce frame 與落點座標是否合理。
7. `landing_heatmap.png` / `landing_zones.png`：確認落點分布是否符合影片內容。
