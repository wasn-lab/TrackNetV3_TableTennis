# 系統架構與設計說明

本文件說明 `TrackNetV3_TableTennis` 目前版本的整體系統架構、資料流、主要模組設計，以及各模組之間的關係。  
本文件偏向「系統設計說明」，不是單純的執行教學；執行方式請參考根目錄 `README.md` 與 `speed_analysis/README.md`。

---

## 1. 專案目標

本專案是從 TrackNetV3 修改而來，主要目標是讓原本的高速小球追蹤模型可以應用在桌球影片上，並進一步加入桌球分析需要的後處理功能。

目前系統主要支援：

1. 對桌球影片進行球點追蹤。
2. 輸出每一個 frame 的球座標 CSV。
3. 使用 InpaintNet 補回短暫缺失的球點。
4. 使用 helper table 取得桌面幾何與 near-net box。
5. 根據球點軌跡切分 stroke。
6. 計算每個 stroke 在 near-net 區域附近的最大球速。
7. 進行 bounce / landing 分析。
8. 輸出速度 CSV、落點 CSV、統計圖、速度圖與 overlay 影片。
9. 支援單支影片與整個資料夾批次處理。

目前系統是 **單相機 2D 幾何估計 + 桌面比例校正 + 球高近似補償 + GT scale 校正**，不是完整的雙相機 3D 重建。

---

## 2. 整體資料流

目前主要流程可以整理成：

```text
Input Video
    ↓
predict.py
    ↓
TrackNetV3 ball detection
    ↓
Candidate selection / tracking post-process
    ↓
Inpaint mask generation
    ↓
InpaintNet trajectory recovery
    ↓
*_ball.csv
    ↓
helper_table.py
    ↓
*_helper_table.json
    ↓
stroke_zone_analysis.py
    ↓
stroke detection
    ↓
net-zone speed estimation
    ↓
bounce / landing analysis
    ↓
CSV outputs + plots + overlay video
```

更具體地說：

```text
video.mp4
  ├── predict.py
  │     ├── TrackNet inference
  │     ├── temporal ensemble
  │     ├── candidate selection
  │     ├── reset / stale filtering
  │     ├── generate Inpaint_Mask
  │     ├── InpaintNet recovery
  │     └── *_ball.csv
  │
  ├── speed_analysis/helper_table.py
  │     └── *_helper_table.json
  │
  └── speed_analysis/stroke_zone_analysis.py
        ├── *_ball_current_speed_detail.csv
        ├── *_ball_current_speed_summary.csv
        ├── *_ball_current_speed_plot.png
        ├── *_ball_current_speed_overlay.mp4
        ├── landing_detail.csv
        ├── zone_stats.csv
        ├── landing_heatmap.png
        └── landing_zones.png
```

---

## 3. 根目錄主要模組

## 3.1 `predict.py`

`predict.py` 是球點預測的主程式，負責讀取影片、載入模型、執行 TrackNetV3 / InpaintNet，並輸出球點 CSV 與可選的預測影片。

目前支援：

| 功能 | 說明 |
|---|---|
| 單支影片預測 | 使用 `--video_file` |
| 整個資料夾批次預測 | 使用 `--video_dir`，會遞迴搜尋 `.mp4` / `.MP4` |
| 大影片模式 | 使用 `--large_video`，避免一次載入整部影片 |
| temporal ensemble | 支援 `nonoverlap`、`average`、`weight` |
| 輸出預測影片 | 使用 `--output_video` |
| 影片編碼選擇 | `--video_codec h264_nvenc` 或 `--video_codec libx264` |
| timing report | 顯示資料讀取、GPU inference、後處理、輸出影片等耗時 |

### `predict.py` 的輸入

```text
video_file 或 video_dir
TrackNet 權重
InpaintNet 權重，可選
save_dir
```

### `predict.py` 的輸出

```text
*_ball.csv
*_predict.mp4，若有加 --output_video
```

### `*_ball.csv` 欄位

目前 `write_pred_csv()` 會輸出：

| 欄位 | 說明 |
|---|---|
| `Frame` | frame 編號 |
| `Visibility` | 是否有有效球點，1 代表有球，0 代表無球 |
| `X` | 球中心 x 座標 |
| `Y` | 球中心 y 座標 |
| `Inpaint_Mask` | 是否為需要 InpaintNet 補點的位置 |

---

## 3.2 `dataset.py`

`dataset.py` 負責影片 frame 讀取與 Dataset 建立。

目前比較重要的是 `Video_IterableDataset`，它用在 `--large_video` 模式。

### 設計目的

原本如果一次把整部影片載入記憶體，長影片會很容易爆 RAM。  
因此目前大影片模式使用串流讀取與 sliding window 的方式處理影片。

### 目前特性

| 功能 | 說明 |
|---|---|
| 串流讀取影片 | 不一次讀完整部影片 |
| deque sliding window | 保留模型需要的連續 frame |
| median background 抽樣 | 不一定要用所有 frame 產生 median |
| 支援 `--max_sample_num` | 控制最多取樣 frame 數 |
| 支援 `--video_range` | 指定用哪一段影片產生 median background |
| background mode 預處理加速 | 針對 `concat`、`subtract`、`subtract_concat` 等模式降低重複運算 |

---

## 3.3 `utils/general.py`

`utils/general.py` 是通用工具函式集合，目前與輸出有關的修改較重要。

主要包含：

| 函式 / 功能 | 說明 |
|---|---|
| `write_pred_csv()` | 輸出球點 CSV |
| `write_pred_video()` | 輸出畫上球點與軌跡的影片 |
| `FFmpegWriter` | 使用 ffmpeg subprocess 輸出 mp4 |
| 模型建立工具 | 建立 TrackNet / InpaintNet |
| 影片讀取工具 | 輔助讀取影片資訊 |

目前影片輸出支援：

```text
h264_nvenc：NVIDIA GPU 硬體編碼，速度快但環境不一定支援
libx264：CPU 編碼，較慢但相容性高
```

如果遇到 NVENC 錯誤，通常不是模型推論錯，而是輸出 mp4 時 ffmpeg 編碼失敗。

---

## 3.4 `test.py`

`predict.py` 會使用 `test.py` 中的部分後處理 function。

目前 README 中提到會引用：

| function | 用途 |
|---|---|
| `get_ensemble_weight()` | temporal weighted ensemble |
| `generate_inpaint_mask()` | 決定哪些缺失片段要給 InpaintNet 補 |
| `predict_location_candidates()` | 從 heatmap 中找出多個候選球點 |
| `select_best_candidate()` | 從候選球點中選出最合理的球 |
| `should_reset_track()` | 判斷是否要 reset tracking 狀態 |

---

## 4. 球點追蹤後處理設計

桌球影片的困難點在於球很小、速度快、容易有殘影，且場景中可能有反光、白點、其他球或背景干擾。  
因此目前系統不是單純取 heatmap 最大點，而是加入後處理流程。

---

## 4.1 Candidate Selection

TrackNet heatmap 可能同時出現多個候選點，因此系統會先從 heatmap 中取出候選球點，再根據歷史軌跡選出最合理者。

目前流程概念：

```text
TrackNet heatmap
    ↓
predict_location_candidates()
    ↓
最多保留數個候選球點
    ↓
select_best_candidate()
    ↓
根據 history、距離、方向、候選點大小等條件選出一個球點
```

### 設計目的

1. 避免只取最大 heatmap 區塊時誤抓背景白點。
2. 避免球短暫消失後直接跳到錯誤候選。
3. 盡量維持軌跡連續性。
4. 減少多球或殘影干擾。

### 需要注意

candidate selection 仍然不是百分之百可靠。  
如果畫面中同時有多顆球，或真正的球短暫不可見，系統仍可能選到錯誤候選點。

---

## 4.2 Tracking Reset / Stale Filter

追蹤過程中，如果系統持續停在某個錯誤位置，會造成後續 frame 都被錯誤 history 影響。  
因此目前有 `should_reset_track()` 判斷是否需要 reset。

### reset 的概念

```text
目前 history
    ↓
判斷是否 stale / 是否靠近邊界 / 是否異常
    ↓
如果需要 reset
    ↓
清空或暫時忽略舊軌跡
    ↓
從新的候選點重新開始追蹤
```

### stale reset 後的設計

目前 `predict.py` 會使用類似以下狀態：

```text
history：最近球點紀錄
miss_count：連續 miss frame 數
ignore_stale_until：reset 後暫時忽略舊位置到哪一個 frame
ignore_stale_pos：stale reset 發生時的舊位置
```

這樣做的原因是：  
如果系統剛剛判定某個位置是 stale ball，就不應該在下一個 frame 又馬上選回同一個錯誤位置。

---

## 4.3 Inpaint Mask

`generate_inpaint_mask()` 的目的是決定哪些 `Visibility = 0` 的 frame 要交給 InpaintNet 補點。

不是所有缺失點都應該補。  
例如：

| 情況 | 是否應該補 |
|---|---|
| 球真的飛出畫面 | 不應補 |
| 影片開頭球還沒出現 | 不應補 |
| 影片結尾球已離開 | 不應補 |
| 中間短暫漏偵測 | 可以補 |
| 前後軌跡明顯不是同一顆球 | 不應補 |

目前設計重點：

```text
只補「前後都有合理球點」的短暫缺失片段
```

也就是：

```text
有球 → 短暫消失 → 有球
```

才比較適合交給 InpaintNet 補。

---

## 5. `speed_analysis/` 模組設計

`speed_analysis/` 是本專案後半段的分析模組，主要負責把球點 CSV 轉成 stroke、速度、落點與視覺化結果。

主要檔案：

| 檔案 | 功能 |
|---|---|
| `stroke_zone_analysis.py` | 主流程，整合 stroke 切分、速度計算、landing 分析與視覺化 |
| `stroke_analysis.py` | stroke 切分 helper |
| `bounce_landing_analysis.py` | bounce / landing 分析 |
| `helper_table.py` | 建立桌面四角與 near-net box |
| `plot_speed.py` | 將速度結果畫成折線圖 |
| `plot_speed_bounce.py` | 速度與 bounce 相關繪圖 |
| `check_fps.py` | 檢查影片 FPS |
| `export_overlay_video.py` | 輸出 overlay 輔助工具 |

---

## 5.1 `helper_table.py`

`helper_table.py` 用來建立桌面幾何資訊。

它的核心用途是：

1. 取得桌球桌四個角點。
2. 建立 near-net box。
3. 將這些資訊輸出成 `*_helper_table.json`。
4. 讓 `stroke_zone_analysis.py` 後續讀取使用。

### helper table 點選順序

目前 helper table 點選順序為：

```text
LF -> RF -> RB -> LB
左前 -> 右前 -> 右後 -> 左後
```

但後續轉成桌面座標時，會改成：

```text
LB -> RB -> RF -> LF
左後 -> 右後 -> 右前 -> 左前
```

### 桌面世界座標

目前桌面尺寸固定使用：

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

## 5.2 `stroke_analysis.py`

`stroke_analysis.py` 負責 stroke 切分。

### 基本流程

```text
讀取 ball.csv
    ↓
找出連續 Visibility = 1 的 visible runs
    ↓
在每段 run 內尋找穩定往左移動的起點
    ↓
檢查後續是否形成往右移動
    ↓
若形成有效擊球，輸出 valid stroke
    ↓
若沒有形成有效擊球，但符合條件，輸出 no_hit
```

### 目前 stroke 判斷重點

目前的 stroke detection 主要是針對「球先往左，再往右」的軌跡設計。  
也就是先找到一段穩定左移作為 stroke start，再確認後續是否有往右移動並到達右側畫面。

### 重要條件

| 條件 | 說明 |
|---|---|
| `Visibility = 1` | 只在可見球點中切分 |
| frame 連續 | 中間若 frame 不連續，會切段 |
| `max_step_th` | 相鄰球點距離太大視為跳點 |
| `min_left_segments` | 起點前需要穩定左移 |
| `right_side_ratio` | 有效 stroke 需要到達右側區域 |
| `min_candidate_frames` | 有效 stroke 最短長度 |
| `min_no_hit_candidate_frames` | no_hit 候選最短長度 |

### `note` 可能值

| note | 說明 |
|---|---|
| 空字串 | 正常有效 stroke |
| `no_hit` | 沒有形成有效擊球，但保留為 no_hit 片段 |
| `net_stop` | 可能停在 near-net box 或打到網 |
| `no_clean_speed_segment` | 找不到合理速度段 |
| `no_ball_in_net_zone` | 沒有進入 near-net box |

---

## 5.3 `stroke_zone_analysis.py`

`stroke_zone_analysis.py` 是 speed analysis 的主流程。

它整合：

1. 讀取 ball CSV。
2. 讀取影片資訊或使用 CSV-only 模式。
3. 讀取 helper table。
4. 建立 table polygon 與 near-net box。
5. 呼叫 `detect_strokes_from_runs()` 切分 stroke。
6. 計算每個 frame / 每個 stroke 的速度候選。
7. 選出 near-net box 附近最大合理速度。
8. 呼叫 `bounce_landing_analysis.py` 合併落點分析。
9. 輸出 detail CSV、summary CSV、plot 與 overlay video。

### 目前重要固定值

```text
TABLE_W = 274.0
TABLE_H = 152.5
MAX_SPEED_KMH = 130.0
SPEED_GT_SCALE_FACTOR = 0.75
BALL_HEIGHT_Y_OFFSET_PX = 110.0
ROBUST_NET_TOP_N = 1
DEFAULT_PLANE_HEIGHT_CM = 26.0
DEFAULT_CAMERA_FOCAL_SCALE = 1.0
DEFAULT_PLANE_DEBUG_MAX_HEIGHT_CM = 50.0
```

### 速度候選

每個 frame 會嘗試多種速度段：

| 類型 | 說明 |
|---|---|
| `1f` | 相鄰 1 frame 的速度 |
| `2f` | 間隔 2 frame 的速度 |
| `c2f` | centered 2-frame 速度，即前後各 1 frame |

### near-net speed

目前主要速度指標是：

```text
net_zone_max_speed_kmh
```

它代表該 stroke 在 near-net box 附近的最大合理速度。  
這比直接取整段 stroke 的最大速度更穩定，因為遠端誤抓點或畫面邊界錯誤比較不容易影響 near-net speed。

---

## 5.4 `bounce_landing_analysis.py`

`bounce_landing_analysis.py` 負責 bounce frame 與落點分析。

目前模組做兩件事：

1. 在每個 stroke 中尋找 bounce frame。
2. 將 bounce 點轉成桌面公分座標，輸出落點細節、落點熱力圖與統計。

### bounce 判斷概念

bounce 判斷不是單純找最低點，而是使用 image-space 的 Y 軌跡變化搭配局部規則。

典型 bounce 在影像中會出現：

```text
球往下掉：image Y 增加
碰桌後往上彈：image Y 減少
```

因此 bounce 附近會像 V 形或局部轉折。

### 使用 table projection 的目的

`bounce_landing_analysis.py` 會使用 homography 把 bounce 點投影到桌面座標：

```text
pixel coordinate → table cm coordinate
```

但這裡的投影主要用來：

1. 判斷落點是否接近桌面。
2. 判斷是否在右半桌。
3. 計算落點的 x_cm / y_cm。
4. 分配落點 zone。

它不是完整 3D 高度估計。

### 目前落點格線

目前 code 中設定：

```text
GRID_COLS = 8
GRID_ROWS = 4
```

因此桌面會被切成：

```text
8 columns × 4 rows = 32 zones
```

zone label 的實際輸出格式會依程式實作決定，但概念上是依據 `x_cm` 與 `y_cm` 對應到欄列。

### 右半桌限制

目前設定：

```text
RIGHT_HALF_ONLY = True
```

代表落點分析主要只保留網子右半邊的落點，也就是：

```text
x_cm >= TABLE_W / 2
```

這符合目前針對指定方向球路分析的設計。

---

## 6. 速度估計設計

## 6.1 基本公式

球速基本概念是：

```text
速度 = 真實距離 / 時間
```

其中時間由 frame gap 與 fps 得到：

```text
time_sec = frame_gap / fps
```

影像中的球點位移是 pixel，因此需要先轉成真實距離：

```text
pixel displacement → cm displacement
```

最後轉成 km/h：

```text
speed_kmh = distance_cm / time_sec * 0.036
```

其中 `0.036` 是從 `cm/s` 轉成 `km/h` 的係數。

---

## 6.2 Pixel-to-cm 換算

桌球桌在影像中有透視變形，因此不能用單一固定比例換算所有位置。

同樣是 1 pixel：

```text
靠近鏡頭的區域代表的真實距離較小
遠離鏡頭的區域代表的真實距離較大
```

因此目前系統會根據：

1. 桌面四角點。
2. 桌球桌真實長寬。
3. 球在影像中的 y 位置。
4. 遠端桌邊寬度。
5. 近端桌邊寬度。
6. 當下位置的 local table width。
7. 深度補償比例。
8. 球高平面補償比例。

來估計該位置附近比較合理的 cm / pixel 比例。

---

## 6.3 深度補償 Depth Ratio

由於桌球桌在影像中有明顯透視效果，遠端桌邊看起來較窄，近端桌邊看起來較寬。  
如果直接使用固定比例計算球速，會導致不同深度位置的速度估計不一致。

因此目前系統會使用桌面幾何估計球所在位置的深度比例，也就是 detail CSV 中的：

```text
blue_depth_ratio
```

這個值可以理解為：

```text
球目前所在 y 位置，相對於桌面近端 / 遠端透視變化的比例
```

它的用途是補償影像深度造成的比例差異。

概念上：

```text
遠端桌邊比較窄 → 同樣 pixel 位移代表較大的真實距離
近端桌邊比較寬 → 同樣 pixel 位移代表較小的真實距離
```

因此系統不只使用固定的 `cm_per_px`，而是根據球點所在的 y 位置，估計該位置的桌面 local width，再推算較合理的 pixel-to-cm 比例。

目前 detail CSV 中與深度補償相關的欄位包含：

```text
blue_depth_ratio
blue_local_width_px
far_width_px
near_width_px
avg_width_px
avg_px_per_cm
speed_mid_y
speed_table_y
```

其中：

| 欄位 | 說明 |
|---|---|
| `blue_depth_ratio` | 根據桌面平面估計出的深度補償比例 |
| `blue_local_width_px` | 球所在 y 位置附近的桌面寬度 pixel |
| `far_width_px` | 遠端桌邊在影像中的寬度 |
| `near_width_px` | 近端桌邊在影像中的寬度 |
| `avg_width_px` | 遠端與近端桌邊平均寬度 |
| `avg_px_per_cm` | 平均 pixel/cm |
| `speed_mid_y` | 速度段中間位置的 y |
| `speed_table_y` | 經過 y offset 修正後，用來查表的 y 位置 |

這個深度補償主要是處理「球在桌面不同深度位置時，pixel-to-cm 比例不同」的問題。  
它屬於桌面平面上的透視深度補償，和下一節的球高平面補償不同。

---

## 6.4 球高平面補償 Height-plane Correction

球點偵測到的是影像中的球中心，但球實際上在桌面上方，不一定位於桌面平面。

如果只使用桌面平面計算，會假設球永遠貼在桌面上。  
但實際上球在空中時，影像中的球中心與它在桌面上的投影位置會有偏差，因此需要加入球高補償。

目前 `stroke_zone_analysis.py` 中有球高補償設計：

```text
BALL_HEIGHT_Y_OFFSET_PX = 110.0
DEFAULT_PLANE_HEIGHT_CM = 26.0
DEFAULT_CAMERA_FOCAL_SCALE = 1.0
```

概念是：

1. 先用 `BALL_HEIGHT_Y_OFFSET_PX` 將偵測到的球中心 y 位置修正到較接近桌面投影的位置。
2. 再建立一個假想的球高平面，也就是桌面上方約 `DEFAULT_PLANE_HEIGHT_CM` 的平面。
3. 比較桌面平面與球高平面在影像中的比例差。
4. 將這個比例作為速度修正的一部分。

目前 detail CSV 中與球高平面補償相關的欄位包含：

```text
plane_scale_ratio
orange_blue_ratio_x
orange_blue_ratio_y
orange_blue_ratio_len
orange_local_width_px
blue_original_table_height_px
blue_rectified_table_height_px
```

其中：

| 欄位 | 說明 |
|---|---|
| `plane_scale_ratio` | 球高平面相對於桌面平面的速度比例補償 |
| `orange_blue_ratio_x` | 高度平面與桌面平面在 x 方向的比例差 |
| `orange_blue_ratio_y` | 高度平面與桌面平面在 y 方向的比例差 |
| `orange_blue_ratio_len` | 高度平面與桌面平面整體長度比例差 |
| `orange_local_width_px` | 球高平面在該位置的 local width |
| `blue_original_table_height_px` | 原始影像中的桌面高度 |
| `blue_rectified_table_height_px` | rectified 後的桌面高度 |

這部分可以理解成：

```text
blue plane = 原本桌面平面
orange plane = 假想的球高平面
```

系統會比較 blue plane 與 orange plane 的比例差，得到球高造成的速度補償。

需要注意的是，這仍然是單相機近似方法，不是真正的 3D 重建。  
目前系統沒有直接恢復球的真實三維座標，也沒有估計每一顆球在每一 frame 的實際高度。

---

## 6.5 最終速度選擇

目前系統會同時計算多種 frame gap 的速度候選：

```text
speed_1f_kmh
speed_2f_kmh
speed_c2f_kmh
```

大致意義如下：

| 欄位 | 說明 |
|---|---|
| `speed_1f_kmh` | 相鄰 1 frame 的速度 |
| `speed_2f_kmh` | 間隔 2 frames 的速度 |
| `speed_c2f_kmh` | centered 2-frame 速度 |

系統會根據目前的規則選出較合理的速度，存成：

```text
best_speed_kmh
best_speed_type
best_speed_start_frame
best_speed_end_frame
```

其中：

| 欄位 | 說明 |
|---|---|
| `best_speed_kmh` | 最終選出的速度 |
| `best_speed_type` | 該速度來自 `1f`、`2f` 或 `c2f` |
| `best_speed_start_frame` | 速度段起始 frame |
| `best_speed_end_frame` | 速度段結束 frame |

速度也會經過上限過濾，目前 code 中設定：

```text
MAX_SPEED_KMH = 130.0
```

如果速度候選超過這個值，通常會被視為不合理候選，避免單點誤抓造成極端速度。

---

## 6.6 GT Scale

目前最後速度會套用：

```text
SPEED_GT_SCALE_FACTOR = 0.75
```

也就是：

```text
final_speed = raw_speed * SPEED_GT_SCALE_FACTOR
```

這個值用於讓估計速度更接近外部 ground truth 或校正資料。

需要注意：

1. `SPEED_GT_SCALE_FACTOR` 是最後一層整體比例校正。
2. 它不會解決球點誤抓或 helper table 標錯的問題。
3. 如果更換相機、場地、影片 fps、helper table 或速度驗證方式，這個值都應重新驗證。
4. 若有新的發球機、測速儀或人工 ground truth 資料，應重新估計這個比例。

---

## 6.7 整體速度修正流程

目前速度估計可以整理成以下流程：

```text
球點 pixel 座標
    ↓
計算 frame 間 pixel 位移
    ↓
根據 fps 計算時間
    ↓
根據桌面四角點取得桌面 pixel-to-cm 比例
    ↓
根據球所在 y 位置套用深度補償 blue_depth_ratio
    ↓
根據 y offset 修正球中心到桌面投影位置
    ↓
根據球高平面套用 plane_scale_ratio
    ↓
得到 raw speed
    ↓
套用 SPEED_GT_SCALE_FACTOR
    ↓
得到 final speed
```

簡化來看：

```text
final_speed
= pixel displacement
× table cm/px scale
× depth compensation
× height-plane compensation
÷ time
× GT scale
```

因此目前速度不是只靠一個固定比例算出來，而是同時考慮：

1. 影像位移。
2. fps。
3. 桌面真實尺寸。
4. 桌面透視深度。
5. 球中心 y offset。
6. 球高平面補償。
7. 最終 GT scale。

---

## 7. 視覺化設計

目前系統支援兩類影片視覺化：

## 7.1 TrackNet 預測影片

由 `predict.py --output_video` 輸出。

通常會畫出：

1. frame index。
2. 目前球點。
3. 歷史軌跡。

用途是檢查 TrackNet / InpaintNet 的球點是否合理。

---

## 7.2 Speed Analysis Overlay

由 `stroke_zone_analysis.py --save_video` 輸出。

通常會畫出：

1. table polygon。
2. near-net box。
3. stroke 軌跡。
4. stroke start / bounce / end。
5. near-net max speed 段。
6. speed label。
7. valid / note 資訊。

用途是確認速度、落點與 stroke 判斷是否符合實際影片。

---

## 8. 設計限制

目前系統有以下限制：

1. 主要依賴單相機畫面，沒有完整 3D 重建。
2. 若球高度變化很大，桌面平面換算會有誤差。
3. 若 helper table 四角點不準，速度與落點都會偏。
4. 若影片 fps 錯誤，速度會整體錯誤。
5. 若畫面中有多顆球，candidate selection 仍可能選錯。
6. 若 TrackNet 誤抓，1-frame speed 會容易產生 spike。
7. 目前 stroke detection 主要針對特定球路方向設計，換方向可能需要調整條件。
8. near-net box 是根據 helper table 與目前場景設計，換相機或場地後可能需要重標。

---

## 9. 維護建議

接手維護時，建議優先檢查：

1. `predict.py` 是否能正常輸出 `*_ball.csv`。
2. `*_ball.csv` 的球點是否合理。
3. `*_helper_table.json` 是否對應正確影片。
4. `stroke_zone_analysis.py` 是否能產生 summary / detail CSV。
5. overlay 影片中的 table、near-net box、速度線段是否畫在合理位置。
6. `SPEED_GT_SCALE_FACTOR` 是否符合目前驗證資料。
7. 若速度異常，先檢查球點與 helper table，不要直接調速度比例。
