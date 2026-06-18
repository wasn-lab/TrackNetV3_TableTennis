# 輸出檔案說明

本文件說明 `TrackNetV3_TableTennis` 目前版本會產生的主要輸出檔案，以及每個檔案的用途、欄位意義與檢查方式。

建議接手者不要只看單一 CSV 數字，而是搭配：

```text
ball CSV
speed summary CSV
speed detail CSV
speed plot
overlay video
landing outputs
```

一起確認結果是否合理。

---

## 1. 輸出檔案總覽

完整流程可能產生：

```text
*_ball.csv
*_predict.mp4
*_helper_table.json
*_ball_current_speed_detail.csv
*_ball_current_speed_summary.csv
*_ball_current_speed_plot.png
*_ball_current_speed_overlay.mp4
landing_detail.csv
zone_stats.csv
landing_heatmap.png
landing_zones.png
```

依照用途可以分成：

| 類型 | 檔案 | 用途 |
|---|---|---|
| 球點追蹤結果 | `*_ball.csv` | 每 frame 球座標 |
| 球點視覺化 | `*_predict.mp4` | 檢查 TrackNet / InpaintNet 結果 |
| 桌面標定 | `*_helper_table.json` | 桌面四角與 near-net box |
| 速度主結果 | `*_ball_current_speed_summary.csv` | 每段 stroke 的主要速度結果 |
| 速度細節 | `*_ball_current_speed_detail.csv` | 每 frame 速度與 debug 欄位 |
| 速度圖 | `*_ball_current_speed_plot.png` | 速度折線圖 |
| 速度 overlay | `*_ball_current_speed_overlay.mp4` | 檢查速度段、table、near-net box |
| 落點細節 | `landing_detail.csv` | bounce / landing 詳細資料 |
| 落點統計 | `zone_stats.csv` | 每個 zone 的落點數 |
| 落點圖 | `landing_heatmap.png`、`landing_zones.png` | 落點分布視覺化 |

---

## 2. `*_ball.csv`

## 2.1 來源

由 `predict.py` 產生。

## 2.2 用途

這是後續所有分析的基礎輸入。  
速度、stroke、bounce、landing 都依賴這份球點 CSV。

## 2.3 常見欄位

| 欄位 | 說明 |
|---|---|
| `Frame` | frame 編號 |
| `Visibility` | 是否有有效球點 |
| `X` | 球中心 x pixel 座標 |
| `Y` | 球中心 y pixel 座標 |
| `Inpaint_Mask` | 是否為 InpaintNet 補點位置 |

## 2.4 如何檢查

建議搭配 `*_predict.mp4` 檢查：

1. 球點是否跟著真正的球。
2. 是否有突然跳到其他球。
3. 是否在球飛出畫面後還繼續補點。
4. 是否有長時間消失。
5. Inpaint 補點是否合理。

如果 `*_ball.csv` 已經有誤抓，後續速度通常也會錯。

---

## 3. `*_predict.mp4`

## 3.1 來源

由 `predict.py --output_video` 產生。

## 3.2 用途

用來檢查球點追蹤結果。

影片通常會顯示：

1. 目前 frame index。
2. 目前球點。
3. 歷史軌跡。

## 3.3 什麼時候一定要看

以下情況一定要看：

| 情況 | 原因 |
|---|---|
| 速度異常偏高 | 可能有誤抓 |
| stroke 數量不對 | 可能球點中斷或切錯 |
| bounce frame 不合理 | 可能球點軌跡錯 |
| 多球場景 | 可能抓到 distractor |
| 換新影片或相機 | 需要確認 TrackNet 是否穩定 |

---

## 4. `*_helper_table.json`

## 4.1 來源

由 `speed_analysis/helper_table.py` 建立。

## 4.2 用途

提供桌面幾何資訊給後續分析使用。

主要包含：

1. 桌面四角點。
2. near-net box。
3. 投影與比例換算所需資訊。

## 4.3 對後續輸出的影響

helper table 會影響：

| 後續項目 | 影響 |
|---|---|
| pixel-to-cm 比例 | 速度距離換算 |
| near-net box | 判斷球是否進入網前速度分析區 |
| homography | bounce 點轉成桌面公分座標 |
| landing zone | 判斷落在哪個區域 |
| overlay video | 桌面與 near-net box 畫的位置 |

如果 helper table 錯，不能只修 CSV，必須重新標定。

---

## 5. `*_ball_current_speed_summary.csv`

## 5.1 定位

這是速度分析最重要的主結果檔。  
如果要整理每一球或每一段 stroke 的結果，通常看這份。

## 5.2 來源

由 `speed_analysis/stroke_zone_analysis.py` 產生。

## 5.3 用途

用來查看每個 stroke 的：

1. 起始 frame。
2. 結束 frame。
3. bounce frame。
4. near-net 最大速度。
5. 最大速度使用的方法。
6. 最大速度對應的 frame 區間。
7. landing zone。
8. 是否 valid。
9. note 註記。

## 5.4 常見欄位

| 欄位 | 說明 |
|---|---|
| `stroke_id` | stroke 編號 |
| `frame_start` | stroke 起始 frame |
| `frame_end` | stroke 結束 frame |
| `bounce_frame` | 偵測到的 bounce frame，沒有則可能為 0 |
| `net_zone_max_speed_kmh` | near-net box 附近最大速度 |
| `net_zone_max_speed_type` | 該速度來自 `1f`、`2f` 或 `c2f` |
| `net_zone_max_speed_start_frame` | 最大速度段起始 frame |
| `net_zone_max_speed_end_frame` | 最大速度段結束 frame |
| `zone_label` | 落點區域標籤 |
| `in_table` | 落點是否在桌面範圍內 |
| `valid` | 是否為有效 stroke |
| `note` | 額外註記 |

## 5.5 如何使用

### 正常報告速度

優先使用：

```text
net_zone_max_speed_kmh
```

因為它是針對 near-net box 附近取最大合理速度，比整段球路最大值更穩定。

### 檢查速度來源

看：

```text
net_zone_max_speed_type
net_zone_max_speed_start_frame
net_zone_max_speed_end_frame
```

如果速度來自 `1f`，代表使用相鄰 frame。  
如果來自 `2f` 或 `c2f`，代表使用較長 frame gap，通常對抖動較不敏感。

---

## 6. `*_ball_current_speed_detail.csv`

## 6.1 定位

這是逐 frame 速度與 debug 檔。  
它不是主要報告檔，而是用來檢查速度為什麼會這樣算。

## 6.2 來源

由 `speed_analysis/stroke_zone_analysis.py` 產生。

## 6.3 用途

當 summary 裡某個速度看起來不合理時，需要回來看 detail。

例如：

1. 為什麼某段速度特別高？
2. 最大速度是由哪兩個 frame 算出來？
3. 當下的 pixel-to-cm 比例是多少？
4. 有沒有套用高度平面補償？
5. raw speed 與 final speed 差多少？
6. 該 frame 是否被用作 net max speed？

## 6.4 常見欄位分組

### 基本欄位

| 欄位 | 說明 |
|---|---|
| `Frame` | frame 編號 |
| `X` | 球中心 x 座標 |
| `Y` | 球中心 y 座標 |
| `Visibility` | 是否有有效球點 |
| `in_net` | 是否在 near-net 分析區域 |
| `use_for_net_max` | 是否可用於 net max speed |

### 速度欄位

| 欄位 | 說明 |
|---|---|
| `speed_1f_kmh` | 相鄰 1 frame 速度 |
| `speed_2f_kmh` | 間隔 2 frames 速度 |
| `speed_c2f_kmh` | centered 2-frame 速度 |
| `best_speed_kmh` | 當前 frame 選出的最佳速度 |
| `best_speed_table_kmh` | 桌面比例下的速度 |
| `best_speed_plane_kmh` | 套用高度平面補償後的速度 |
| `best_speed_raw_before_gt_scale` | 套用 GT scale 前的速度 |
| `best_speed_type` | 使用的速度類型 |
| `best_speed_start_frame` | 速度段起始 frame |
| `best_speed_end_frame` | 速度段結束 frame |

### 比例與幾何欄位

| 欄位 | 說明 |
|---|---|
| `sx_cm_per_px_used` | x 方向使用的 cm / pixel |
| `sy_cm_per_px_used` | y 方向使用的 cm / pixel |
| `sx_table_cm_per_px` | 桌面平面 x 比例 |
| `sy_table_cm_per_px` | 桌面平面 y 比例 |
| `plane_scale_ratio` | 高度平面補償比例 |
| `blue_depth_ratio` | 藍色桌面平面的深度比例 |
| `blue_local_width_px` | 當前位置藍色平面 local width |
| `orange_local_width_px` | 高度平面 local width |
| `far_width_px` | 遠端桌邊寬度 |
| `near_width_px` | 近端桌邊寬度 |
| `avg_width_px` | 遠近桌邊平均寬度 |
| `avg_px_per_cm` | 平均 pixel/cm |
| `speed_mid_y` | 速度段中點 y |
| `speed_table_y` | 經 y offset 修正後用於查表的 y |

### 平面 debug 欄位

| 欄位 | 說明 |
|---|---|
| `orange_blue_ratio_x` | 高度平面與桌面平面 x 比例差 |
| `orange_blue_ratio_y` | 高度平面與桌面平面 y 比例差 |
| `orange_blue_ratio_len` | 高度平面與桌面平面長度比例差 |
| `blue_original_table_height_px` | 原影像桌面高度 |
| `blue_rectified_table_height_px` | rectified 後桌面高度 |

## 6.5 異常速度檢查方式

如果某段速度異常，建議：

1. 在 summary 找到 `net_zone_max_speed_start_frame` 與 `net_zone_max_speed_end_frame`。
2. 回 detail CSV 找對應 frame。
3. 檢查 X/Y 是否突然跳動。
4. 檢查 `Visibility` 是否為 1。
5. 檢查 `best_speed_raw_before_gt_scale` 是否本身就很高。
6. 檢查 `plane_scale_ratio` 是否異常。
7. 打開 overlay 影片看該 frame 是否抓錯球。

---

## 7. `*_ball_current_speed_plot.png`

## 7.1 來源

由速度分析流程或 `plot_speed.py` 產生。

## 7.2 用途

速度折線圖用來快速觀察：

1. 整體速度分布。
2. 是否出現 spike。
3. 哪些 stroke 特別快。
4. 是否有明顯誤抓造成異常點。
5. 不同影片或球種的速度趨勢。

## 7.3 如何解讀

### 正常情況

速度應該大致落在合理範圍內，並且每段 stroke 的速度變化不應有極端尖峰。

### 異常情況

如果出現很尖的單點 spike，常見原因是：

1. 單 frame 誤抓。
2. 球點突然跳到別顆球。
3. helper table 比例異常。
4. fps 設錯。
5. 速度段跨到錯誤點。

---

## 8. `*_ball_current_speed_overlay.mp4`

## 8.1 來源

由 `stroke_zone_analysis.py --save_video` 產生。

## 8.2 用途

這是檢查速度分析最直觀的輸出。

通常會畫出：

1. 目前 frame。
2. 球點。
3. stroke 軌跡。
4. table polygon。
5. near-net box。
6. near-net max speed 的線段。
7. 速度文字。
8. bounce frame。
9. landing / zone 資訊。
10. valid / note。

## 8.3 為什麼重要

CSV 數字本身看不出球點是否抓錯。  
overlay 可以直接看出：

| 問題 | overlay 中可能看到 |
|---|---|
| 球點抓錯 | 點不在真球上 |
| near-net box 錯 | 黃框不在合理位置 |
| helper table 錯 | 桌面 polygon 偏掉 |
| 速度段錯 | 速度線段連到不合理位置 |
| bounce 錯 | bounce 標記不在落桌瞬間 |
| stroke 切錯 | 軌跡段落不完整或跨多球 |

---

## 9. `landing_detail.csv`

## 9.1 來源

由 `bounce_landing_analysis.py` 產生，通常透過 `stroke_zone_analysis.py` 呼叫。

## 9.2 用途

記錄每個 stroke 的 bounce / landing 詳細結果。

## 9.3 常見欄位

| 欄位 | 說明 |
|---|---|
| `stroke_id` | stroke 編號 |
| `bounce_frame` | bounce frame |
| `ball_px` | bounce frame 的 x pixel |
| `ball_py` | bounce frame 的 y pixel |
| `x_cm` | 投影到桌面後的 x 公分 |
| `y_cm` | 投影到桌面後的 y 公分 |
| `in_table` | 是否在桌面範圍 |
| `in_table_strict` | 是否嚴格在桌面內 |
| `in_table_relaxed` | 是否在放寬 margin 後的桌面內 |
| `right_half_landing` | 是否在右半桌 |
| `bounce_type` | bounce 判斷類型 |
| `zone_label` | 落點區域 |

## 9.4 `bounce_type` 概念

目前 bounce detection 不只一種規則，可能包含：

| 類型 | 概念 |
|---|---|
| normal V-shaped bounce | 下降後上升的明顯 V 形 |
| terminal bounce | 球在 stroke 尾端落桌，後續反彈 frame 不足 |
| flat rebound | 落桌前較平，但落桌後明顯上升 |

實際欄位值依 code 輸出為準。

---

## 10. `zone_stats.csv`

## 10.1 來源

由 landing analysis 產生。

## 10.2 用途

統計每個落點區域有幾顆球。

可用於：

1. 看落點分布。
2. 比較不同球種落點。
3. 產生落點 heatmap。
4. 整理實驗結果。

---

## 11. `landing_heatmap.png`

## 11.1 用途

顯示落點熱力圖。

適合用來快速觀察：

1. 落點集中在哪裡。
2. 是否多數落在右半桌。
3. 是否有異常落點跑到桌外。
4. 不同影片 / 球種落點分布差異。

---

## 12. `landing_zones.png`

## 12.1 用途

顯示落點散佈與 zone。

適合用來檢查：

1. 單顆球落點。
2. zone 分配是否合理。
3. homography 是否偏掉。
4. 桌面切格是否符合預期。

---

## 13. 建議輸出檢查順序

每次跑完後建議照以下順序檢查：

```text
1. *_predict.mp4
2. *_helper_table.json / helper table 視覺化
3. *_ball_current_speed_overlay.mp4
4. *_ball_current_speed_plot.png
5. *_ball_current_speed_summary.csv
6. *_ball_current_speed_detail.csv
7. landing_heatmap.png / landing_zones.png
8. landing_detail.csv
```

### 為什麼不是先看 summary CSV？

因為 summary CSV 的數字可能看起來很合理，但其實球點可能抓錯。  
如果要確認結果可信，overlay video 比 CSV 更重要。

---

## 14. 哪些檔案是正式結果？

如果要整理報告，優先使用：

```text
*_ball_current_speed_summary.csv
landing_detail.csv
zone_stats.csv
landing_heatmap.png
landing_zones.png
```

如果要 debug，使用：

```text
*_ball.csv
*_ball_current_speed_detail.csv
*_predict.mp4
*_ball_current_speed_overlay.mp4
```

如果要重新分析，必要 input 是：

```text
*_ball.csv
*_helper_table.json
原始影片，若要 overlay 或自動讀 fps
```

---

## 15. 常見輸出異常與可能原因

| 輸出異常 | 可能原因 |
|---|---|
| summary 沒有速度 | 球沒有進入 near-net box，或找不到 clean speed segment |
| 速度超高 | 誤抓球點、fps 錯、helper table 錯 |
| plot 有尖峰 | 單 frame 或少數 frame 誤抓 |
| overlay 桌面歪掉 | helper table 四角點錯 |
| overlay 沒有影片 | 沒提供 `--video_file` 或影片找不到 |
| landing 都在桌外 | helper table 錯或 bounce frame 錯 |
| zone label 不合理 | homography 或桌角點錯 |
| stroke 數量太少 | Visibility 中斷、stroke detection 條件太嚴 |
| stroke 數量太多 | 誤抓點或軌跡被切碎 |
