# Speed Analysis

`speed_analysis/` 主要負責把 TrackNetV3 輸出的球軌跡結果進一步整理成 stroke、球速、網前速度、落點區域與視覺化影片。

目前這一套分析流程是針對這批桌球影片場景設計，也就是固定視角、固定球桌位置、球主要由左往右飛行的影片。若要換成其他場地、不同鏡頭角度或不同桌面顏色，可能會需要重新調整 table detection、net zone 與 stroke 判斷參數。

---

## 主要檔案說明

| 檔案 | 功能 |
|---|---|
| `stroke_analysis.py` | 根據球軌跡 CSV 切分 stroke，判斷每一球的 start、hit、end、bounce |
| `stroke_zone_analysis.py` | 整合 stroke detection、table / net zone detection、速度計算與落點分析，是目前主要使用的分析程式 |
| `table_tracker.py` | 偵測球桌邊線、網子位置，並建立 table corners 與 net-front zone |
| `table_analysis.py` | 單獨檢查 table / net zone 偵測結果，可輸出 zone CSV 與 overlay 影片 |
| `bounce_landing_analysis.py` | 根據 bounce frame 判斷球是否落在桌面內，並輸出落點區域 |
| `plot_speed.py` | 將 `stroke_zone.csv` 中的速度欄位畫成折線圖，方便觀察每個 stroke 的速度變化 |

> `plot_compare_speed.py` 是之前測試 raw / corr 速度比較時使用的獨立工具，目前主流程沒有呼叫

---

## 輸入資料

主要輸入包含：

```txt
*_predict.mp4
*_ball.csv
```

其中 `*_predict.mp4` 是 TrackNetV3 輸出的預測影片，`*_ball.csv` 是球軌跡資料。

`*_ball.csv` 至少需要包含以下欄位：

| 欄位 | 說明 |
|---|---|
| `Frame` | 影片 frame 編號 |
| `Visibility` | 球是否被偵測到，1 表示有效點 |
| `X` | 球心 x 座標 |
| `Y` | 球心 y 座標 |

---

## 執行方式

### 單一影片分析

```bash
python speed_analysis/stroke_zone_analysis.py --video_file path/to/xxx_predict.mp4 --ball_csv path/to/xxx_ball.csv --save_dir path/to/output_dir
```

### 整個資料夾批次分析

```bash
python speed_analysis/stroke_zone_analysis.py --video_root path/to/pred_result_folder
```

批次模式會自動尋找資料夾底下的：`*_predict.mp4`

並尋找同資料夾中對應的：`*_ball.csv`，如果找不到對應 CSV，該影片會被跳過。

---

## Stroke 切分邏輯

目前 stroke 的判斷是根據球的連續可見軌跡進行切分。程式會先找出連續 `Visibility = 1` 的區段，再從中判斷是否形成一次有效擊球。

目前主要採用的是左到右的擊球邏輯：

1. 先找到一段穩定往左移動的球軌跡，作為 stroke start 的依據。
2. 再尋找球從往左移動轉為往右移動的 turning point，作為 hit frame。
3. hit 之後持續往右移動的區段會被視為該 stroke 的後段。
4. 若球沒有明顯形成有效 hit，則會標記為 `no_hit`。
5. 若球在網前區域停止或 bounce，會額外加上 `net_hit` 或 `net_stop` note。

常見輸出欄位如下：

| 欄位 | 說明 |
|---|---|
| `stroke_id` | 第幾個 stroke |
| `frame_start` | 該 stroke 起始 frame |
| `hit_frame` | 判斷出的擊球 frame |
| `frame_end` | 該 stroke 結束 frame |
| `bounce_frame` | 判斷出的 bounce frame，若沒有則為 0 |
| `valid` | 是否為有效 stroke |
| `note` | 額外註記，例如 `no_hit`、`net_hit`、`net_stop` |

---

## Table / Net Zone 偵測

速度分析需要知道球桌位置與網前區域，因此程式會從影片畫面中偵測 table corners 與 net zone。

目前 table detection 的設計大致如下：

1. 使用影像中的球桌線條與桌面顏色特徵進行偵測。
2. 找出桌面的四個角點，作為 pixel-to-cm 轉換依據。
3. 找出網子位置。
4. 根據網子線與球桌邊界建立 6 點的 net-front zone。
5. 每個 stroke 會在 hit 附近取數個 frame 偵測 table / net zone，並取平均後作為該 stroke 的幾何資訊。

輸出的 zone detail 會存成：

```txt
xxx_zone_detail.csv
```

其中包含每個 stroke 對應的 table corners 與 net-front zone 座標。

### 重要限制

目前 table / net zone 的寫法是針對現在這批影片場景調整的，不是通用型 table detector。

也就是說，它目前比較適合：

- 固定攝影機角度
- 球桌位置大致固定
- 桌面與線條顏色明顯
- 網子位置穩定
- 球主要由左往右移動

如果換成以下情況，可能需要重新調整或改寫：

- 攝影機角度改變
- 球桌在畫面中的位置不同
- 桌面顏色、光線或背景差異太大
- 球桌邊線不清楚
- 球員或其他物件遮住球桌
- 球的移動方向不是目前假設的左到右

因此目前的 table detection 比較像是「針對本資料集場景調整過的幾何輔助方法」，而不是可以直接泛化到所有桌球影片的模型。

---

## 速度分析

速度使用的 frame 是從 hit_frame 到 bounce_frame，如果沒有 bounce_frame 就是算到 frame_end，主要為了避免 avg_speed 在計算時因彈跳後速度下降的問題

這邊的設計會有一個問題是如果是打到網子的情況，很有可能因為彈跳幅度不夠，所以沒有被判斷成 bounce_frame，這種情況的 avg_speed 會略低，但我們主要是在意 net_zone_max_speed，所以這部份沒有去進行改善

### 速度計算方式

速度計算會先把球的 pixel 位移轉換成實際距離，再根據影片 FPS 換算成 km/h。

球桌實際尺寸使用：

```txt
TABLE_W = 274.0 cm
TABLE_H = 152.5 cm
```

程式會根據偵測到的 table corners 估算 pixel 到 cm 的比例。
目前會計算：

```txt
sx_cm_per_px
sy_cm_per_px
```

| 參數 | 說明 |
|---|---|
| `sx_cm_per_px` | x 方向的 cm / pixel 比例 |
| `sy_cm_per_px` | y 方向的 cm / pixel 比例 |

實際速度計算時，會使用混合比例：

```txt
scale = sx + 0.25 * (sy - sx)
```

這樣做是為了避免單純使用 x 或 y 方向比例造成速度過低或過高。

速度公式概念如下：

```txt
distance_cm = sqrt(dx_cm^2 + dy_cm^2)
time_sec = frame_gap / fps
speed_kmh = distance_cm / time_sec * 0.036
```

其中 `0.036` 是將 `cm/s` 轉換成 `km/h` 的係數。


### 速度取樣方式

目前每個 frame 會嘗試計算多種速度候選：

| 類型 | 說明 |
|---|---|
| `1f` | 使用相鄰 1 frame 的位移計算速度。當前 frame 為 T，用的就是 T ~ T+1|
| `2f` | 使用間隔 2 frame 的位移計算速度。當前 frame 為 T，用的就是 T ~ T+2|
| `c2f` | 使用前後各 1 frame，也就是 centered 2-frame 的方式計算速度。當前 frame 為 T，用的就是 T-1 ~ T+1|

每個 frame 會從這些速度候選中選擇較大的合理速度作為該 frame 的代表速度。

目前有速度上限過濾：`MAX_SPEED_KMH = 115.0`

如果某段速度超過上限，會被視為不合理資料並排除，避免錯誤偵測點造成速度異常放大。


### 速度輸出欄位

主要速度結果會輸出在：`xxx_stroke_zone.csv`

常見欄位如下：

| 欄位 | 說明 |
|---|---|
| `avg_speed_kmh` | 該 stroke 的平均速度 |
| `max_speed_kmh` | 該 stroke 的最大速度 |
| `net_zone_max_speed_kmh` | 球通過網前區域附近的最大速度 |
| `valid` | 是否為有效 stroke |
| `note` | 額外註記 |
| `zone_label` | 落點區域 |
| `in_table` | 是否落在桌面內 |

另外會輸出更詳細的速度比較資料：`xxx_stroke_speed_compare.csv`

常見欄位如下：

| 欄位 | 說明 |
|---|---|
| `net_zone_max_speed_type` | 最後選到的速度類型，例如 `1f`、`2f`、`c2f` |
| `net_zone_max_speed_start_frame` | 該段速度的起始 frame |
| `net_zone_max_speed_end_frame` | 該段速度的結束 frame |
| `net_zone_max_speed_1f_kmh` | net zone 中 1f 計算出的最大速度 |
| `net_zone_max_speed_2f_kmh` | net zone 中 2f 計算出的最大速度 |
| `net_zone_max_speed_c2f_kmh` | net zone 中 c2f 計算出的最大速度 |
| `sx_cm_per_px` | x 方向 cm / pixel 比例 |
| `sy_cm_per_px` | y 方向 cm / pixel 比例 |


### Net Zone Max Speed

`net_zone_max_speed_kmh` 是特別關注的速度指標。它代表球通過網前區域附近時的最大速度。

計算方式大致如下：

1. 根據 table / net 偵測結果建立 net-front zone。
2. 找出 stroke 中落在 net-front zone 附近的球點。
3. 為了避免剛好少一個 frame 導致沒有算到，會把 net zone 內的點前後各擴一個 segment。
4. 在這些 segment 中選出最大的合理速度，作為 `net_zone_max_speed_kmh`。
5. 同時記錄該速度是由 `1f`、`2f` 還是 `c2f` 選出。

這個指標主要用來觀察球過網附近的速度，避免只看整段 stroke 的最大速度時，被其他位置的雜訊或錯誤點影響。

---

## 落點分析

`bounce_landing_analysis.py` 主要負責將每個 stroke 的 bounce point 轉換成實際桌面座標，並進一步判斷球是否落在桌面內，以及落在哪一個區域。

這個模組的目的不是單純記錄 bounce frame，而是把影片中的落點像素座標轉換成桌面上的實際公分座標，讓後續可以用 heatmap、落點散佈圖與 CSV 統計來觀察選手的擊球落點分布。

### 單獨使用時的資料來源

直接執行 `bounce_landing_analysis.py` 時，落點分析主要會使用兩種資料：

| 資料 | 說明 |
|---|---|
| `stroke_zone.csv` | 每個 stroke 的 `stroke_id`、`bounce_frame`，以及該 stroke 對應的桌面四角點資訊 |
| `ball.csv` | TrackNetV3 輸出的逐幀球軌跡資料，包含 `Frame`、`Visibility`、`X`、`Y` |

其中 `bounce_frame` 會用來找出該次擊球的落點 frame，再從 `ball.csv` 中取得該 frame 的球心座標。

**這部分已經融合進 `stroke_zone_analyziz.py` 中了，所以不須單獨的執行**

### Perspective Transform

因為影片是從側邊或斜角拍攝，球桌在畫面中通常會呈現梯形，而不是標準矩形。所以不能直接用像素座標判斷落點位置，必須先把畫面中的球桌轉換成真實桌面的平面座標。

目前使用 OpenCV 的 Homography 進行透視轉換：

```python
cv2.findHomography()
cv2.perspectiveTransform()
```

轉換方式是：

1. 取得該 stroke 對應的桌面四角點。
2. 將畫面中的桌面四角點對應到真實桌面尺寸。
3. 將球的像素座標 `(ball_px, ball_py)` 投影到桌面平面。
4. 得到以桌面左上角為原點的 `(x_cm, y_cm)` 公分座標。

球桌實際尺寸設定為：

```txt
TABLE_W = 274.0 cm
TABLE_H = 152.5 cm
```

也就是標準桌球桌的長與寬。

這樣做的好處是，即使每個 stroke 偵測到的桌面角點有些微差異，也可以根據該次 stroke 的桌面位置重新校正落點。

### 落點區域切分

轉換成桌面座標後，程式會將桌面切成：`6 × 3 = 18 格`

也就是：

| 方向 | 切分方式 | 用途 |
|---|---|---|
| 橫向 | 6 等分 | 區分近網、中場、底線等深度位置 |
| 縱向 | 3 等分 | 區分左、中、右路線 |

區域標籤格式為：`C1R1 ~ C6R3`

其中：

| 標籤 | 說明 |
|---|---|
| `C` | Column，代表橫向第幾格 |
| `R` | Row，代表縱向第幾格 |

例如：`C6R2`，代表落點位於第 6 欄、第 2 列。

### 容錯機制

如果 `bounce_frame` 當下球不可見，也就是 `Visibility = 0`，程式不會直接放棄，而是會往前後搜尋附近 frame 的球座標。

目前的設計是：`bounce_frame ± 2 frames`

也就是在 bounce frame 前後 2 幀內，尋找最近的可見球點。

如果找到可見球點，就使用該點作為落點座標。如果找不到，或是 perspective transform 失敗，則該 stroke 的落點會被標記為無效。

另外，如果轉換後的 `(x_cm, y_cm)` 超出桌面範圍，也會標記為：`in_table = False`

代表該球可能是出界球，或是 bounce frame / table detection 有誤。

### 輸出結果

#### `landing_heatmap.png` — 落點熱力圖

![landing_heatmap](../images/landing_heatmap.png)

這張圖會統計每個落點區域被擊中的次數。

圖中的資訊包含：

- 每一格的命中次數
- 每一格的區域編號，例如 `C1R1`、`C6R3`
- 中間的白色虛線代表球網位置
- 顏色越深代表該區域落點次數越多

這張圖主要用來快速觀察選手的落點習慣，例如是否經常打到某一側、是否集中在近網或底線區域。

#### `landing_zones.png` — 落點散佈圖

![landing_zones](../images/landing_zones.png)

這張圖會顯示每一球的實際落點位置。

圖中的資訊包含：

- 綠色方框：標準桌面範圍
- 紅色圓點：落在桌面內的球
- 灰色叉號：判斷為出界或桌面外的球
- 每個點旁邊會標示對應的 `stroke_id`

這張圖適合用來回查單一 stroke 的落點位置，比 heatmap 更適合做逐球檢查。

#### `landing_detail.csv` — 詳細落點資料

`landing_detail.csv` 會記錄每一個 stroke 的落點詳細資訊。

| 欄位 | 說明 |
|---|---|
| `stroke_id` | 擊球編號 |
| `bounce_frame` | 該 stroke 的落點 frame |
| `ball_px` | bounce frame 對應的球心 x 像素座標 |
| `ball_py` | bounce frame 對應的球心 y 像素座標 |
| `x_cm` | 轉換後的桌面 x 公分座標 |
| `y_cm` | 轉換後的桌面 y 公分座標 |
| `zone_col` | 落點所在欄位 |
| `zone_row` | 落點所在列 |
| `zone_label` | 落點區域標籤，例如 `C6R2` |
| `in_table` | 是否落在桌面內 |

### 輸出欄位整合

落點分析完成後，部分結果也會整合回主要的 `stroke_zone.csv`。

常見欄位包含：

| 欄位 | 說明 |
|---|---|
| `in_table` | 該 stroke 是否落在桌面內 |
| `zone_label` | 該 stroke 的落點區域 |
| `zone_col` | 落點所在欄 |
| `zone_row` | 落點所在列 |

因此後續分析時，可以直接從 `stroke_zone.csv` 看到每一球的速度、note、valid 狀態與落點區域。

### 應用方式

落點分析的結果可以用在：

1. 觀察選手是否有固定落點習慣。
2. 比較不同場次或不同選手的落點分布。
3. 找出攻擊集中區域或較少使用的落點區域。
4. 搭配速度分析，觀察高速球主要落在哪些位置。
5. 搭配發球、接球、正手、反手等分類，做更細的戰術分析。

### 目前限制

目前落點分析仍然依賴前面的 table detection 與 bounce detection，因此有幾個限制：

1. 如果桌面四角點偵測不準，轉換後的公分座標也會偏移。
2. 如果 `bounce_frame` 判斷錯誤，落點位置也會錯。
3. 如果球在 bounce frame 附近沒有被正確偵測，即使有 ±2 frames 的容錯，仍可能找不到正確落點。
4. 目前落點是以 2D 桌面平面投影為主，沒有估計球的 3D 高度。
5. 若影片場景、拍攝角度或桌面位置改變，可能需要重新調整 table detection 相關參數。

---

## 視覺化輸出

分析完成後會輸出一支視覺化影片：`xxx_stroke_zone_visualize.mp4`

影片中會標示：

- 目前球的位置
- stroke 軌跡
- start / hit / bounce / end
- table polygon
- net-front zone
- stroke 是否 valid

這支影片主要用來檢查 stroke 切分、hit 判斷、bounce 判斷與 net zone 是否合理。

---

## 速度折線圖

如果想把速度結果畫成折線圖，可以使用：

```bash
python speed_analysis/plot_speed.py --input path/to/xxx_stroke_zone.csv
```

預設會畫：`net_zone_max_speed_kmh`

如果要改畫其他速度欄位，可以使用 `--speed`：

```bash
python speed_analysis/plot_speed.py --input path/to/xxx_stroke_zone.csv --speed max_speed_kmh
```

也可以直接輸入資料夾，程式會自動尋找底下所有： `*_stroke_zone.csv`，並分別輸出折線圖。

![net_zone_max_speed](../images/C0050_predict_stroke_zone_net_zone_max_speed_kmh_line.png)

---

## 輸出檔案整理

執行 `stroke_zone_analysis.py` 後，常見輸出如下：

| 檔案 | 說明 |
|---|---|
| `xxx_stroke_zone.csv` | 最主要的 stroke 統計結果，包含速度、valid、note、落點等資訊 |
| `xxx_zone_detail.csv` | 每個 stroke 使用的 table corners 與 net zone 座標 |
| `xxx_stroke_speed_compare.csv` | 更詳細的速度來源與比例資訊 |
| `xxx_stroke_zone_visualize.mp4` | stroke、table、net zone 的視覺化影片 |
| `landing_heatmap.png` | 落點熱力圖，顯示各區域落點次數 |
| `landing_zones.png` | 落點散佈圖，顯示每一球的實際落點位置 |
| `landing_detail.csv` | 每個 stroke 的落點詳細資料 |
| `zone_stats.csv` | 各落點區域的統計結果 |

---

## 可調整參數

目前大部分參數已經針對這批影片調整過，通常可以先用預設值跑。只有在 stroke、table zone、net zone 或速度結果明顯不合理時，再依照下面表格調整。

### stroke_zone_analysis.py 主要參數

| 參數 | 目前預設值 | 建議範圍 | 作用 | 什麼時候調 |
|---|---:|---:|---|---|
| `min_left_segments` | `5` | `3 ~ 8` | hit 前至少要有幾段往左移動，才會被視為可能的 stroke start | 假 stroke 太多就調大；真正 stroke 被漏掉就調小 |
| `min_candidate_frames` | `52` | `35 ~ 70` | 一個 stroke 至少要持續幾個 frame 才保留 | 太短的假 stroke 太多就調大；短球被漏掉就調小 |
| `max_step_th` | `130.0` | `100 ~ 160` | 相鄰 frame 最大允許位移，避免跳點 | 球速快、容易斷就調大；跳點太多就調小 |
| `max_abs_dy_th` | `45.0` | `35 ~ 70` | hit 前往左移動時，y 方向最大允許變化 | 高拋或角度變化大就調大；亂點太多就調小 |
| `left_half_ratio` | `0.5` | `0.45 ~ 0.55` | hit frame 必須發生在畫面左半邊的比例範圍 | hit 太晚就調小；hit 抓不到就調大 |
| `right_side_ratio` | `0.5` | `0.5 ~ 0.7` | stroke end 至少要到畫面右側多少比例，才算有效 stroke | 假 stroke 太多就調大；有效 stroke 被標成 `no_hit` 就調小 |
| `zone_window` | `2` | `1 ~ 5` | 偵測 table / net zone 時，取 hit frame 前後幾個 frame 平均 | table 偵測不穩就調大；畫面變化大就調小 |
| `up_px` | `140` | `100 ~ 180` | net-front zone 往上延伸的高度 | net zone 太小抓不到速度就調大；抓到太多非網前區域就調小 |
| `left_shift_px` | `160` | `120 ~ 220` | net-front zone 往左延伸的寬度 | net zone 沒涵蓋球過網區域就調大；範圍太大就調小 |

### 速度相關固定值

| 參數 | 目前值 | 建議範圍 | 作用 | 什麼時候調 |
|---|---:|---:|---|---|
| `MAX_SPEED_KMH` | `115.0` | `100 ~ 120` | 過濾不合理的速度異常值 | 如果明顯錯點造成速度爆高，就調低；如果合理高速球被濾掉，就調高 |
| `TABLE_W` | `274.0` | 不建議調 | 標準桌球桌長度 cm | 固定值 |
| `TABLE_H` | `152.5` | 不建議調 | 標準桌球桌寬度 cm | 固定值 |
| `scale = sx + 0.25 * (sy - sx)` | `0.25` | `0.2 ~ 0.4` | x/y 比例混合，用來讓速度換算不要過低或過高 | 速度整體偏低可略調大；速度整體偏高可略調小 |

### stroke_analysis.py 內部判斷值

這些目前不是 CLI 參數，而是寫在程式內部。通常不需要改，除非 stroke 判斷明顯不穩。

| 變數 | 目前值 | 建議範圍 | 作用 |
|---|---:|---:|---|
| `right_x_th` | `frame_w * 0.65` | `0.6 ~ 0.75` | bounce 必須出現在畫面右側，避免把 hit 附近誤判成 bounce |
| `max_backward_tol` | `12.0` | `8 ~ 20` | hit 後允許少量往回移動 |
| `max_nonforward_count` | `3` | `2 ~ 5` | hit 後最多允許幾次非向右移動 |
| `local_window` | `3` | `2 ~ 5` | hit 附近檢查局部最低點的範圍 |
| `future_window` | `8` | `6 ~ 12` | hit 後檢查球是否穩定往右的範圍 |
| `min_rise_px` | `80.0` | `60 ~ 120` | hit 後 x 至少要往右增加多少 pixel |
| `min_net_right` | `3` | `2 ~ 5` | hit 後往右移動次數需大於往左移動次數多少 |
| `abs(dx) > 140` | `140` | `120 ~ 180` | 過濾 hit 後不合理 x 大跳 |
| `abs(dy) > 60` | `60` | `50 ~ 90` | 過濾 hit 後不合理 y 大跳 |

---

## 目前限制與注意事項

目前這套 speed analysis 是為了這批影片設計的，因此有幾個限制需要注意：

1. **Table detection 不是通用方法**  
   目前 table / net 偵測是依照現有影片的視角、桌面顏色、線條位置調整的。如果換場地或換鏡頭，可能需要重新調整參數。

2. **速度是 2D 平面估計**  
   目前速度主要根據畫面中的 2D 軌跡與球桌比例換算，沒有真正估計球的 3D 高度。因此球離桌面越高，透視投影造成的誤差可能越明顯。

3. **目前假設主要是左到右擊球**  
   stroke detection 的 hit 判斷主要依賴球從左移轉成右移的 turning point。如果影片方向不同，需要修改判斷邏輯。

4. **速度結果依賴球軌跡品質**  
   如果 TrackNetV3 偵測錯球、漏球、或 inpaint 補點不準，速度結果也會受到影響。

5. **net zone 速度依賴 net zone 偵測穩定度**  
   如果 table / net zone 偵測不準，`net_zone_max_speed_kmh` 可能會抓不到或抓到錯誤區段。

6. **MAX_SPEED_KMH 是人工過濾上限**  
   目前使用 `MAX_SPEED_KMH = 115.0` 過濾異常速度。這個數值是為了排除 stroke 抓取的明顯錯誤點，不代表真實球速一定不會超過此值，因為目前世界紀錄上最高球速為 116 km/h，所以訂了這個值。
