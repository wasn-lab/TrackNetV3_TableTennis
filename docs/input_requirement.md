# 輸入資料規範與注意事項

本文件說明 `TrackNetV3_TableTennis` 目前版本所需的輸入資料格式，以及影片、球點 CSV、helper table 在使用時需要注意的地方。

本系統的結果非常依賴輸入品質。  
如果影片、fps、球點 CSV 或 helper table 有問題，後續速度、落點、bounce frame 都可能不準。

> 現在這個版本一定要用 120 fps，同時球員一定要在左側向右擊球，球員一定要完整地進入畫面 (左側的球路不能斷掉)
> 如果之後換成別的 fps，predict.py 的細節 pixel 距離等設定要做修改
---

## 1. 輸入資料總覽

完整流程會用到以下輸入：

```text
原始影片 video.mp4
TrackNet 權重檔
InpaintNet 權重檔
*_ball.csv
*_helper_table.json
```

不同階段需要的輸入不同：

| 階段 | 需要輸入 | 主要輸出 |
|---|---|---|
| 球點預測 | 影片、TrackNet 權重、InpaintNet 權重 | `*_ball.csv`、`*_predict.mp4` |
| helper table 標記 | 影片或影片 frame | `*_helper_table.json` |
| 速度分析 | `*_ball.csv`、`*_helper_table.json`、影片可選 | speed detail / summary / plot / overlay |
| 落點分析 | stroke 結果、ball CSV、table corners | landing detail / heatmap / zones |

---

## 2. 影片輸入規範

## 2.1 支援格式

目前主要支援：

```text
.mp4
.MP4
```

批次模式下，`predict.py --video_dir` 會遞迴搜尋 `.mp4` / `.MP4`。  
`stroke_zone_analysis.py` 在批次分析時，也會嘗試根據 `*_ball.csv` 的檔名尋找對應影片。

---

## 2.2 影片與 CSV 必須對應

這點非常重要。

`*_ball.csv` 必須來自同一支影片，不能拿 A 影片的 CSV 去配 B 影片。  
如果影片與 CSV 不對應，會造成：

1. frame index 對不上。
2. overlay 畫錯位置。
3. fps 錯誤。
4. 速度結果錯誤。
5. bounce / landing 對不到真實畫面。

建議命名保持一致：

```text
C0086.MP4
C0086_ball.csv
C0086_helper_table.json
```

---

## 2.3 FPS 必須正確

速度計算直接依賴 fps：

```text
time_sec = frame_gap / fps
```

因此 fps 錯誤會造成速度整體錯誤。

例如：

```text
影片實際是 60 fps，但分析時用 120 fps
→ 時間被估成一半
→ 速度大約變成 2 倍
```

### 有影片時

如果有提供 `--video_file`，`stroke_zone_analysis.py` 會優先從影片讀取 fps。

### CSV-only 模式

如果只提供 CSV，沒有提供影片，程式會使用 CLI 的 `--fps` 參數。  
目前預設通常是：

```text
--fps 120.0
```

因此 CSV-only 模式下，一定要確認 `--fps` 是否正確。

---

## 2.4 影片解析度不可隨意改變

helper table 與球座標都是 pixel 座標，會綁定原始影片解析度。

如果影片有以下變動：

1. resize
2. crop
3. padding
4. 旋轉
5. 重新輸出造成解析度改變

原本的 `*_helper_table.json` 就不能直接使用，必須重新標記。

### 錯誤例子

```text
原始影片：1920x1080
helper_table.json：根據 1920x1080 標記
後來把影片縮成 1280x720
仍使用舊 helper_table.json
```

這樣桌角座標會完全錯位，速度與落點都會錯。

---

## 2.5 相機位置需固定

本系統假設同一支影片內相機視角固定。

如果影片中途有：

1. 相機移動
2. 相機震動很大
3. zoom in / zoom out
4. 畫面裁切改變
5. 桌面位置改變

則 helper table 會失效。

如果相機在不同影片之間有移動，也需要重新產生 helper table。

---

## 2.6 桌面四角盡量清楚

helper table 需要標記桌面四角。  
因此影片中桌面四角最好清楚可見。

如果桌角被遮擋，可能造成：

1. 四角點標不準。
2. homography 投影錯誤。
3. pixel-to-cm 換算錯誤。
4. 落點座標偏移。
5. near-net box 位置錯誤。

如果真的有遮擋，建議根據桌面邊線延伸合理估計，但後續結果要搭配 overlay 檢查。

---

## 2.7 球點可見性

TrackNetV3 對高速小球追蹤已經有一定能力，但下列情況仍可能影響結果：

| 情況 | 可能影響 |
|---|---|
| 球被球拍遮擋 | 球點消失或偏移 |
| 球被身體遮擋 | 連續 miss |
| 球飛出畫面 | 不應強行補點 |
| 球太模糊 | heatmap 可能不穩 |
| 背景白點很多 | 容易選錯 candidate |
| 場邊有其他球 | 多球干擾 |
| 球與桌線顏色接近 | 偵測不穩 |
| 光線反光 | 可能誤抓亮點 |

---

## 2.8 多球場景注意事項

如果畫面中同時有多顆球，系統可能會在候選點選擇時抓到錯誤球。

常見情況：

1. 發球機旁邊還有其他球。
2. 桌面或地上有白球。
3. 另一顆球進入畫面。
4. 球員手上或球拍附近有球。
5. 背景有類似球的白點。

可能造成：

```text
球點突然跳到另一顆球
→ 速度 spike
→ stroke 切分錯誤
→ bounce frame 錯誤
→ landing zone 錯誤
```

建議每次分析後一定要看 overlay video，而不是只看 CSV 數字。

---

## 3. `*_ball.csv` 輸入規範

## 3.1 必要欄位

速度分析至少需要：

```text
Frame
Visibility
X
Y
```

目前 `predict.py` 輸出的 CSV 也會包含：

```text
Inpaint_Mask
```

欄位說明：

| 欄位 | 必要 | 說明 |
|---|---|---|
| `Frame` | 是 | frame 編號 |
| `Visibility` | 是 | 是否有有效球點，1 代表有球 |
| `X` | 是 | 球中心 x pixel 座標 |
| `Y` | 是 | 球中心 y pixel 座標 |
| `Inpaint_Mask` | 否，但建議保留 | 是否為 InpaintNet 補點位置 |

---

## 3.2 `Visibility` 的意義

`Visibility = 1`：

```text
該 frame 有有效球點
```

`Visibility = 0`：

```text
該 frame 沒有可靠球點
```

在 stroke detection 中，通常會先找連續 `Visibility = 1` 的片段。  
如果中間有 `Visibility = 0`，可能會切斷 stroke。

### 注意

`Visibility = 1` 不代表一定正確。  
如果 TrackNet 誤抓了別的白點，仍可能是 `Visibility = 1`，所以要用 overlay 檢查。

---

## 3.3 X / Y 座標規範

`X` 與 `Y` 是影像 pixel 座標：

```text
X：由左到右增加
Y：由上到下增加
```

例如 1920x1080 影片：

```text
左上角 = (0, 0)
右下角 = (1919, 1079)
```

如果 X/Y 出現異常跳動，例如：

```text
前一 frame: (500, 400)
下一 frame: (1500, 800)
```

可能代表誤抓，會造成速度異常偏高。

---

## 3.4 Frame 必須連續或可合理對應

`Frame` 欄位最好從 0 或 1 開始連續遞增。  
如果中間缺 frame，stroke detection 可能會被切斷。

目前 `stroke_analysis.py` 在收集 visible runs 時，會要求：

```text
下一個 visible row 的 Frame = 目前 Frame + 1
```

如果 frame 不連續，就會斷成不同 run。

---

## 3.5 舊 CSV 覆蓋問題

如果重跑 `predict.py` 或速度分析，請確認輸出檔案是最新版本。

常見錯誤：

```text
以為已經重新跑球點
但實際 speed analysis 讀到的是舊的 *_ball.csv
```

建議：

1. 重跑前確認 `save_dir`。
2. 重跑後看檔案修改時間。
3. 必要時先刪除舊 output。
4. overlay 影片也要重新產生。

---

## 4. `*_helper_table.json` 輸入規範

## 4.1 helper table 的用途

helper table 是後續速度與落點分析的關鍵輸入。

它提供：

1. 桌面四角點。
2. 桌面幾何。
3. near-net box。
4. table polygon。
5. pixel-to-cm 換算基礎。
6. landing homography 投影所需資料。

如果 helper table 錯，後續所有幾何相關結果都會錯。

---

## 4.2 點選順序

目前 helper table 點選順序為：

```text
LF -> RF -> RB -> LB
左前 -> 右前 -> 右後 -> 左後
```

這是人工標記時需要注意的順序。

後續 `stroke_zone_analysis.py` 會轉成桌面座標使用的順序：

```text
LB -> RB -> RF -> LF
左後 -> 右後 -> 右前 -> 左前
```

如果點選順序錯，table polygon、homography、near-net box 都會錯。

---

## 4.3 每個相機視角都需要自己的 helper table

helper table 不能隨便共用。

以下情況都需要重新產生：

| 情況 | 是否需重標 |
|---|---|
| 同一支影片重新分析，沒有改解析度 | 不一定 |
| 同一場地但相機移動 | 需要 |
| 不同影片、不同視角 | 需要 |
| 影片 crop / resize | 需要 |
| 桌子位置改變 | 需要 |
| 相機 zoom 改變 | 需要 |
| 輸出影片解析度改變 | 需要 |

---

## 4.4 near-net box

目前主流程會使用 helper table 產生 near-net box。

near-net box 不是簡單的 2D 矩形，而是 helper table 產生的 8 個投影點。  
`stroke_zone_analysis.py` 會保留這 8 個點，不再壓成舊版 6 點 polygon。

輸出欄位通常包含：

```text
net_p1_x, net_p1_y
net_p2_x, net_p2_y
...
net_p8_x, net_p8_y
```

在 overlay 影片中，near-net box 會用來判斷球是否進入網前分析區域。

---

## 5. 速度分析輸入

`stroke_zone_analysis.py` 的主要輸入有：

```text
--video_file
--ball_csv
--save_dir
--helper_table_json
```

### 有影片模式

如果要輸出 overlay video，必須提供影片：

```text
--video_file path/to/video.mp4
--save_video
```

有影片時，程式會從影片讀取：

1. fps
2. frame width
3. frame height
4. total frames
5. 影像內容，用於輸出 overlay

### CSV-only 模式

如果不提供影片，只提供 CSV，也可以跑分析，但無法輸出 overlay video。

這時需要注意：

```text
--fps
--frame_w
--frame_h
```

因為沒有影片可以讀取 fps 與解析度。

---

## 6. Batch 模式輸入

批次分析時常用：

```text
--video_root
--save_root
```

概念是：

| 參數 | 說明 |
|---|---|
| `--video_root` | 原始影片所在資料夾 |
| `--save_root` | TrackNet 預測結果與 ball CSV 所在資料夾 |

程式會在 `save_root` 下尋找：

```text
*_ball.csv
*_bass.csv
```

並在同一資料夾找對應的：

```text
*_helper_table.json
```

如果要輸出視覺化影片，還需要能從 `video_root` 找到對應 MP4。

---

## 7. 建議檢查清單

每次分析前，建議照下面檢查：

```text
[ ] 影片可以正常播放
[ ] 影片 fps 已確認
[ ] 影片解析度沒有被改過
[ ] ball CSV 來自同一支影片
[ ] ball CSV 包含 Frame / Visibility / X / Y
[ ] ball CSV 不是舊檔
[ ] helper table 對應同一支影片與同一解析度
[ ] helper table 四角點順序正確
[ ] 若使用 batch，video_root 與 save_root 對應正確
[ ] 若要輸出影片，確認 ffmpeg / codec 可用
```

---

## 8. 常見輸入錯誤與影響

| 問題 | 可能結果 |
|---|---|
| fps 錯 | 速度整體偏高或偏低 |
| helper table 不對 | 速度比例、落點座標、near-net box 都錯 |
| 影片與 CSV 不對應 | overlay 與速度完全不可信 |
| X/Y 有誤抓 | 速度 spike |
| Visibility 錯 | stroke 切分錯 |
| 多球干擾 | 抓到 distractor |
| 球長時間遮擋 | stroke 被切斷 |
| 沒有影片但 fps 沒設 | CSV-only 速度不準 |
| helper table 點選順序錯 | homography 方向錯、落點 zone 錯 |
