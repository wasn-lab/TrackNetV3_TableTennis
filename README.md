# TrackNetV3 for Table Tennis

TrackNetV3 主要由兩個模型組成：
- 第一個模型負責追蹤影片中球的位置，產生初步軌跡
- 第二個模型則負責對漏偵測或不連續的軌跡進行 inpainting 補點

[模型下載](https://1drv.ms/u/c/ab3b33d5410e04f3/IQCwzwpuGP6pSpgw0VyyRSCzAa4jTyVFYiFWUgSd8gPeCf0?e=hWQh7G)

- Develop Environment

```text
vast.ai 更新中
GPU environment is recommended
```

- Clone this repository.

```bash
git clone https://github.com/wasn-lab/TrackNetV3_TableTennis.git
cd TrackNetV3_TableTennis
```

- Create environment.

```bash
conda create -n tracknetV3 python=3.8
conda activate tracknetV3
```

- Install the requirements.
```bash
pip install -r requirements.txt
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
```

---

## 主要內容

從原版 TrackNetV3 修改而來，目前主要修改重點放在：

- `predict.py`
  - 支援單支影片預測
  - 支援整個資料夾批次預測
  - 支援輸出預測影片
  - 支援大影片模式
  - 修改 inpaint mask 的補洞條件

- `utils/general.py`
  - 修改輸出影片格式
  - 修改預測軌跡顯示方式
  - 修改 `generate_inpaint_mask()`

- `test.py`
  - 主要保留原版 evaluation / testing 流程
  - 目前也放了一些 `predict.py` 會引用的後處理 function，例如 `get_ensemble_weight()`、`generate_inpaint_mask()`、`predict_location_candidates()`、`select_best_candidate()`、`should_reset_track()` 等

train 相關流程基本沿用原版 TrackNetV3，之後如果要重新訓練，可以參考[TrackNetV3 原始專案](https://github.com/qaz812345/TrackNetV3)

[speed analysis](./speed_analysis) 的詳細介紹另外放在 `speed_analysis/`中

---

## 檔案說明

| 檔案 / 資料夾 | 用途 |
|---|---|
| `predict.py` | 主要預測程式，用來對影片產生球的位置 csv，也可以輸出畫上軌跡的影片 |
| `test.py` | 原版測試與 evaluation 流程，包含 heatmap 轉座標、ensemble 權重、評估指標等 |
| `train.py` | 訓練 TrackNet / InpaintNet，基本沿用原版 |
| `dataset.py` | Dataset 與影片 frame 讀取相關設定 |
| `model.py` | TrackNet 與 InpaintNet model 定義 |
| `utils/general.py` | 通用工具函式，例如模型建立、影片讀取、csv 輸出、影片輸出、inpaint mask |
| `preprocess.py` | 原版資料前處理 |
| `generate_mask_data.py` | 產生 InpaintNet 訓練用的 mask data |
| `correct_label.py` | 修正 label 相關工具 |
| `error_analysis.py` | 原版 error analysis 介面 |
| `requirements.txt` | 環境套件 |
| `speed_analysis/` | 速度、落點、stroke 分析 |

---

## predict.py 使用方式

### 單支影片預測

```bash
python predict.py --video_file 048/C0045.mp4 --tracknet_file exp/TrackNet_best.pt --inpaintnet_file exp/InpaintNet_best.pt --save_dir 048 --eval_mode weight --output_video --large_video
```

### 整個資料夾預測

```bash
python predict.py --video_dir /home/code-server/NO3 --tracknet_file exp/TrackNet_best.pt --inpaintnet_file exp/InpaintNet_best.pt --save_dir /home/code-server/NO3/pred_result --eval_mode weight --output_video --large_video
```

### predict.py 參數說明

| 參數 | 說明 |
|---|---|
| `--video_file` | 單支影片路徑 |
| `--video_dir` | 整個資料夾批次預測 |
| `--tracknet_file` | TrackNet 權重檔 |
| `--inpaintnet_file` | InpaintNet 權重檔，不填則只跑 TrackNet |
| `--batch_size` | inference batch size，預設 16 |
| `--eval_mode` | temporal ensemble 模式，可選 `nonoverlap`、`average`、`weight` |
| `--max_sample_num` | 大影片產生 median background 時最多取樣幾個 frame |
| `--video_range` | 指定用哪一段影片秒數產生 background，例如 `324,330` |
| `--save_dir` | 輸出資料夾 |
| `--large_video` | 大影片模式，使用 IterableDataset，避免記憶體爆掉 |
| `--output_video` | 是否輸出畫上軌跡的影片 |
| `--traj_len` | 輸出影片中顯示幾個 frame 的歷史軌跡，預設 8 |

### 輸出檔案

每支影片會輸出：

| 檔案 | 說明 |
|---|---|
| `影片名稱_ball.csv` | 每一 frame 的球座標 |
| `影片名稱_predict.mp4` | 如果有加 `--output_video`，會輸出畫上球軌跡的影片 |

csv 格式：

| 欄位 | 說明 |
|---|---|
| `Frame` | frame 編號 |
| `Visibility` | 是否有偵測到球，1 代表有球，0 代表無球 |
| `X` | 球的 x 座標 |
| `Y` | 球的 y 座標 |
| `Inpaint_Mask` | 有沒有做 inpaint ，1 代表有，0 代表沒有 |

---

## 修改 / 新增的核心邏輯
這部分主要說明本專案為了讓 TrackNetV3 更適合桌球影片，額外修改或新增的後處理邏輯。主要包含：

- `generate_inpaint_mask()`：決定哪些缺失軌跡要交給 InpaintNet 補
- `select_best_candidate()`：當同一 frame 有多個候選球點時，選出最可能是真球的位置
- `should_reset_track()`：判斷目前是不是追錯球，是否需要重新開始追蹤
- `write_pred_video()`：輸出預測影片，方便檢查軌跡結果
- `predict.py`：支援單支影片與整個資料夾批次預測

### `generate_inpaint_mask()`：控制哪些缺失片段要補

`generate_inpaint_mask()` 的作用是產生 InpaintNet 要補的 mask。TrackNet 預測後，可能會有一些 frame 沒有偵測到球，也就是：

```csv
Visibility = 0
X = 0
Y = 0
```

但不是所有 `Visibility = 0` 都應該補。例如：

- 球真的飛出畫面，不應該補
- 影片開頭還沒出現球，不應該補
- 影片結尾球已經離開畫面，不應該補
- 中間短暫 miss，才適合交給 InpaintNet 補

所以這邊的設計是只補「前後都有球」的短暫缺失片段，也就是：`有球 → 短暫消失 → 有球`，這種情況才會補。


#### Function 參數

```python
def generate_inpaint_mask(
    pred_dict,
    frame_w,
    frame_h,
    max_gap=8,
    border_margin_x=150,
    max_angle_diff=100.0,
    min_valid_run=1,
    angle_check_min_gap=4,
    max_reverse_dx=40.0,
):
```

上面是 function 的預設值。

目前 `predict.py` 實際呼叫時使用的是：

```python
tracknet_pred_dict['Inpaint_Mask'] = generate_inpaint_mask(
    tracknet_pred_dict,
    frame_w=w,
    frame_h=h,
    max_gap=14,
    border_margin_x=160,
    max_angle_diff=100.0,
    min_valid_run=1,
    angle_check_min_gap=14,
)
```


#### 參數說明

| 參數 | function 預設值 | predict.py 目前值 | 說明 |
|---|---:|---:|---|
| `pred_dict` | - | - | TrackNet 預測結果，包含 `Frame`、`X`、`Y`、`Visibility` |
| `frame_w` | - | 原影片寬度 | 用來判斷右邊界 |
| `frame_h` | - | 原影片高度 | 目前保留參數，但這版主要使用 `frame_w` 判斷右邊界 |
| `max_gap` | `8` | `14` | 最多允許連續幾個 frame 消失還可以補 |
| `border_margin_x` | `150` | `160` | 右邊界保護範圍，gap 前後球點太靠近右邊界時不補 |
| `max_angle_diff` | `100.0` | `100.0` | 較長 gap 前後移動方向允許的最大角度差 |
| `min_valid_run` | `1` | `1` | gap 前後至少需要幾個連續可見點 |
| `angle_check_min_gap` | `4` | `14` | gap 長度達到這個值才做方向檢查 |
| `max_reverse_dx` | `40.0` | `40.0` | 較長 gap 前後 x 方向明顯反轉時的判斷門檻 |


#### 參數調整建議

| 問題 | 建議調整 |
|---|---|
| 很多短暫 miss 沒有被補 | 調大 `max_gap` |
| 補太多不該補的長洞 | 調小 `max_gap` |
| 球快到右邊界時被錯補 | 調大 `border_margin_x` |
| 右邊界附近明明還在畫面內卻補不到 | 調小 `border_margin_x` |
| 長 gap 前後不是同一顆球卻被補 | 調小 `max_angle_diff` 或 `max_reverse_dx` |
| 長 gap 明明合理但沒有補 | 調大 `max_angle_diff` 或 `max_reverse_dx` |
| 想讓補洞更保守 | 調大 `min_valid_run` |
| 想讓比較短的 gap 也檢查方向 | 調小 `angle_check_min_gap` |
| 短 gap 被方向限制擋掉 | 調大 `angle_check_min_gap` |


#### 補洞流程

```text
讀取 TrackNet 預測結果
↓
找到 Visibility = 0 的連續缺失片段
↓
確認缺失片段不是影片開頭或結尾
↓
確認 gap 長度沒有超過 max_gap
↓
確認 gap 前後都有可見球點
↓
確認 gap 前後球點沒有太靠近右邊界
↓
確認 gap 前後有足夠的連續可見點
↓
如果是短 gap，直接標記為需要補
↓
如果是長 gap，額外檢查方向角度與 x 方向反轉
↓
產生 Inpaint_Mask
```


#### 目前設計重點

這個版本是為了高速桌球影片調整過的補洞邏輯。

主要想法是：

```text
短 gap 優先補
長 gap 才做方向檢查
右邊界附近不補，避免把已經出畫面的球補回來
```

目前 `predict.py` 使用 `max_gap=14`、`angle_check_min_gap=14`，代表 14 frame 以內的 gap 才有機會補，而且小於 14 frame 的 gap 會比較直接地被補；達到 14 frame 的 gap 才會額外檢查方向角度與 x 方向反轉。

桌球球速快，frame 之間位移可能比較大。 如果補洞條件太嚴格，很多真正的短暫 miss 會補不到。


#### 注意事項

`generate_inpaint_mask()` 只負責決定哪些 frame 要補，不負責重新選球，也不負責判斷哪一顆候選球才是正確的球。

如果前後的球點本來就選錯，InpaintNet 也會根據錯誤點去補。

因此整體結果會受到前面流程影響：

```text
TrackNet heatmap
↓
predict_location_candidates()
↓
select_best_candidate()
↓
generate_inpaint_mask()
↓
InpaintNet 補軌跡
```

`generate_inpaint_mask()` 的重點不是把所有 `Visibility = 0` 都拿去補，而是只補「前後軌跡合理的短暫缺失」。

---

### `select_best_candidate()`：從多個候選球點中選出最合理的位置

`select_best_candidate()` 是用來處理同一個 frame 中有多個候選球點的情況。TrackNet 輸出的 heatmap 可能會偵測到多個亮點，例如：

```text
真正的球
背景中的白點
殘影
球桌反光
遠處其他球
```

如果只選 heatmap 中面積最大的點，很容易在球消失、球速快、背景干擾多的情況下選錯。所以這裡的設計不是單純選最大的 candidate，而是會根據前面的軌跡 `history` 判斷哪一個候選點最合理。


#### 前一階段：`predict_location_candidates()`

在進入 `select_best_candidate()` 之前，`predict.py` 會先從 heatmap 裡取出多個候選點：

```python
MAX_CANDIDATES = 3

candidates = predict_location_candidates(
    heatmap,
    max_candidates=MAX_CANDIDATES,
)
```

也就是每個 frame 最多先保留 3 個候選球點，再交給 `select_best_candidate()` 判斷哪一個最合理。


#### Function 參數

```python
def select_best_candidate(
    candidates,
    history,
    miss_count=0,
    min_area_no_history=6.0,
    min_area_with_history=2.0,
    min_y=350,
    max_y=900,
    debug=False,
):
```


#### 參數說明

| 參數 | 目前值 | 說明 |
|---|---:|---|
| `candidates` | - | 當前 frame 從 heatmap 找到的候選球點，最多 3 個 |
| `history` | - | 前面 frame 的球軌跡紀錄，格式為 `(x, y, visibility)` |
| `miss_count` | `0` | 目前已經連續 miss 幾個 frame |
| `min_area_no_history` | `6.0` | 沒有歷史軌跡時，候選點最小面積限制 |
| `min_area_with_history` | `2.0` | 有歷史軌跡時，候選點最小面積限制 |
| `min_y` | `350` | 候選點 y 座標下限 |
| `max_y` | `900` | 候選點 y 座標上限 |
| `debug` | `False` | 是否輸出 debug 訊息 |

另外 `predict.py` 裡有一個 history 長度限制：`HISTORY_SIZE = 8`

代表 `history` 最多只保留最近 8 筆追蹤狀態，避免太久以前的軌跡影響目前選點。


#### 基本選點流程

```text
讀取當前 frame 的 candidates
↓
如果沒有 candidates，回傳 None
↓
先用 min_y / max_y 過濾候選點
↓
如果過濾後沒有 candidates，回傳 None
↓
從 history 取出過去 visibility = 1 的有效球點
↓
如果沒有有效 history，用面積選球
↓
如果有有效 history，根據上一個球點與預測位置篩選 candidates
↓
排除不合理的 y 跳動
↓
排除不合理的 x 跳動
↓
排除方向突然反轉的 candidate
↓
選出最接近預測位置的 candidate
```


#### y 範圍限制

一開始會先用 `min_y` 和 `max_y` 過濾 candidates：

```python
candidates = [c for c in candidates if min_y <= c["cy"] <= max_y]
```

目前設定：

```python
min_y = 350
max_y = 900
```

也就是說，只有 y 座標在這個範圍內的候選點才會被考慮。

這個限制主要是為了排除背景中的錯誤亮點，例如：

```text
太上方的亮點
太下方的反光
畫面中不可能是球的位置
```

#### 沒有 history 時的選點方式

如果目前還沒有任何有效歷史軌跡：

```python
if not valid_history:
```

就不能根據前一個球點或方向來判斷。

這時候會改用 candidate 面積判斷：

```python
valid_candidates = [c for c in candidates if c["area"] >= min_area_no_history]
return max(valid_candidates, key=lambda c: c["area"])
```

目前設定：`min_area_no_history = 6.0`

意思是沒有歷史軌跡時，只接受 `area >= 6` 的 candidate。 如果通過條件的候選點有多個，就選面積最大的那個。

這通常會發生在：

```text
影片一開始
reset 之後
前面都沒有成功偵測到球
```

#### 有 history 時的選點方式

如果已經有有效球點，會使用上一個球的位置作為判斷基準：

```python
last_x, last_y = valid_history[-1]
```

如果至少有兩個歷史點，會計算上一段的移動方向：

```python
hist_dx = last_x - prev_x
hist_dy = last_y - prev_y
```

然後估計下一個 frame 的預測位置：

```python
pred_x = last_x + hist_dx
pred_y = last_y + hist_dy
```

也就是假設球會大致延續上一段的移動方向。 後面選 candidate 時，會優先選接近這個預測位置的點。


#### 根據 `miss_count` 放寬 x 方向距離

`miss_count` 會影響 x 方向允許的最大跳動距離：

```python
if miss_count == 0:
    max_x_gap = 130.0
elif miss_count <= 3:
    max_x_gap = 350.0
else:
    max_x_gap = 550.0
```

整理如下：

| `miss_count` 狀況 | `max_x_gap` | 意義 |
|---|---:|---|
| `miss_count == 0` | `130.0` | 沒有 miss，球應該離上一點不會太遠 |
| `miss_count <= 3` | `350.0` | 短暫 miss，允許球移動更遠 |
| `miss_count > 3` | `550.0` | miss 較久，允許更大的 x 位移 |

這樣設計的原因是球如果連續幾個 frame 沒被偵測到，重新出現時會離上一個位置比較遠，所以 miss 越久，x 方向限制會越寬鬆。


#### candidate 過濾條件

對每一個 candidate，會依序檢查以下條件。

##### 1. 面積太小不選

```python
if area < min_area_with_history:
    continue
```

目前設定：`min_area_with_history = 2.0`

有 history 時，候選點面積至少要大於等於 2。

##### 2. y 方向跳太遠不選

```python
if y_to_last > 100:
    continue
```

如果候選點和上一個球點的 y 距離超過 100 pixel，就不選。

##### 3. x 方向跳太遠不選

```python
if x_to_last > max_x_gap:
    continue
```

如果候選點和上一個球點的 x 距離超過目前允許的 `max_x_gap`，就不選。

##### 4. 沒有 miss 時，避免方向突然反轉

只有在以下條件成立時才會檢查：

```python
len(valid_history) >= 2 and miss_count == 0
```

如果前一段 x 方向是往右，但這個 candidate 突然往左太多，就不選：

```python
if hist_dx > 12 and dx < -12:
    continue
```

反過來也一樣：

```python
if hist_dx < -12 and dx > 12:
    continue
```

##### 5. 沒有 miss 時，candidate 不能離預測位置太遠

同樣只在以下情況檢查：

```python
len(valid_history) >= 2 and miss_count == 0
```

如果 candidate 離預測位置太遠，就不選：

```python
if x_to_pred > 120 or y_to_pred > 80:
    continue
```

目前限制是：

| 距離 | 最大允許值 |
|---|---:|
| `x_to_pred` | `120` |
| `y_to_pred` | `80` |

#### reset 後的選點方式

如果 `should_reset_track()` 判斷需要 reset，`predict.py` 會把選點用的 history 清空：

```python
select_history = [] if need_reset else track_state["history"]
select_miss_count = 0 if need_reset else track_state["miss_count"]
```

意思是 reset 後不再用舊軌跡限制 candidate，而是讓 `select_best_candidate()` 用「沒有 history」的方式重新選球。

這樣可以避免錯誤軌跡一直影響後面的選點。

#### 最後如何選出 best candidate

通過所有條件後，會從剩下的 `valid_candidates` 裡選出最合理的一個：

```python
best = min(
    valid_candidates,
    key=lambda item: (
        item["x_to_pred"] if item["x_to_pred"] is not None else item["x_to_last"],
        item["y_to_pred"] if item["y_to_pred"] is not None else 9999,
        item["x_to_last"],
        -item["area"],
    )
)
```

優先順序可以理解成：

```text
1. 優先選最接近預測 x 位置的 candidate
2. 再看誰比較接近預測 y 位置
3. 再看誰比較接近上一個球點
4. 如果都差不多，選面積比較大的
```

也就是說，這裡不是單純選最大面積，而是優先選軌跡最合理的點。

#### 參數調整建議

| 問題 | 建議調整 |
|---|---|
| 常常選到畫面上方或下方的背景點 | 調整 `min_y` / `max_y` |
| 沒有 history 時一開始容易選錯 | 調大 `min_area_no_history` |
| 有 history 時小雜訊被選到 | 調大 `min_area_with_history` |
| 球速快，短暫 miss 後抓不回來 | 放大 `max_x_gap` 的設定 |
| 沒有 miss 時仍然常常跳到背景球 | 縮小 `x_to_pred` / `y_to_pred` 限制 |
| 球方向變化大但被擋掉 | 放寬方向反轉條件 |
| y 方向跳動較大導致抓不到 | 放寬 `y_to_last > 100` 的限制 |
| reset 後又選回同一個背景球 | 參考 `should_reset_track()` 裡的 ignore stale 設定 |

#### 設計重點

`select_best_candidate()` 的核心想法是：候選點不只要像球，也要符合前後軌跡。

它主要用來解決：

```text
背景球干擾
球速快造成 frame 間距大
短暫 miss 後重新抓球
heatmap 有多個亮點
球點突然跳到錯的位置
```

整體來說，這個 function 是讓 TrackNetV3 更適合桌球影片的關鍵修改之一。

---

### `should_reset_track()`：判斷是否重新追蹤

`should_reset_track()` 是用來判斷目前的追蹤狀態是否還可信。在桌球影片中，TrackNet 有時候會遇到這些情況：

```text
球飛出畫面
球被人或球拍遮住
球短暫消失
背景白點被誤認成球
球停在某個位置幾乎不動
前面選錯球後，後面一路追錯
```

如果程式繼續相信目前的 history，後面的 `select_best_candidate()` 可能會一直根據錯誤的上一點去選球，導致整段軌跡都偏掉。所以 `should_reset_track()` 的目的就是：當目前軌跡看起來已經不可信時，清掉追蹤狀態，重新開始找球。

#### Function 參數

```python
def should_reset_track(
    history,
    frame_w,
    frame_h,
    border_margin=40,
    stale_frames=6,
    stale_avg_step_thresh=6.5,
    stale_y_span_thresh=12.0,
    stale_x_span_thresh=35.0,
    debug=False,
):
```

目前 `predict.py` 實際呼叫時也是使用這組值：

```python
need_reset, reset_reason = should_reset_track(
    track_state["history"],
    frame_w=frame_w,
    frame_h=frame_h,
    border_margin=40,
    stale_frames=6,
    stale_avg_step_thresh=6.5,
    stale_y_span_thresh=12.0,
    stale_x_span_thresh=35.0,
)
```

#### 參數說明

| 參數 | 目前值 | 說明 |
|---|---:|---|
| `history` | - | 前面 frame 的追蹤紀錄，格式為 `(x, y, visibility)` |
| `frame_w` | - | 原影片寬度，用來判斷球是否靠近左右邊界 |
| `frame_h` | - | 原影片高度，用來判斷球是否靠近上下邊界 |
| `border_margin` | `40` | 距離畫面邊界多少 pixel 內視為靠近邊界 |
| `stale_frames` | `6` | 檢查最近幾個有效球點是否幾乎不動 |
| `stale_avg_step_thresh` | `6.5` | 最近幾個有效球點的平均移動距離門檻 |
| `stale_y_span_thresh` | `12.0` | 最近幾個有效球點的 y 方向最大變化門檻 |
| `stale_x_span_thresh` | `35.0` | 最近幾個有效球點的 x 方向最大變化門檻 |
| `debug` | `False` | 是否輸出 reset 原因 |

#### track_state 狀態

`predict.py` 會用 `track_state` 紀錄目前追蹤狀態：

```python
track_state = {
    "history": [],
    "miss_count": 0,
    "ignore_stale_until": -1,
    "ignore_stale_pos": None,
}
```

| 狀態 | 說明 |
|---|---|
| `history` | 最近的球點紀錄 |
| `miss_count` | 目前連續 miss 的 frame 數 |
| `ignore_stale_until` | stale reset 後，在指定 frame 前暫時忽略舊位置附近的候選點 |
| `ignore_stale_pos` | stale reset 發生時最後一個有效球點位置 |

另外 `predict.py` 目前設定：`HISTORY_SIZE = 8`，所以 history 最多保留最近 8 筆狀態。

#### 回傳值

```python
return need_reset, reset_reason
```

| 回傳值 | 說明 |
|---|---|
| `need_reset` | `True` 代表需要 reset，`False` 代表繼續使用目前 history |
| `reset_reason` | reset 原因，目前可能是 `"border_out"`、`"stale_ball"` 或 `None` |

目前 reset 原因主要有兩種：

| reset reason | 說明 |
|---|---|
| `"border_out"` | 球靠近邊界，而且移動方向是往畫面外 |
| `"stale_ball"` | 最近幾個球點幾乎不動，可能追到背景球或停留的錯誤點 |

#### 基本 reset 流程

```text
讀取 history
↓
只保留 visibility = 1 的有效球點
↓
如果有效球點少於 2 個，不 reset
↓
檢查最後兩個有效球點的移動方向
↓
如果球靠近邊界，而且正在往畫面外移動，reset
↓
如果有效球點數量足夠，檢查最近 stale_frames 個點
↓
計算最近幾個點的平均移動距離
↓
計算最近幾個點的 x 方向變化範圍
↓
計算最近幾個點的 y 方向變化範圍
↓
如果球幾乎停在同一區域，reset
↓
其他情況不 reset
```

#### 1. 邊界 reset：`border_out`

第一種 reset 是判斷球是不是已經接近畫面邊界，而且正在往畫面外移動。

程式會先取最後兩個有效球點：

```python
x1, y1 = valid_history[-2]
x2, y2 = valid_history[-1]
vx, vy = x2 - x1, y2 - y1
```

接著判斷最後一個點是否靠近邊界：

```python
near_border = (
    x2 < border_margin or
    x2 > (frame_w - 1 - border_margin) or
    y2 < border_margin or
    y2 > (frame_h - 1 - border_margin)
)
```

目前 `border_margin=40`，代表球距離畫面上下左右邊界 40 pixel 內，都會被視為靠近邊界。

但是「靠近邊界」還不一定 reset，還要同時判斷移動方向是不是往畫面外：

```python
moving_outward = (
    (x2 < border_margin and vx < 0) or
    (x2 > (frame_w - 1 - border_margin) and vx > 0) or
    (y2 < border_margin and vy < 0) or
    (y2 > (frame_h - 1 - border_margin) and vy > 0)
)
```

也就是：

```text
靠近左邊界，且還在往左移動
靠近右邊界，且還在往右移動
靠近上邊界，且還在往上移動
靠近下邊界，且還在往下移動
```

如果同時符合 `near_border` 和 `moving_outward`，就 reset：

```python
if near_border and moving_outward:
    return True, "border_out"
```

這樣可以避免球已經飛出畫面後，程式還繼續沿用舊的 history 去追錯點。


#### 2. 停滯 reset：`stale_ball`

第二種 reset 是判斷最近幾個球點是不是幾乎沒有移動。

如果有效球點數量大於等於 `stale_frames`，就取最近幾個點來檢查：

```python
recent = valid_history[-stale_frames:]
```

目前：`stale_frames = 6`，也就是檢查最近 6 個有效球點。

接著計算三個值：

```text
avg_step：最近幾個點之間的平均移動距離
x_span：最近幾個點的 x 最大變化範圍
y_span：最近幾個點的 y 最大變化範圍
```

如果三個條件都成立，就判斷為 `stale_ball`：

```python
if (
    avg_step <= stale_avg_step_thresh and
    y_span <= stale_y_span_thresh and
    x_span <= stale_x_span_thresh
):
    return True, "stale_ball"
```

目前門檻是：

| 條件 | 目前值 | 意義 |
|---|---:|---|
| `avg_step <= stale_avg_step_thresh` | `6.5` | 最近幾個點平均每次移動小於等於 6.5 pixel |
| `y_span <= stale_y_span_thresh` | `12.0` | 最近幾個點 y 方向總變化小於等於 12 pixel |
| `x_span <= stale_x_span_thresh` | `35.0` | 最近幾個點 x 方向總變化小於等於 35 pixel |

這個設計主要是為了處理：

```text
背景球
球桌上的固定白點
錯誤殘影
模型長時間停在某個錯誤位置
```

真正的高速桌球通常不會連續好幾個有效點都幾乎停在同一個小範圍內，所以這種情況通常代表追蹤已經不可靠。

#### 3. stale reset 後暫時忽略舊位置

這是 `predict.py` 裡額外做的處理，不是在 `should_reset_track()` function 裡面。

如果 reset 原因是 `"stale_ball"`，程式會記住最後一個有效位置，並在接下來一段 frame 內忽略這個位置附近的候選點：

```python
if reset_reason == "stale_ball" and last_valid is not None:
    track_state["ignore_stale_until"] = int(f_i) + 80
    track_state["ignore_stale_pos"] = last_valid
```

目前設定：

| 參數 / 條件 | 目前值 | 說明 |
|---|---:|---|
| `ignore_stale_until` | `目前 frame + 80` | stale reset 後 80 frame 內啟用忽略 |
| `ignore_stale_pos` | `last_valid` | 要忽略的舊位置中心 |

接著在選 candidate 前，會把靠近 stale 位置的候選點濾掉：

```python
if abs(c["cx"] - sx) <= 80 and abs(c["cy"] - sy) <= 50:
    continue
```

目前忽略範圍是：

| 方向 | 範圍 |
|---|---:|
| x 方向 | `±80 px` |
| y 方向 | `±50 px` |

這個設計是為了避免剛 reset 完，又馬上選回同一個停在畫面上的背景球或錯誤點。

只要後面成功選到新的球點，就會停止 ignore stale：

```python
track_state["ignore_stale_until"] = -1
track_state["ignore_stale_pos"] = None
```

#### reset 後對選點的影響

如果 `should_reset_track()` 回傳需要 reset，`predict.py` 會把這一輪傳進 `select_best_candidate()` 的 history 清空：

```python
select_history = [] if need_reset else track_state["history"]
select_miss_count = 0 if need_reset else track_state["miss_count"]
```

因此 reset 後會變成：

```text
不再用舊 history 預測方向
不再用舊 miss_count 放寬距離
改成重新從目前 frame 的 candidates 中找球
```

如果 reset 後這一 frame 沒選到新球點，程式會清空 history 並把 `miss_count` 歸零：

```python
track_state["history"] = []
track_state["miss_count"] = 0
```

如果 reset 後有選到新球點，就會用這個點開始新的 history：

```python
track_state["history"] = [(cx_pred, cy_pred, 1)]
track_state["miss_count"] = 0
```

#### 參數調整建議

| 問題 | 建議調整 |
|---|---|
| 球靠近邊界時太早 reset | 調小 `border_margin` |
| 球已經出畫面但還沒有 reset | 調大 `border_margin` |
| 球速較慢時被誤判成 stale | 降低 stale 判斷敏感度，例如降低 `stale_avg_step_thresh` 或增加 `stale_frames` |
| 背景球停在畫面中卻沒有 reset | 提高 stale 判斷敏感度，例如提高 `stale_avg_step_thresh` 或減少 `stale_frames` |
| reset 太頻繁，導致軌跡容易斷 | 放寬 stale 條件 |
| reset 太晚，錯誤軌跡延續太久 | 收緊 stale 條件 |
| stale reset 後又選回同一個錯誤點 | 加大 ignore stale 範圍，或延長 `ignore_stale_until` |
| stale reset 後正確球剛好在舊位置附近，導致選不到 | 縮小 ignore stale 範圍，或縮短 `ignore_stale_until` |


#### 設計重點

`should_reset_track()` 的重點是不要讓錯誤的 history 一直影響後面的選球。

在目前流程中：

```text
history 會影響 select_best_candidate()
select_best_candidate() 會影響每一 frame 選到哪一顆球
選到的球點會影響後面的 generate_inpaint_mask()
generate_inpaint_mask() 又會影響 InpaintNet 補軌跡
```

所以如果 history 已經不可信，就要 reset。

reset 後的概念是：

```text
不要再用舊的軌跡方向去限制候選點，
讓程式重新從目前 frame 的 candidates 裡找最可能的球。
```

---

## train 相關

目前 training 流程基本沿用原版 TrackNetV3。

原版流程包含：

1. prepare dataset
2. preprocess
3. train TrackNet
4. generate mask data
5. train InpaintNet
6. evaluate

如果只是使用目前這份專案做桌球影片預測，通常不需要重新訓練。

如果要重新訓練，請參考原版 TrackNetV3

## speed_analysis

速度、落點、stroke 分析之後另外寫在：
[Speed Analysis README](./speed_analysis/README.md)
