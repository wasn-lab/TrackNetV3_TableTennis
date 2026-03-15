# python split_data.py --val_ratio 0.1
'''
  data
    ├─ train
    |   ├── match1/
    |   │   ├── csv/
    |   │   │   └── 1_01_00_ball.csv
    |   │   │    
    |   │   └──  frame/
    |   │       └── 1_01_00/
    |   │           ├── 0.png
    |   │           ├── 1.png
    |   │           ├── …
    |   │           └── *.png 
    |   ├── match2/
    |   └── match3/
    └── val
        ├── match1/
        ├── match2/
        └── match3/
  
'''

from pathlib import Path
import argparse
import pandas as pd
from PIL import Image


SRC_ROOT = Path("//home/code-server//opentt")
OUT_DATA = Path("/home/code-server/opentt/tracknetv3_data/data")

SOURCES = [
    "game3_55s_1m35s_clip",
    "60FPS_merged",
    "120FPS_merged",
]

RALLY_ID = "1_01_00"  # 只是名字，跟作者範例一致

def mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def symlink(src: Path, dst: Path):
    if dst.exists() or dst.is_symlink():
        if dst.is_dir():
            raise IsADirectoryError(f"Destination is a directory, not file: {dst}")
        dst.unlink()
    dst.symlink_to(src.resolve())

def detect_cols(df: pd.DataFrame):
    cols = {c.lower(): c for c in df.columns}
    def pick(*names):
        for n in names:
            if n in cols:
                return cols[n]
        return None
    # 優先用欄位名；找不到就用前四欄猜
    c_frame = pick("frame", "frame_id") or df.columns[0]
    c_x     = pick("x", "ball_x")       or df.columns[1]
    c_y     = pick("y", "ball_y")       or df.columns[2]
    c_vis   = pick("visibility", "vis", "visible") or df.columns[3]
    return c_frame, c_vis, c_x, c_y

def load_labels(src_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(src_csv)
    c_frame, c_vis, c_x, c_y = detect_cols(df)
    out = df[[c_frame, c_vis, c_x, c_y]].copy()
    out.columns = ["Frame", "Visibility", "X", "Y"]
    out = out.sort_values("Frame").reset_index(drop=True)
    return out

def get_max_frame(images_dir: Path) -> int:
    mx = -1
    for p in images_dir.iterdir():
        if p.suffix.lower() != ".jpg":
            continue
        try:
            idx = int(p.stem)
            mx = max(mx, idx)
        except:
            pass
    if mx < 0:
        raise ValueError(f"No jpg frames in {images_dir}")
    return mx

def stage_segment(split_dir: Path, match_name: str, src_name: str,
                  start_f: int, end_f: int):
    """
    建立：
      split_dir/matchX/frame/1_01_00/0.jpg...
      split_dir/matchX/csv/1_01_00_ball.csv
    並把 Frame 重新從 0 開始
    """
    src_dir = SRC_ROOT / src_name
    images_dir = src_dir / "images"
    labels_csv = src_dir / "labels.csv"

    if not images_dir.is_dir():
        raise FileNotFoundError(images_dir)
    if not labels_csv.is_file():
        raise FileNotFoundError(labels_csv)

    match_dir = split_dir / match_name
    frame_dst = match_dir / "frame" / RALLY_ID
    csv_dst   = match_dir / "csv" / f"{RALLY_ID}_ball.csv"
    mkdir(frame_dst)
    mkdir(csv_dst.parent)

    # 1) CSV：切段 + Frame rebase
    df = load_labels(labels_csv)
    seg = df[(df["Frame"] >= start_f) & (df["Frame"] <= end_f)].copy()
    seg["Frame"] = seg["Frame"] - start_f
    seg = seg.fillna(0)

    # --- convert normalized X/Y (0~1) to pixel if needed ---
    # read one frame to get (w,h)
    any_img = images_dir / f"{start_f}.jpg"
    if not any_img.exists():
        jpgs = sorted(images_dir.glob("*.jpg"))
        if not jpgs:
            raise ValueError(f"No jpg frames in {images_dir}")
        any_img = jpgs[0]
    w, h = Image.open(any_img).size

    # if looks normalized, convert to pixel
    if seg["X"].max() <= 1.5 and seg["Y"].max() <= 1.5:
        seg["X"] = seg["X"] * w
        seg["Y"] = seg["Y"] * h

    seg.to_csv(csv_dst, index=False)
    ''' 檢查座標還原
    python - <<'PY'
    import os
    import pandas as pd
    import cv2
    import glob

    csv_path = "/mnt/storage/opentt/tracknetv3_data/data/val/match2/csv/1_01_00_ball.csv"
    frame_dir = "/mnt/storage/opentt/tracknetv3_data/data/val/match2/frame/1_01_00"
    out_dir = "/mnt/storage/opentt/tracknetv3_data/debug_xy"
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    df = df[df["Visibility"] == 1].head(10)

    # 自動判斷副檔名
    ext = ".jpg"
    if not os.path.exists(os.path.join(frame_dir, "0.jpg")):
        ext = ".png"

    for _, row in df.iterrows():
        f = int(row["Frame"])
        x, y = float(row["X"]), float(row["Y"])
        img_path = os.path.join(frame_dir, f"{f}{ext}")
        img = cv2.imread(img_path)
        if img is None:
            continue
        # 畫點（BGR: 紅點）
        cv2.circle(img, (int(x), int(y)), 6, (0,0,255), -1)
        cv2.imwrite(os.path.join(out_dir, f"debug_{f}.jpg"), img)

    print("Saved to:", out_dir)
    PY

    '''

    # 2) Frames：逐張 symlink，並重新命名成 0.jpg...
    for old_f in range(start_f, end_f + 1):
        src_jpg = images_dir / f"{old_f}.jpg"
        if not src_jpg.exists():
            continue
        new_f = old_f - start_f
        dst_jpg = frame_dst / f"{new_f}.jpg"
        symlink(src_jpg, dst_jpg)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_ratio", type=float, default=0.1,
                    help="each video: take last ratio as val (e.g. 0.1 = 10%)")
    args = ap.parse_args()

    # 檢查資料
    for s in SOURCES:
        base = SRC_ROOT / s
        if not (base / "images").is_dir():
            raise FileNotFoundError(base / "images")
        if not (base / "labels.csv").is_file():
            raise FileNotFoundError(base / "labels.csv")

    train_dir = OUT_DATA / "train"
    val_dir   = OUT_DATA / "val"
    mkdir(train_dir)
    mkdir(val_dir)

    # 三支影片都進 train+val
    for i, src_name in enumerate(SOURCES, start=1):
        images_dir = SRC_ROOT / src_name / "images"
        max_f = get_max_frame(images_dir)
        total = max_f + 1
        split_f = int(total * (1 - args.val_ratio))

        if split_f <= 1 or split_f >= total:
            raise ValueError(f"{src_name} split invalid: total={total}, split_f={split_f}")

        # train: [0, split_f-1]
        stage_segment(train_dir, f"match{i}", src_name, 0, split_f - 1)

        # val: [split_f, max_f]
        stage_segment(val_dir, f"match{i}", src_name, split_f, max_f)

        print(f"{src_name}: total={total}, train=0..{split_f-1}, val={split_f}..{max_f}")

    print("\nDone. TrackNetV3-ready dataset at:")
    print(OUT_DATA)

if __name__ == "__main__":
    main()
