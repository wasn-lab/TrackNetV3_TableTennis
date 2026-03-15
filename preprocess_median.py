# 原本的preprocess是從video切出frame，現在已經有frame，所以直接算median.npz
'''
想像你在看一段桌球影片：
桌子、地板、牆壁、球網 → 幾乎不動
桌球 → 又小、又快、只出現幾 frame

如果你把很多 frame 疊起來取 median：桌球在每個 pixel 出現的次數非常少，桌子、背景在同一個 pixel 出現很多次
- median 會把球「洗掉」
- 剩下的就是一張「純背景圖」

median.npz = 一張「沒有球的背景圖」
'''

import os
import glob
import numpy as np
import cv2
from tqdm import tqdm

DATA_DIR = "/mnt/storage/opentt/tracknetv3_data/data"

def compute_match_median_from_frames(match_dir, max_frames=1800):
    """
    Compute median image for a match from existing jpg frames.
    Save to: <match_dir>/median.npz with key 'median' (RGB uint8).
    """
    frame_root = os.path.join(match_dir, "frame")
    if not os.path.isdir(frame_root):
        print(f"[skip] no frame dir: {frame_root}")
        return False

    rally_dirs = sorted([d for d in glob.glob(os.path.join(frame_root, "*")) if os.path.isdir(d)])
    if len(rally_dirs) == 0:
        print(f"[skip] no rally dirs under: {frame_root}")
        return False

    # Collect frame file paths (jpg) across rallies
    all_jpg = []
    for rdir in rally_dirs:
        jpgs = sorted(glob.glob(os.path.join(rdir, "*.jpg")))
        if jpgs:
            all_jpg.extend(jpgs)

    if len(all_jpg) == 0:
        print(f"[skip] no jpg found under: {frame_root}")
        return False

    # Sample to max_frames to control memory/time
    if len(all_jpg) > max_frames:
        step = len(all_jpg) // max_frames
        sampled = all_jpg[::step][:max_frames]
    else:
        sampled = all_jpg

    frames = []
    for fp in sampled:
        img = cv2.imread(fp, cv2.IMREAD_COLOR)  # BGR
        if img is None:
            continue
        frames.append(img)

    if len(frames) == 0:
        print(f"[skip] all frames failed to read in: {match_dir}")
        return False

    # median in BGR then convert to RGB
    median_bgr = np.median(np.stack(frames, axis=0), axis=0)
    median_bgr = np.clip(median_bgr, 0, 255).astype(np.uint8)
    median_rgb = median_bgr[..., ::-1]

    out_file = os.path.join(match_dir, "median.npz")
    np.savez(out_file, median=median_rgb)
    print(f"[done] saved {out_file} (frames used: {len(frames)})")
    return True


def run_split(split):
    split_dir = os.path.join(DATA_DIR, split)
    if not os.path.isdir(split_dir):
        print(f"[skip] split not found: {split_dir}")
        return

    match_dirs = sorted([d for d in glob.glob(os.path.join(split_dir, "match*")) if os.path.isdir(d)])
    for mdir in tqdm(match_dirs, desc=f"median {split}"):
        compute_match_median_from_frames(mdir, max_frames=1800)


if __name__ == "__main__":
    for split in ["train", "val"]:   
        run_split(split)

    print("All done.")
