"""Run TrackNetV3 inference on one video or a folder of videos.

Examples:
    CUDA_VISIBLE_DEVICES=2 python predict.py --video_file 048/C0045.mp4 --tracknet_file exp/TrackNet_best.pt --inpaintnet_file exp/InpaintNet_best.pt --save_dir 048 --eval_mode weight --output_video --large_video
    CUDA_VISIBLE_DEVICES=2 python predict.py --video_dir /home/code-server/NO3 --tracknet_file exp/TrackNet_best.pt --inpaintnet_file exp/InpaintNet_best.pt --save_dir /home/code-server/NO3/pred_result --eval_mode weight --output_video --large_video
"""

import os
import argparse
import cv2
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from test import get_ensemble_weight, generate_inpaint_mask, predict_location_candidates, select_best_candidate, should_reset_track
from dataset import Shuttlecock_Trajectory_Dataset, Video_IterableDataset
from utils.general import *

import queue, threading
from functools import wraps
import time
from contextlib import contextmanager
from collections import defaultdict

import queue, threading
import subprocess
import numpy as np


class PrefetchLoader:
    """用背景 thread 預先把 DataLoader 的 batch 抓出來塞進 queue"""
    def __init__(self, loader, max_prefetch=4):
        self.loader = loader
        self.q = queue.Queue(maxsize=max_prefetch)
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self):
        try:
            for batch in self.loader:
                self.q.put(batch)
        finally:
            self.q.put(None)  # sentinel

    def __iter__(self):
        while True:
            batch = self.q.get()
            if batch is None:
                return
            yield batch


# ─────────────────────────────────────────────────────────────────────────────
# 計時工具
# ─────────────────────────────────────────────────────────────────────────────

class StageTimer:
    """累計各階段耗時，最後印出報表"""
    def __init__(self):
        self.stats = defaultdict(lambda: {"time": 0.0, "count": 0})
        self._wall_start = time.perf_counter()  # 用來計算 unaccounted

    @contextmanager
    def track(self, name, verbose=False):
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.stats[name]["time"] += elapsed
            self.stats[name]["count"] += 1

    def add(self, name, elapsed):
        """直接累加時間（給 timed_loader 用）"""
        self.stats[name]["time"] += elapsed
        self.stats[name]["count"] += 1

    def report(self, title="Timing Report", wall_time=None):
        print("=" * 90)
        print(title)
        print("-" * 90)
        total = sum(v["time"] for v in self.stats.values())
        # 基準：若有給 wall_time 用 wall_time，否則用 total
        base = wall_time if wall_time else total

        for name, v in sorted(self.stats.items(), key=lambda x: -x[1]["time"]):
            pct = v["time"] / base * 100 if base > 0 else 0
            avg = v["time"] / max(v["count"], 1)
            print(f"  {name:<45s}  {v['time']:8.3f}s  "
                  f"x{v['count']:4d}  avg={avg*1000:8.1f}ms  {pct:5.1f}%")
        print("-" * 90)
        print(f"  {'[tracked total]':<45s}  {total:8.3f}s")
        if wall_time:
            unaccounted = wall_time - total
            unacc_pct = unaccounted / wall_time * 100 if wall_time > 0 else 0
            print(f"  {'[wall time]':<45s}  {wall_time:8.3f}s")
            print(f"  {'[UNACCOUNTED]':<45s}  {unaccounted:8.3f}s  "
                  f"{'':19s}{unacc_pct:5.1f}%  ← 沒被 track 的時間")
        print("=" * 90)


def time_stamp(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        print(f"\n[START] {func.__name__}()")
        result = func(*args, **kwargs)
        print(f"[DONE ] {func.__name__}() took {time.perf_counter()-start:.3f}s")
        return result
    return wrapper


def timed_loader(loader, timer, name):
    """把 DataLoader 迭代取 batch 的等待時間也計入 timer"""
    it = iter(loader)
    while True:
        t0 = time.perf_counter()
        try:
            batch = next(it)
        except StopIteration:
            return
        timer.add(name, time.perf_counter() - t0)
        yield batch


# ─────────────────────────────────────────────────────────────────────────────
# 其他工具函式
# ─────────────────────────────────────────────────────────────────────────────

def collect_video_files(video_dir):
    video_files = []
    for root, _, files in os.walk(video_dir):
        for fname in files:
            if fname.endswith('.mp4') or fname.endswith('.MP4'):
                video_files.append(os.path.join(root, fname))
    video_files.sort()
    return video_files


@contextmanager
def _nullctx():
    yield


# ─────────────────────────────────────────────────────────────────────────────
# predict()   （內容與原本相同，僅保留）
# ─────────────────────────────────────────────────────────────────────────────

def predict(indices, y_pred=None, c_pred=None, img_scaler=(1, 1), track_state=None, timer=None):
    if track_state is None:
        track_state = {
            "history": [],
            "miss_count": 0,
            "ignore_stale_until": -1,
            "ignore_stale_pos": None,
        }

    MAX_CANDIDATES = 3
    HISTORY_SIZE = 8
    pred_dict = {'Frame': [], 'X': [], 'Y': [], 'Visibility': []}
    batch_size, seq_len = indices.shape[0], indices.shape[1]

    with (timer.track("post/to_numpy") if timer else _nullctx()):
        indices = indices.detach().cpu().numpy() if torch.is_tensor(indices) else indices.numpy()
        if y_pred is not None:
            y_pred = y_pred > 0.5
            y_pred = y_pred.detach().cpu().numpy() if torch.is_tensor(y_pred) else y_pred
            y_pred = to_img_format(y_pred)
        if c_pred is not None:
            c_pred = c_pred.detach().cpu().numpy() if torch.is_tensor(c_pred) else c_pred

    prev_f_i = -1
    for n in range(batch_size):
        for f in range(seq_len):
            f_i = indices[n][f][1]
            if f_i != prev_f_i:
                if c_pred is not None:
                    c_p = c_pred[n][f]
                    cx_pred = int(c_p[0] * WIDTH * img_scaler[0])
                    cy_pred = int(c_p[1] * HEIGHT * img_scaler[1])

                elif y_pred is not None:
                    with (timer.track("post/heatmap_candidates") if timer else _nullctx()):
                        y_p = y_pred[n][f]
                        heatmap = to_img(y_p)
                        candidates = predict_location_candidates(heatmap, max_candidates=MAX_CANDIDATES)

                    with (timer.track("post/scale_candidates") if timer else _nullctx()):
                        scaled_candidates = []
                        for c in candidates:
                            scaled_candidates.append({
                                "x": int(c["x"] * img_scaler[0]),
                                "y": int(c["y"] * img_scaler[1]),
                                "w": int(c["w"] * img_scaler[0]),
                                "h": int(c["h"] * img_scaler[1]),
                                "cx": c["cx"] * img_scaler[0],
                                "cy": c["cy"] * img_scaler[1],
                                "area": c["area"],
                            })

                    frame_w = int(WIDTH * img_scaler[0])
                    frame_h = int(HEIGHT * img_scaler[1])

                    with (timer.track("post/should_reset") if timer else _nullctx()):
                        need_reset, reset_reason = should_reset_track(
                            track_state["history"],
                            frame_w=frame_w, frame_h=frame_h,
                            border_margin=40, stale_frames=6,
                            stale_avg_step_thresh=6.5,
                            stale_y_span_thresh=12.0,
                            stale_x_span_thresh=35.0,
                        )

                    if need_reset:
                        valid_history = [(x, y) for (x, y, vis) in track_state["history"] if vis == 1]
                        last_valid = valid_history[-1] if valid_history else None
                        print(f"[reset] frame={f_i}, reason={reset_reason}, "
                              f"miss_count={track_state['miss_count']}, "
                              f"history_len={len(track_state['history'])}, "
                              f"last_valid={last_valid}")
                        if reset_reason == "stale_ball" and last_valid is not None:
                            track_state["ignore_stale_until"] = int(f_i) + 80
                            track_state["ignore_stale_pos"] = last_valid

                    select_history    = [] if need_reset else track_state["history"]
                    select_miss_count = 0  if need_reset else track_state["miss_count"]

                    with (timer.track("post/stale_filter") if timer else _nullctx()):
                        if (track_state.get("ignore_stale_pos") is not None and
                                int(f_i) <= track_state.get("ignore_stale_until", -1)):
                            sx, sy = track_state["ignore_stale_pos"]
                            filtered = [c for c in scaled_candidates
                                        if not (abs(c["cx"]-sx) <= 80 and abs(c["cy"]-sy) <= 50)]
                            if len(filtered) != len(scaled_candidates):
                                print(f"[ignore_stale] frame={f_i}, ignore_pos=({sx},{sy}), "
                                      f"before={len(scaled_candidates)}, after={len(filtered)}")
                            scaled_candidates = filtered

                    with (timer.track("post/select_best") if timer else _nullctx()):
                        chosen = select_best_candidate(
                            candidates=scaled_candidates,
                            history=select_history,
                            miss_count=select_miss_count
                        )

                    if chosen is None:
                        cx_pred, cy_pred = 0, 0
                        if need_reset:
                            track_state["history"] = []
                            track_state["miss_count"] = 0
                        else:
                            track_state["miss_count"] += 1
                            track_state["history"].append((0, 0, 0))
                    else:
                        cx_pred = int(chosen["cx"])
                        cy_pred = int(chosen["cy"])
                        track_state["ignore_stale_until"] = -1
                        track_state["ignore_stale_pos"] = None
                        if need_reset:
                            track_state["history"] = [(cx_pred, cy_pred, 1)]
                            track_state["miss_count"] = 0
                        else:
                            track_state["history"].append((cx_pred, cy_pred, 1))
                            track_state["miss_count"] = 0

                    if len(track_state["history"]) > HISTORY_SIZE:
                        track_state["history"] = track_state["history"][-HISTORY_SIZE:]
                else:
                    raise ValueError('Invalid input')

                vis_pred = 0 if cx_pred == 0 and cy_pred == 0 else 1
                pred_dict['Frame'].append(int(f_i))
                pred_dict['X'].append(cx_pred)
                pred_dict['Y'].append(cy_pred)
                pred_dict['Visibility'].append(vis_pred)
                prev_f_i = f_i
            else:
                break

    return pred_dict, track_state


# ─────────────────────────────────────────────────────────────────────────────
# main()
# ─────────────────────────────────────────────────────────────────────────────

@time_stamp
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_file',     type=str, default=None)
    parser.add_argument('--video_dir',      type=str, default=None)
    parser.add_argument('--tracknet_file',  type=str)
    parser.add_argument('--inpaintnet_file',type=str, default='')
    parser.add_argument('--batch_size',     type=int, default=16)
    parser.add_argument('--eval_mode',      type=str, default='weight',
                        choices=['nonoverlap', 'average', 'weight'])
    parser.add_argument('--max_sample_num', type=int, default=100)
    parser.add_argument('--video_range',
                        type=lambda s: [int(x) for x in s.split(',')], default=(10,20))
    parser.add_argument('--save_dir',       type=str, default='pred_result')
    parser.add_argument('--large_video',    action='store_true', default=False)
    parser.add_argument('--output_video',   action='store_true', default=False)
    parser.add_argument('--traj_len',       type=int, default=8)
    parser.add_argument('--video_codec', type=str, default='h264_nvenc',
                    choices=['h264_nvenc', 'libx264'],
                    help='Output video codec. h264_nvenc needs NVIDIA GPU.')
    args = parser.parse_args()

    if args.video_file is None and args.video_dir is None:
        raise ValueError('Please provide --video_file or --video_dir')
    if args.video_file is not None and args.video_dir is not None:
        raise ValueError('Please provide only one of --video_file or --video_dir')

    if args.video_file is not None:
        video_list = [args.video_file]
        base_dir = None
    else:
        video_list = collect_video_files(args.video_dir)
        base_dir = args.video_dir
        print(f'Found {len(video_list)} videos in {args.video_dir}')
        if not video_list:
            raise ValueError(f'No mp4 video found in {args.video_dir}')

    num_workers  = args.batch_size if args.batch_size <= 16 else 16
    video_range  = args.video_range if args.video_range else None
    large_video  = args.large_video

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # ═══════════════════════════════════════════════════════════════════════
    # 全域 timer — 記錄模型載入、overall
    # ═══════════════════════════════════════════════════════════════════════
    global_timer = StageTimer()
    global_wall_start = time.perf_counter()

    # ── 載入模型 ────────────────────────────────────────────────────────────
    with global_timer.track("0_setup/load_tracknet"):
        tracknet_ckpt    = torch.load(args.tracknet_file, map_location="cpu", weights_only=False)
        tracknet_seq_len = tracknet_ckpt['param_dict']['seq_len']
        bg_mode          = tracknet_ckpt['param_dict']['bg_mode']
        tracknet         = get_model('TrackNet', tracknet_seq_len, bg_mode).to(device)
        tracknet.load_state_dict(tracknet_ckpt['model'])

    if args.inpaintnet_file:
        with global_timer.track("0_setup/load_inpaintnet"):
            inpaintnet_ckpt    = torch.load(args.inpaintnet_file, map_location="cpu", weights_only=False)
            inpaintnet_seq_len = inpaintnet_ckpt['param_dict']['seq_len']
            inpaintnet         = get_model('InpaintNet').to(device)
            inpaintnet.load_state_dict(inpaintnet_ckpt['model'])
    else:
        inpaintnet = None

    # ── 每支影片 ────────────────────────────────────────────────────────────
    for video_file in video_list:

        # per-video timer + wall clock
        timer = StageTimer()
        video_wall_start = time.perf_counter()

        video_name = os.path.splitext(os.path.basename(video_file))[0]
        if base_dir is None:
            save_subdir = args.save_dir
        else:
            rel_dir = os.path.relpath(os.path.dirname(video_file), base_dir)
            save_subdir = args.save_dir if rel_dir == '.' else os.path.join(args.save_dir, rel_dir)

        out_csv_file   = os.path.join(save_subdir, f'{video_name}_ball.csv')
        out_video_file = os.path.join(save_subdir, f'{video_name}_predict.mp4')
        os.makedirs(save_subdir, exist_ok=True)

        print('=' * 80)
        print('Processing:', video_file)

        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            print(f"Skip unreadable video: {video_file}")
            continue

        fps      = cap.get(cv2.CAP_PROP_FPS)
        w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w_scaler = w / WIDTH
        h_scaler = h / HEIGHT
        img_scaler = (w_scaler, h_scaler)
        cap.release()

        tracknet_pred_dict = {
            'Frame': [], 'X': [], 'Y': [], 'Visibility': [], 'Inpaint_Mask': [],
            'Img_scaler': (w_scaler, h_scaler), 'Img_shape': (w, h)
        }
        track_state = {
            "history": [], "miss_count": 0,
            "ignore_stale_until": -1, "ignore_stale_pos": None,
        }

        # ════════════════════════════════════════════════════════════════════
        # TrackNet 推論
        # ════════════════════════════════════════════════════════════════════
        tracknet.eval()
        seq_len = tracknet_seq_len

        # ── 建立 Dataset（拆成 dataset + dataloader 兩段）─────────────────
        if args.eval_mode == 'nonoverlap':
            if large_video:
                with timer.track("1_tracknet/dataset_init.dataset_ctor"):
                    dataset = Video_IterableDataset(
                        video_file, seq_len=seq_len, sliding_step=seq_len,
                        bg_mode=bg_mode, max_sample_num=args.max_sample_num,
                        video_range=video_range)
                with timer.track("1_tracknet/dataset_init.dataloader_ctor"):
                    data_loader = DataLoader(dataset, batch_size=args.batch_size,
                                            shuffle=False, drop_last=False,
                                            num_workers=1, prefetch_factor=4)
                print(f'Video length: {dataset.video_len}')
            else:
                with timer.track("1_tracknet/dataset_init.generate_frames"):
                    frame_list = generate_frames(video_file)
                with timer.track("1_tracknet/dataset_init.dataset_ctor"):
                    dataset = Shuttlecock_Trajectory_Dataset(
                        seq_len=seq_len, sliding_step=seq_len,
                        data_mode='heatmap', bg_mode=bg_mode,
                        frame_arr=np.array(frame_list)[:, :, :, ::-1], padding=True)
                with timer.track("1_tracknet/dataset_init.dataloader_ctor"):
                    data_loader = DataLoader(dataset, batch_size=args.batch_size,
                                            shuffle=False, num_workers=num_workers, drop_last=False)
        else:
            if large_video:
                with timer.track("1_tracknet/dataset_init.dataset_ctor"):
                    dataset = Video_IterableDataset(
                        video_file, seq_len=seq_len, sliding_step=1,
                        bg_mode=bg_mode, max_sample_num=args.max_sample_num,
                        video_range=video_range)
                with timer.track("1_tracknet/dataset_init.dataloader_ctor"):
                    data_loader = DataLoader(dataset, batch_size=args.batch_size,
                                            shuffle=False, drop_last=False,
                                            num_workers=1, prefetch_factor=4)
                video_len = dataset.video_len
                print(f'Video length: {video_len}')
            else:
                with timer.track("1_tracknet/dataset_init.generate_frames"):
                    frame_list = generate_frames(video_file)
                with timer.track("1_tracknet/dataset_init.dataset_ctor"):
                    dataset = Shuttlecock_Trajectory_Dataset(
                        seq_len=seq_len, sliding_step=1,
                        data_mode='heatmap', bg_mode=bg_mode,
                        frame_arr=np.array(frame_list)[:, :, :, ::-1])
                with timer.track("1_tracknet/dataset_init.dataloader_ctor"):
                    data_loader = DataLoader(dataset, batch_size=args.batch_size,
                                            shuffle=False, num_workers=num_workers, drop_last=False)
                video_len = len(frame_list)

        # ── 推論迴圈 ─────────────────────────────────────────────────────
        if args.eval_mode == 'nonoverlap':
            for step, (i, x) in enumerate(tqdm(
                timed_loader(data_loader, timer, "1_tracknet/dataloader_wait"))):

                with timer.track("1_tracknet/data_to_gpu"):
                    x = x.float().to(device)

                with timer.track("1_tracknet/gpu_inference"):
                    with torch.no_grad():
                        y_pred = tracknet(x).detach().cpu()

                with timer.track("1_tracknet/post_predict"):
                    tmp_pred, track_state = predict(
                        i, y_pred=y_pred, img_scaler=img_scaler,
                        track_state=track_state, timer=timer)
                    for key in tmp_pred:
                        tracknet_pred_dict[key].extend(tmp_pred[key])

        else:
            # weight / average — temporal ensemble
            num_sample, sample_count = video_len - seq_len + 1, 0
            buffer_size = seq_len - 1
            batch_i     = torch.arange(seq_len)
            frame_i     = torch.arange(seq_len - 1, -1, -1)
            y_pred_buffer = torch.zeros((buffer_size, seq_len, HEIGHT, WIDTH), dtype=torch.float32)
            weight        = get_ensemble_weight(seq_len, args.eval_mode)

            prefetch_loader = PrefetchLoader(data_loader, max_prefetch=4)
            for step, (i, x) in enumerate(tqdm(
                timed_loader(prefetch_loader, timer, "1_tracknet/dataloader_wait"))):

                with timer.track("1_tracknet/data_to_gpu"):
                    x = x.float().to(device)

                b_size, seq_len = i.shape[0], i.shape[1]

                with timer.track("1_tracknet/gpu_inference"):
                    with torch.no_grad():
                        y_pred = tracknet(x).detach().cpu()

                with timer.track("1_tracknet/ensemble_buffer"):
                    y_pred_buffer = torch.cat((y_pred_buffer, y_pred), dim=0)
                    ensemble_i      = torch.empty((0, 1, 2), dtype=torch.float32)
                    ensemble_y_pred = torch.empty((0, 1, HEIGHT, WIDTH), dtype=torch.float32)

                    for b in range(b_size):
                        if sample_count < buffer_size:
                            y_pred_e = y_pred_buffer[batch_i+b, frame_i].sum(0) / (sample_count+1)
                        else:
                            y_pred_e = (y_pred_buffer[batch_i+b, frame_i] * weight[:, None, None]).sum(0)

                        ensemble_i      = torch.cat((ensemble_i, i[b][0].reshape(1,1,2)), dim=0)
                        ensemble_y_pred = torch.cat((ensemble_y_pred, y_pred_e.reshape(1,1,HEIGHT,WIDTH)), dim=0)
                        sample_count += 1

                        if sample_count == num_sample:
                            y_zero_pad = torch.zeros((buffer_size, seq_len, HEIGHT, WIDTH), dtype=torch.float32)
                            y_pred_buffer = torch.cat((y_pred_buffer, y_zero_pad), dim=0)
                            for f in range(1, seq_len):
                                y_pred_e = y_pred_buffer[batch_i+b+f, frame_i].sum(0) / (seq_len-f)
                                ensemble_i      = torch.cat((ensemble_i, i[-1][f].reshape(1,1,2)), dim=0)
                                ensemble_y_pred = torch.cat((ensemble_y_pred, y_pred_e.reshape(1,1,HEIGHT,WIDTH)), dim=0)

                with timer.track("1_tracknet/post_predict"):
                    tmp_pred, track_state = predict(
                        ensemble_i, y_pred=ensemble_y_pred, img_scaler=img_scaler,
                        track_state=track_state, timer=timer)
                    for key in tmp_pred:
                        tracknet_pred_dict[key].extend(tmp_pred[key])

                y_pred_buffer = y_pred_buffer[-buffer_size:]

        # ════════════════════════════════════════════════════════════════════
        # InpaintNet 推論
        # ════════════════════════════════════════════════════════════════════
        if inpaintnet is not None:
            inpaintnet.eval()
            seq_len = inpaintnet_seq_len

            with timer.track("2_inpaint/gen_mask"):
                tracknet_pred_dict['Inpaint_Mask'] = generate_inpaint_mask(
                    tracknet_pred_dict, frame_w=w, frame_h=h,
                    max_gap=14, border_margin_x=160, max_angle_diff=100.0,
                    min_valid_run=1, angle_check_min_gap=14)

            inpaint_pred_dict = {'Frame': [], 'X': [], 'Y': [], 'Visibility': []}

            with timer.track("2_inpaint/dataset_init"):
                if args.eval_mode == 'nonoverlap':
                    dataset     = Shuttlecock_Trajectory_Dataset(
                        seq_len=seq_len, sliding_step=seq_len,
                        data_mode='coordinate', pred_dict=tracknet_pred_dict, padding=True)
                    data_loader = DataLoader(dataset, batch_size=args.batch_size,
                                            shuffle=False, num_workers=num_workers, drop_last=False)
                else:
                    dataset     = Shuttlecock_Trajectory_Dataset(
                        seq_len=seq_len, sliding_step=1,
                        data_mode='coordinate', pred_dict=tracknet_pred_dict)
                    data_loader = DataLoader(dataset, batch_size=args.batch_size,
                                            shuffle=False, num_workers=num_workers, drop_last=False)

            if args.eval_mode == 'nonoverlap':
                for step, (i, coor_pred, inpaint_mask) in enumerate(tqdm(
                    timed_loader(data_loader, timer, "2_inpaint/dataloader_wait"))):

                    with timer.track("2_inpaint/data_to_gpu"):
                        coor_pred, inpaint_mask = coor_pred.float(), inpaint_mask.float()

                    with timer.track("2_inpaint/gpu_inference"):
                        with torch.no_grad():
                            coor_inpaint = inpaintnet(coor_pred.to(device), inpaint_mask.to(device)).detach().cpu()
                            coor_inpaint = coor_inpaint * inpaint_mask + coor_pred * (1 - inpaint_mask)

                    with timer.track("2_inpaint/post_predict"):
                        th_mask = ((coor_inpaint[:,:,0] < COOR_TH) & (coor_inpaint[:,:,1] < COOR_TH))
                        coor_inpaint[th_mask] = 0.
                        tmp_pred, _ = predict(i, c_pred=coor_inpaint, img_scaler=img_scaler, timer=timer)
                        for key in tmp_pred:
                            inpaint_pred_dict[key].extend(tmp_pred[key])
            else:
                weight       = get_ensemble_weight(seq_len, args.eval_mode)
                num_sample   = len(dataset)
                sample_count = 0
                buffer_size  = seq_len - 1
                batch_i      = torch.arange(seq_len)
                frame_i      = torch.arange(seq_len - 1, -1, -1)
                coor_inpaint_buffer = torch.zeros((buffer_size, seq_len, 2), dtype=torch.float32)

                for step, (i, coor_pred, inpaint_mask) in enumerate(tqdm(
                    timed_loader(data_loader, timer, "2_inpaint/dataloader_wait"))):

                    with timer.track("2_inpaint/data_to_gpu"):
                        coor_pred, inpaint_mask = coor_pred.float(), inpaint_mask.float()
                        b_size = i.shape[0]

                    with timer.track("2_inpaint/gpu_inference"):
                        with torch.no_grad():
                            coor_inpaint = inpaintnet(coor_pred.to(device), inpaint_mask.to(device)).detach().cpu()
                            coor_inpaint = coor_inpaint * inpaint_mask + coor_pred * (1 - inpaint_mask)

                    with timer.track("2_inpaint/ensemble_buffer"):
                        th_mask = ((coor_inpaint[:,:,0] < COOR_TH) & (coor_inpaint[:,:,1] < COOR_TH))
                        coor_inpaint[th_mask] = 0.
                        coor_inpaint_buffer = torch.cat((coor_inpaint_buffer, coor_inpaint), dim=0)
                        ensemble_i            = torch.empty((0, 1, 2), dtype=torch.float32)
                        ensemble_coor_inpaint = torch.empty((0, 1, 2), dtype=torch.float32)

                        for b in range(b_size):
                            if sample_count < buffer_size:
                                ci = coor_inpaint_buffer[batch_i+b, frame_i].sum(0) / (sample_count+1)
                            else:
                                ci = (coor_inpaint_buffer[batch_i+b, frame_i] * weight[:, None]).sum(0)
                            ensemble_i            = torch.cat((ensemble_i, i[b][0].view(1,1,2)), dim=0)
                            ensemble_coor_inpaint = torch.cat((ensemble_coor_inpaint, ci.view(1,1,2)), dim=0)
                            sample_count += 1

                            if sample_count == num_sample:
                                coor_zero_pad = torch.zeros((buffer_size, seq_len, 2), dtype=torch.float32)
                                coor_inpaint_buffer = torch.cat((coor_inpaint_buffer, coor_zero_pad), dim=0)
                                for f in range(1, seq_len):
                                    ci = coor_inpaint_buffer[batch_i+b+f, frame_i].sum(0) / (seq_len-f)
                                    ensemble_i            = torch.cat((ensemble_i, i[-1][f].view(1,1,2)), dim=0)
                                    ensemble_coor_inpaint = torch.cat((ensemble_coor_inpaint, ci.view(1,1,2)), dim=0)

                    with timer.track("2_inpaint/post_predict"):
                        th_mask = ((ensemble_coor_inpaint[:,:,0] < COOR_TH) & (ensemble_coor_inpaint[:,:,1] < COOR_TH))
                        ensemble_coor_inpaint[th_mask] = 0.
                        tmp_pred, _ = predict(ensemble_i, c_pred=ensemble_coor_inpaint,
                                              img_scaler=img_scaler, timer=timer)
                        for key in tmp_pred:
                            inpaint_pred_dict[key].extend(tmp_pred[key])

                    coor_inpaint_buffer = coor_inpaint_buffer[-buffer_size:]

        # ════════════════════════════════════════════════════════════════════
        # 儲存結果
        # ════════════════════════════════════════════════════════════════════
        pred_dict = inpaint_pred_dict if inpaintnet is not None else tracknet_pred_dict

        if 'Inpaint_Mask' not in pred_dict:
            pred_dict['Inpaint_Mask'] = (tracknet_pred_dict.get('Inpaint_Mask')
                                         or [0] * len(pred_dict['Frame']))

        with timer.track("3_output/write_csv"):
            write_pred_csv(pred_dict, save_file=out_csv_file)

        if args.output_video:
            with timer.track("3_output/write_video"):
                write_pred_video(video_file, pred_dict, save_file=out_video_file,
                                 traj_len=args.traj_len, codec=args.video_codec)

        print(f'Done: {video_file}')

        # ── 印出這支影片的 timing report（帶 wall time 顯示 UNACCOUNTED）──
        video_wall = time.perf_counter() - video_wall_start
        timer.report(title=f'Timing Report — {video_name}', wall_time=video_wall)

        # 累積到全域（方便整批跑完看 overall）
        for name, v in timer.stats.items():
            global_timer.stats[name]["time"] += v["time"]
            global_timer.stats[name]["count"] += v["count"]

    # ── 整批完成後的 Overall Report ─────────────────────────────────────────
    overall_wall = time.perf_counter() - global_wall_start
    global_timer.report(title='Overall Timing Report (all videos)', wall_time=overall_wall)
    print('All done.')


if __name__ == '__main__':
    main()
