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

def collect_video_files(video_dir):
    video_files = []
    for root, _, files in os.walk(video_dir):
        for fname in files:
            if fname.endswith('.mp4') or fname.endswith('.MP4'):
                video_files.append(os.path.join(root, fname))
    video_files.sort()
    return video_files

def predict(indices, y_pred=None, c_pred=None, img_scaler=(1, 1), track_state=None):
    """ Predict coordinates from heatmap or inpainted coordinates.

        Args:
            indices (torch.Tensor): indices of input sequence with shape (N, L, 2)
            y_pred (torch.Tensor, optional): predicted heatmap sequence with shape (N, L, H, W)
            c_pred (torch.Tensor, optional): predicted inpainted coordinates sequence with shape (N, L, 2)
            img_scaler (Tuple): image scaler (w_scaler, h_scaler)

        Returns:
            pred_dict (Dict): dictionary of predicted coordinates
                Format: {'Frame':[], 'X':[], 'Y':[], 'Visibility':[]}
    """

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
    indices = indices.detach().cpu().numpy() if torch.is_tensor(indices) else indices.numpy()

    # Convert heatmap prediction to image format
    if y_pred is not None:
        y_pred = y_pred > 0.5
        y_pred = y_pred.detach().cpu().numpy() if torch.is_tensor(y_pred) else y_pred
        y_pred = to_img_format(y_pred)  # (N, L, H, W)

    # Convert coordinate prediction to numpy
    if c_pred is not None:
        c_pred = c_pred.detach().cpu().numpy() if torch.is_tensor(c_pred) else c_pred

    prev_f_i = -1
    for n in range(batch_size):
        for f in range(seq_len):
            f_i = indices[n][f][1]
            if f_i != prev_f_i:
                if c_pred is not None:
                    # Predict from coordinates
                    c_p = c_pred[n][f]
                    cx_pred, cy_pred = int(c_p[0] * WIDTH * img_scaler[0]), int(c_p[1] * HEIGHT * img_scaler[1])
                elif y_pred is not None:
                    # Predict from heatmap and choose the best candidate
                    y_p = y_pred[n][f]
                    heatmap = to_img(y_p)

                    candidates = predict_location_candidates(
                        heatmap,
                        max_candidates=MAX_CANDIDATES,
                    )

                    # Scale heatmap candidates back to the original video size
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

                    if need_reset:
                        valid_history = [(x, y) for (x, y, vis) in track_state["history"] if vis == 1]
                        last_valid = valid_history[-1] if len(valid_history) > 0 else None
                        print(
                            f"[reset] frame={f_i}, "
                            f"reason={reset_reason}, "
                            f"miss_count={track_state['miss_count']}, "
                            f"history_len={len(track_state['history'])}, "
                            f"last_valid={last_valid}"
                        )

                        # Ignore the previous stale ball for a short period
                        if reset_reason == "stale_ball" and last_valid is not None:
                            track_state["ignore_stale_until"] = int(f_i) + 80
                            track_state["ignore_stale_pos"] = last_valid

                    # Reset means the previous ball has ended
                    select_history = [] if need_reset else track_state["history"]
                    select_miss_count = 0 if need_reset else track_state["miss_count"]

                    # --------------------------------------------------
                    # Avoid selecting candidates near the stale ball
                    # --------------------------------------------------
                    if (
                        track_state.get("ignore_stale_pos") is not None and
                        int(f_i) <= track_state.get("ignore_stale_until", -1)
                    ):
                        sx, sy = track_state["ignore_stale_pos"]

                        filtered_candidates = []
                        for c in scaled_candidates:
                            if abs(c["cx"] - sx) <= 80 and abs(c["cy"] - sy) <= 50:
                                continue
                            filtered_candidates.append(c)

                        if len(filtered_candidates) != len(scaled_candidates):
                            print(
                                f"[ignore_stale] frame={f_i}, "
                                f"ignore_pos=({sx},{sy}), "
                                f"before={len(scaled_candidates)}, after={len(filtered_candidates)}"
                            )

                        scaled_candidates = filtered_candidates

                    chosen = select_best_candidate(candidates=scaled_candidates, history=select_history, miss_count=select_miss_count)

                    if chosen is None:
                        cx_pred, cy_pred = 0, 0

                        if need_reset:
                            # The old ball ended, but the new ball has not appeared yet
                            track_state["history"] = []
                            track_state["miss_count"] = 0
                        else:
                            # Short miss in the same trajectory
                            track_state["miss_count"] += 1
                            track_state["history"].append((0, 0, 0))
                    else:
                        cx_pred = int(chosen["cx"])
                        cy_pred = int(chosen["cy"])

                        # Once a new ball is found, stop ignoring the stale position
                        track_state["ignore_stale_until"] = -1
                        track_state["ignore_stale_pos"] = None

                        if need_reset:
                            # Start a new history after reset
                            track_state["history"] = [(cx_pred, cy_pred, 1)]
                            track_state["miss_count"] = 0
                        else:
                            # Continue the current trajectory
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_file', type=str, default=None, help='file path of the video')
    parser.add_argument('--video_dir', type=str, default=None, help='directory containing mp4 videos')
    parser.add_argument('--tracknet_file', type=str, help='file path of the TrackNet model checkpoint')
    parser.add_argument('--inpaintnet_file', type=str, default='', help='file path of the InpaintNet model checkpoint')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size for inference')
    parser.add_argument('--eval_mode', type=str, default='weight', choices=['nonoverlap', 'average', 'weight'], help='evaluation mode')
    parser.add_argument('--max_sample_num', type=int, default=1800, help='maximum number of frames to sample for generating median image')
    parser.add_argument('--video_range', type=lambda splits: [int(s) for s in splits.split(',')], default=None, help='range of start second and end second of the video for generating median image')
    parser.add_argument('--save_dir', type=str, default='pred_result', help='directory to save the prediction result')
    parser.add_argument('--large_video', action='store_true', default=False, help='whether to process large video')
    parser.add_argument('--output_video', action='store_true', default=False, help='whether to output video with predicted trajectory')
    parser.add_argument('--traj_len', type=int, default=8, help='length of trajectory to draw on video')
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
        if len(video_list) == 0:
            raise ValueError(f'No mp4 video found in {args.video_dir}')

    num_workers = args.batch_size if args.batch_size <= 16 else 16
    video_range = args.video_range if args.video_range else None
    large_video = args.large_video

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # Load model
    tracknet_ckpt = torch.load(args.tracknet_file, map_location="cpu")
    tracknet_seq_len = tracknet_ckpt['param_dict']['seq_len']
    bg_mode = tracknet_ckpt['param_dict']['bg_mode']
    tracknet = get_model('TrackNet', tracknet_seq_len, bg_mode).to(device)
    tracknet.load_state_dict(tracknet_ckpt['model'])

    if args.inpaintnet_file:
        inpaintnet_ckpt = torch.load(args.inpaintnet_file, map_location="cpu")
        inpaintnet_seq_len = inpaintnet_ckpt['param_dict']['seq_len']
        inpaintnet = get_model('InpaintNet').to(device)
        inpaintnet.load_state_dict(inpaintnet_ckpt['model'])
    else:
        inpaintnet = None

    for video_file in video_list:
        video_name = os.path.splitext(os.path.basename(video_file))[0]

        if base_dir is None:
            save_subdir = args.save_dir
        else:
            rel_dir = os.path.relpath(os.path.dirname(video_file), base_dir)
            save_subdir = args.save_dir if rel_dir == '.' else os.path.join(args.save_dir, rel_dir)

        out_csv_file = os.path.join(save_subdir, f'{video_name}_ball.csv')
        out_video_file = os.path.join(save_subdir, f'{video_name}_predict.mp4')

        os.makedirs(save_subdir, exist_ok=True)

        print('=' * 80)
        print('Processing:', video_file)

        cap = cv2.VideoCapture(video_file)
        print("video_file =", video_file)
        print("cap.isOpened() =", cap.isOpened())

        if not cap.isOpened():
            print(f"Skip unreadable video: {video_file}")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        w, h = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        w_scaler, h_scaler = w / WIDTH, h / HEIGHT
        img_scaler = (w_scaler, h_scaler)

        tracknet_pred_dict = {'Frame': [], 'X': [], 'Y': [], 'Visibility': [], 'Inpaint_Mask': [],
                              'Img_scaler': (w_scaler, h_scaler), 'Img_shape': (w, h)}
        track_state = {
            "history": [],
            "miss_count": 0,
            "ignore_stale_until": -1,
            "ignore_stale_pos": None,
        }

        # Run TrackNet
        tracknet.eval()
        seq_len = tracknet_seq_len
        if args.eval_mode == 'nonoverlap':
            # Use non-overlap sampling
            if large_video:
                dataset = Video_IterableDataset(video_file, seq_len=seq_len, sliding_step=seq_len, bg_mode=bg_mode,
                                                max_sample_num=args.max_sample_num, video_range=video_range)
                data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
                print(f'Video length: {dataset.video_len}')
            else:
                # Load all frames into memory
                frame_list = generate_frames(video_file)
                dataset = Shuttlecock_Trajectory_Dataset(seq_len=seq_len, sliding_step=seq_len, data_mode='heatmap', bg_mode=bg_mode,
                                                     frame_arr=np.array(frame_list)[:, :, :, ::-1], padding=True)
                data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, drop_last=False)

            for step, (i, x) in enumerate(tqdm(data_loader)):
                x = x.float().to(device)
                with torch.no_grad():
                    y_pred = tracknet(x).detach().cpu()

                # Predict
                tmp_pred, track_state = predict(i, y_pred=y_pred, img_scaler=img_scaler, track_state=track_state)
                for key in tmp_pred.keys():
                    tracknet_pred_dict[key].extend(tmp_pred[key])
        else:
            # Use overlap sampling for temporal ensemble
            if large_video:
                dataset = Video_IterableDataset(video_file, seq_len=seq_len, sliding_step=1, bg_mode=bg_mode,
                                                max_sample_num=args.max_sample_num, video_range=video_range)
                data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
                video_len = dataset.video_len
                print(f'Video length: {video_len}')

            else:
                # Load all frames into memory
                frame_list = generate_frames(video_file)
                dataset = Shuttlecock_Trajectory_Dataset(seq_len=seq_len, sliding_step=1, data_mode='heatmap', bg_mode=bg_mode,
                                                     frame_arr=np.array(frame_list)[:, :, :, ::-1])
                data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
                video_len = len(frame_list)

            # Initialize temporal ensemble buffers
            num_sample, sample_count = video_len-seq_len+1, 0
            buffer_size = seq_len - 1
            batch_i = torch.arange(seq_len) # [0, 1, 2, 3, 4, 5, 6, 7]
            frame_i = torch.arange(seq_len-1, -1, -1) # [7, 6, 5, 4, 3, 2, 1, 0]
            y_pred_buffer = torch.zeros((buffer_size, seq_len, HEIGHT, WIDTH), dtype=torch.float32)
            weight = get_ensemble_weight(seq_len, args.eval_mode)
            for step, (i, x) in enumerate(tqdm(data_loader)):
                x = x.float().to(device)
                b_size, seq_len = i.shape[0], i.shape[1]
                with torch.no_grad():
                    y_pred = tracknet(x).detach().cpu()

                y_pred_buffer = torch.cat((y_pred_buffer, y_pred), dim=0)
                ensemble_i = torch.empty((0, 1, 2), dtype=torch.float32)
                ensemble_y_pred = torch.empty((0, 1, HEIGHT, WIDTH), dtype=torch.float32)

                for b in range(b_size):
                    if sample_count < buffer_size:
                        # Buffer is not full yet
                        y_pred = y_pred_buffer[batch_i+b, frame_i].sum(0) / (sample_count+1)
                    else:
                        # Normal weighted ensemble
                        y_pred = (y_pred_buffer[batch_i+b, frame_i] * weight[:, None, None]).sum(0)

                    ensemble_i = torch.cat((ensemble_i, i[b][0].reshape(1, 1, 2)), dim=0)
                    ensemble_y_pred = torch.cat((ensemble_y_pred, y_pred.reshape(1, 1, HEIGHT, WIDTH)), dim=0)
                    sample_count += 1

                    if sample_count == num_sample:
                        # Flush remaining frames in the last batch
                        y_zero_pad = torch.zeros((buffer_size, seq_len, HEIGHT, WIDTH), dtype=torch.float32)
                        y_pred_buffer = torch.cat((y_pred_buffer, y_zero_pad), dim=0)

                        for f in range(1, seq_len):
                            # Flush remaining frames in the last sequence
                            y_pred = y_pred_buffer[batch_i+b+f, frame_i].sum(0) / (seq_len-f)
                            ensemble_i = torch.cat((ensemble_i, i[-1][f].reshape(1, 1, 2)), dim=0)
                            ensemble_y_pred = torch.cat((ensemble_y_pred, y_pred.reshape(1, 1, HEIGHT, WIDTH)), dim=0)

                # Predict
                tmp_pred, track_state = predict(ensemble_i, y_pred=ensemble_y_pred, img_scaler=img_scaler, track_state=track_state)
                for key in tmp_pred.keys():
                    tracknet_pred_dict[key].extend(tmp_pred[key])

                # Keep the last predictions for the next ensemble window
                y_pred_buffer = y_pred_buffer[-buffer_size:]
        # Run TrackNetV3 (TrackNet + InpaintNet)
        if inpaintnet is not None:
            inpaintnet.eval()
            seq_len = inpaintnet_seq_len
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

            inpaint_pred_dict = {'Frame': [], 'X': [], 'Y': [], 'Visibility': []}

            if args.eval_mode == 'nonoverlap':
                # Use non-overlap sampling
                dataset = Shuttlecock_Trajectory_Dataset(seq_len=seq_len, sliding_step=seq_len, data_mode='coordinate', pred_dict=tracknet_pred_dict, padding=True)
                data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, drop_last=False)

                for step, (i, coor_pred, inpaint_mask) in enumerate(tqdm(data_loader)):
                    coor_pred, inpaint_mask = coor_pred.float(), inpaint_mask.float()
                    with torch.no_grad():
                        coor_inpaint = inpaintnet(coor_pred.to(device), inpaint_mask.to(device)).detach().cpu()
                        coor_inpaint = coor_inpaint * inpaint_mask + coor_pred * (1-inpaint_mask) # Replace missing coordinates with inpainted coordinates

                    # Remove near-zero coordinate predictions
                    th_mask = ((coor_inpaint[:, :, 0] < COOR_TH) & (coor_inpaint[:, :, 1] < COOR_TH))
                    coor_inpaint[th_mask] = 0.

                    # Predict
                    tmp_pred, _ = predict(i, c_pred=coor_inpaint, img_scaler=img_scaler)
                    for key in tmp_pred.keys():
                        inpaint_pred_dict[key].extend(tmp_pred[key])

            else:
                # Use overlap sampling for temporal ensemble
                dataset = Shuttlecock_Trajectory_Dataset(seq_len=seq_len, sliding_step=1, data_mode='coordinate', pred_dict=tracknet_pred_dict)
                data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
                weight = get_ensemble_weight(seq_len, args.eval_mode)

                # Init buffer params
                num_sample, sample_count = len(dataset), 0
                buffer_size = seq_len - 1
                batch_i = torch.arange(seq_len) # [0, 1, 2, 3, 4, 5, 6, 7]
                frame_i = torch.arange(seq_len-1, -1, -1) # [7, 6, 5, 4, 3, 2, 1, 0]
                coor_inpaint_buffer = torch.zeros((buffer_size, seq_len, 2), dtype=torch.float32)

                for step, (i, coor_pred, inpaint_mask) in enumerate(tqdm(data_loader)):
                    coor_pred, inpaint_mask = coor_pred.float(), inpaint_mask.float()
                    b_size = i.shape[0]
                    with torch.no_grad():
                        coor_inpaint = inpaintnet(coor_pred.to(device), inpaint_mask.to(device)).detach().cpu()
                        coor_inpaint = coor_inpaint * inpaint_mask + coor_pred * (1-inpaint_mask)

                    # Remove near-zero coordinate predictions
                    th_mask = ((coor_inpaint[:, :, 0] < COOR_TH) & (coor_inpaint[:, :, 1] < COOR_TH))
                    coor_inpaint[th_mask] = 0.

                    coor_inpaint_buffer = torch.cat((coor_inpaint_buffer, coor_inpaint), dim=0)
                    ensemble_i = torch.empty((0, 1, 2), dtype=torch.float32)
                    ensemble_coor_inpaint = torch.empty((0, 1, 2), dtype=torch.float32)

                    for b in range(b_size):
                        if sample_count < buffer_size:
                            # Buffer is not full yet
                            coor_inpaint = coor_inpaint_buffer[batch_i+b, frame_i].sum(0)
                            coor_inpaint /= (sample_count+1)
                        else:
                            # Normal weighted ensemble
                            coor_inpaint = (coor_inpaint_buffer[batch_i+b, frame_i] * weight[:, None]).sum(0)

                        ensemble_i = torch.cat((ensemble_i, i[b][0].view(1, 1, 2)), dim=0)
                        ensemble_coor_inpaint = torch.cat((ensemble_coor_inpaint, coor_inpaint.view(1, 1, 2)), dim=0)
                        sample_count += 1

                        if sample_count == num_sample:
                            # Flush remaining frames in the last sequence
                            coor_zero_pad = torch.zeros((buffer_size, seq_len, 2), dtype=torch.float32)
                            coor_inpaint_buffer = torch.cat((coor_inpaint_buffer, coor_zero_pad), dim=0)

                            for f in range(1, seq_len):
                                coor_inpaint = coor_inpaint_buffer[batch_i+b+f, frame_i].sum(0)
                                coor_inpaint /= (seq_len-f)
                                ensemble_i = torch.cat((ensemble_i, i[-1][f].view(1, 1, 2)), dim=0)
                                ensemble_coor_inpaint = torch.cat((ensemble_coor_inpaint, coor_inpaint.view(1, 1, 2)), dim=0)

                    # Remove near-zero coordinate predictions
                    th_mask = ((ensemble_coor_inpaint[:, :, 0] < COOR_TH) & (ensemble_coor_inpaint[:, :, 1] < COOR_TH))
                    ensemble_coor_inpaint[th_mask] = 0.

                    # Predict
                    tmp_pred, _ = predict(ensemble_i, c_pred=ensemble_coor_inpaint, img_scaler=img_scaler)
                    for key in tmp_pred.keys():
                        inpaint_pred_dict[key].extend(tmp_pred[key])

                    # Keep the last predictions for the next ensemble window
                    coor_inpaint_buffer = coor_inpaint_buffer[-buffer_size:]


        # Save prediction CSV
        pred_dict = inpaint_pred_dict if inpaintnet is not None else tracknet_pred_dict

        if 'Inpaint_Mask' not in pred_dict:
            if 'Inpaint_Mask' in tracknet_pred_dict:
                pred_dict['Inpaint_Mask'] = tracknet_pred_dict['Inpaint_Mask']
            else:
                pred_dict['Inpaint_Mask'] = [0] * len(pred_dict['Frame'])

        write_pred_csv(pred_dict, save_file=out_csv_file)

        # Save visualization video
        if args.output_video:
            write_pred_video(video_file, pred_dict, save_file=out_video_file, traj_len=args.traj_len)

        print(f'Done: {video_file}')

    print('All done.')
