"""Stroke detection helper functions for stroke_zone_analysis.py."""

import math
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd

# Basic helpers

def is_valid_point(row) -> bool:
    return int(row["Visibility"]) == 1


def safe_int(value):
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return int(value)


def ensure_dir(path: str):
    if path:
        import os
        os.makedirs(path, exist_ok=True)


def calc_step(x1, y1, x2, y2) -> float:
    return math.hypot(float(x2) - float(x1), float(y2) - float(y1))


# Stroke detection

def collect_valid_runs(df: pd.DataFrame) -> List[Dict]:
    """Collect consecutive visible frame runs."""
    runs = []
    i = 0
    n = len(df)

    while i < n:
        if not is_valid_point(df.iloc[i]):
            i += 1
            continue

        start = i
        end = i

        while end + 1 < n:
            cur_frame = int(df.iloc[end]["Frame"])
            next_row = df.iloc[end + 1]
            next_frame = int(next_row["Frame"])

            if not is_valid_point(next_row) or next_frame != cur_frame + 1:
                break
            end += 1

        if end > start:
            runs.append({"start_idx": start, "end_idx": end})
        i = end + 1

    return runs


def find_left_start_idx(
    df: pd.DataFrame,
    start_idx: int,
    end_idx: int,
    min_left_segments: int,
    max_step_th: float,
    max_abs_dy_th: float,
) -> Optional[int]:
    """Find the start of a stable left-moving segment."""
    count = 0
    frame_start_idx = None

    for i in range(start_idx, end_idx):
        x1 = float(df.iloc[i]["X"])
        y1 = float(df.iloc[i]["Y"])
        x2 = float(df.iloc[i + 1]["X"])
        y2 = float(df.iloc[i + 1]["Y"])
        dx = x2 - x1
        dy = y2 - y1
        step = calc_step(x1, y1, x2, y2)

        if step <= max_step_th and abs(dy) <= max_abs_dy_th and dx < 0:
            if count == 0:
                frame_start_idx = i
            count += 1
            if count >= min_left_segments:
                return frame_start_idx
        else:
            count = 0
            frame_start_idx = None

    return None


def make_stroke(
    stroke_id: int,
    run_start_idx: int,
    run_end_idx: int,
    frame_start: int,
    frame_end: int,
    stroke_end_idx: int,
    bounce_frame: int,
    valid: int,
    note: str,
) -> Dict:
    return {
        "stroke_id": stroke_id,
        "run_start_idx": run_start_idx,
        "run_end_idx": run_end_idx,
        "frame_start": frame_start,
        "frame_end": frame_end,
        "stroke_end_idx": stroke_end_idx,
        "bounce_frame": bounce_frame,
        "valid": valid,
        "note": note,
    }


def find_jump_end_idx(
    df: pd.DataFrame,
    start_idx: int,
    end_idx: int,
    max_step_th: float,
) -> int:
    """
    Find an early stroke end when the tracked ball jumps too far.

    The input run is already consecutive and visible. This function only checks
    spatial continuity. If the distance between two adjacent points is larger
    than max_step_th, the stroke ends at the previous frame.
    """
    if end_idx <= start_idx:
        return start_idx

    for i in range(start_idx, end_idx):
        x1 = float(df.iloc[i]["X"])
        y1 = float(df.iloc[i]["Y"])
        x2 = float(df.iloc[i + 1]["X"])
        y2 = float(df.iloc[i + 1]["Y"])
        step = calc_step(x1, y1, x2, y2)

        if step > max_step_th:
            return i

    return end_idx


def has_rightward_motion(
    df: pd.DataFrame,
    start_idx: int,
    end_idx: int,
    min_right_frames: int = 2,
    min_right_dx: float = 35.0,
    max_step_th: float = 300.0,
    right_side_x: Optional[float] = None,
) -> bool:
    """Return True only when the ball really turns and reaches the right side."""
    if end_idx <= start_idx:
        return False

    min_x = float(df.iloc[start_idx]["X"])
    min_idx = start_idx
    right_frames = 0
    has_basic_turn = False
    max_x_after_min = min_x

    for i in range(start_idx + 1, end_idx + 1):
        prev = df.iloc[i - 1]
        cur = df.iloc[i]

        x1 = float(prev["X"])
        y1 = float(prev["Y"])
        x2 = float(cur["X"])
        y2 = float(cur["Y"])

        if calc_step(x1, y1, x2, y2) > max_step_th:
            break

        if x2 < min_x:
            min_x = x2
            min_idx = i
            right_frames = 0
            has_basic_turn = False
            max_x_after_min = x2
            continue

        if i > min_idx:
            max_x_after_min = max(max_x_after_min, x2)

        dx_from_min = x2 - min_x

        if i > min_idx and dx_from_min >= min_right_dx:
            right_frames += 1
            if right_frames >= min_right_frames:
                has_basic_turn = True
        else:
            if dx_from_min < min_right_dx * 0.5:
                right_frames = 0

    if not has_basic_turn:
        return False

    if right_side_x is not None and max_x_after_min < right_side_x:
        return False

    return True

def detect_strokes_from_runs(
    df: pd.DataFrame,
    frame_w: int,
    min_left_segments: int = 5,
    min_candidate_frames: int = 35,
    min_no_hit_candidate_frames: int = 20,
    max_step_th: float = 130.0,
    max_abs_dy_th: float = 45.0,
    left_half_ratio: float = 0.35,
    right_side_ratio: float = 0.5,
) -> List[Dict]:
    """
    Simplified stroke detection for net-zone speed + landing analysis.

    Rule:
    - collect each continuous visible run
    - find a stable left-moving segment as the stroke start
    - normally end at the end of the continuous visible run
    - if a neighboring point jumps farther than max_step_th, end early
    - short left-side misses can be exported as no_hit
    - bounce_frame is not detected here; landing analysis fills it later
    """
    strokes = []
    stroke_id = 1

    for run in collect_valid_runs(df):
        run_s = run["start_idx"]
        run_e = run["end_idx"]
        search_idx = run_s

        while search_idx < run_e:
            frame_start_idx = find_left_start_idx(
                df=df,
                start_idx=search_idx,
                end_idx=run_e,
                min_left_segments=min_left_segments,
                max_step_th=max_step_th,
                max_abs_dy_th=max_abs_dy_th,
            )
            if frame_start_idx is None:
                break

            stroke_end_idx = find_jump_end_idx(
                df=df,
                start_idx=frame_start_idx,
                end_idx=run_e,
                max_step_th=max_step_th,
            )

            frame_start = int(df.iloc[frame_start_idx]["Frame"])
            frame_end = int(df.iloc[stroke_end_idx]["Frame"])
            stroke_len = frame_end - frame_start + 1

            if stroke_len >= min_no_hit_candidate_frames:
                has_hit = False

                # Only long enough strokes can become valid hits. Short left-side
                # strokes are kept only as no_hit candidates.
                if stroke_len >= min_candidate_frames:
                    has_hit = has_rightward_motion(
                        df=df,
                        start_idx=frame_start_idx,
                        end_idx=stroke_end_idx,
                        min_right_frames=2,
                        min_right_dx=35.0,
                        max_step_th=max_step_th,
                        right_side_x=frame_w * right_side_ratio,
                    )

                    # If it currently looks like no_hit, look ahead a little.
                    # This recovers cases where the stroke was cut before the
                    # right-moving part became clear, while still requiring
                    # consecutive right movement and reaching the right side.
                    if not has_hit:
                        min_x = float(df.iloc[frame_start_idx]["X"])
                        for k in range(frame_start_idx, stroke_end_idx + 1):
                            min_x = min(min_x, float(df.iloc[k]["X"]))

                        lookahead_end_idx = min(run_e, stroke_end_idx + min_candidate_frames)

                        recover_right_count = 0
                        recovered_idx = None
                        prev_x = float(df.iloc[stroke_end_idx]["X"])

                        for k in range(stroke_end_idx + 1, lookahead_end_idx + 1):
                            x = float(df.iloc[k]["X"])
                            y = float(df.iloc[k]["Y"])
                            px = float(df.iloc[k - 1]["X"])
                            py = float(df.iloc[k - 1]["Y"])

                            if calc_step(px, py, x, y) > max_step_th:
                                break

                            if x > prev_x:
                                recover_right_count += 1
                            else:
                                recover_right_count = 0

                            prev_x = x

                            if (
                                recover_right_count >= 3
                                and x - min_x >= 80.0
                                and x >= frame_w * right_side_ratio
                            ):
                                recovered_idx = k
                                break

                        if recovered_idx is not None:
                            has_hit = True
                            stroke_end_idx = lookahead_end_idx
                            frame_end = int(df.iloc[stroke_end_idx]["Frame"])
                            stroke_len = frame_end - frame_start + 1

                end_x = float(df.iloc[stroke_end_idx]["X"])
                start_x = float(df.iloc[frame_start_idx]["X"])

                # --------------------------------------------------
                # 過濾不是真正 stroke 的 no_hit 片段
                #
                # has_hit=False 的情況，只有「真的一路往左飛出畫面」才保留成 no_hit。
                # 像 C0078 stroke10 這種是合理 no_hit：
                #   start_x 很右、end_x 很左、整體往左位移很大
                #
                # 像 C0078 stroke17 / stroke28 這種只是慢速殘段或追蹤片段：
                #   不是完整一球、沒有明顯從右側飛到左側
                #   就不要 append 成 stroke。
                # --------------------------------------------------
                if not has_hit:
                    total_left_dx = start_x - end_x

                    # no_hit 必須真的飛到很左邊
                    no_hit_end_left = end_x <= frame_w * 0.12

                    # no_hit 必須有足夠大的整體左向位移
                    no_hit_large_left_motion = total_left_dx >= frame_w * 0.45

                    if not (no_hit_end_left and no_hit_large_left_motion):
                        search_idx = stroke_end_idx + 1
                        continue

                strokes.append(make_stroke(
                    stroke_id=stroke_id,
                    run_start_idx=frame_start_idx,
                    run_end_idx=stroke_end_idx,
                    frame_start=frame_start,
                    frame_end=frame_end,
                    stroke_end_idx=stroke_end_idx,
                    bounce_frame=0,
                    valid=1 if has_hit else 0,
                    note="" if has_hit else "no_hit",
                ))
                stroke_id += 1

            # Continue after the early end. If there was no jump, this exits the run.
            search_idx = stroke_end_idx + 1

    return strokes


# Debug drawing helpers

def build_frame_to_stroke_map(strokes: List[Dict]) -> Dict[int, Dict]:
    frame_map = {}
    for stroke in strokes:
        frame_start = stroke.get("frame_start")
        frame_end = stroke.get("frame_end")
        if frame_start is None or frame_end is None:
            continue
        for frame_id in range(int(frame_start), int(frame_end) + 1):
            frame_map[frame_id] = stroke
    return frame_map


def extract_zone_points(row, prefix: str, point_count: int):
    points = []
    for i in range(1, point_count + 1):
        x = row[f"{prefix}_p{i}_x"]
        y = row[f"{prefix}_p{i}_y"]
        if x == "" or y == "" or pd.isna(x) or pd.isna(y):
            return None
        points.append((int(round(float(x))), int(round(float(y)))))
    return points


def draw_polygon(frame, pts, color, thickness=2, fill=False, alpha=0.18):
    pts_np = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
    if fill:
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts_np], color)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.polylines(frame, [pts_np], True, color, thickness, cv2.LINE_AA)
