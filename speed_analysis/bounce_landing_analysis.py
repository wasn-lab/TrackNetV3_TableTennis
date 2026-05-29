"""
Landing analysis with bounce detection.

This module is intended to be called by stroke_zone_analysis_final.py.
It does two jobs:
1. Find bounce_frame inside each stroke using table-plane projection plus
   a piecewise trajectory fit. Projection is used to mark table/right-half status and landing coordinates;
   bounce existence is decided from image-space motion. Table/right-half filters
   are applied only to the selected bounce frame, not to the whole trajectory.
2. Convert the selected bounce point to table cm coordinates and export
   landing_detail.csv, landing_heatmap.png, landing_zones.png, zone_stats.csv.
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap


TABLE_W = 274.0
TABLE_H = 152.5
GRID_COLS = 8
GRID_ROWS = 4
DPI = 150

OUT_DIR = "runs/"
CURRENT_BASE_NAME = None

# Table projection tolerance. Projection is not used as true 3D height; it is
# only used to keep points close to the table and to compute landing coordinates.
TABLE_MARGIN_X_CM = 10.0
TABLE_MARGIN_Y_CM = 15.0

# Target side. In table-plane coordinates, the net is at TABLE_W / 2.
RIGHT_HALF_ONLY = True
RIGHT_HALF_MARGIN_CM = 0.0

# Piecewise-fit bounce detection.
# A real bounce should split the image-space Y trajectory into two short segments:
# before bounce: image Y increases (ball moves down in the image)
# after bounce : image Y decreases (ball moves up in the image)
SMOOTH_IMAGE_Y_WINDOW = 3
FIT_WINDOW = 6
MIN_SEGMENT_POINTS = 3
# FIT_WINDOW is the maximum look-around window. A bounce does not need
# to have a full 6-frame post segment; it only needs enough points to
# prove that the ball starts rising after the bounce.
MIN_PRE_POINTS = 3
MIN_POST_POINTS = 2
POST_EXTRA_FRAMES = 8
MAX_FRAME_GAP_IN_FIT = 4
# Local fitting windows should contain the same ball track.
# If adjacent visible points jump too far, they are probably a new/mis-tracked segment
# and should not be used as post-bounce support.
MAX_LOCAL_STEP_PX = 220.0

# Adaptive thresholds use the local Y range inside each stroke. These floors avoid
# tiny jitter being treated as a bounce.
MIN_DROP_BEFORE_PX_FLOOR = 3.0
MIN_RISE_AFTER_PX_FLOOR = 6.0
MIN_DROP_BEFORE_RATIO = 0.08
MIN_RISE_AFTER_RATIO = 0.10
MIN_SLOPE_PX_PER_FRAME = 0.6
LOCAL_PEAK_TOL_PX = 2.0

# The target stroke should be moving to the right around bounce.
# This is checked on image-space X, but only for the candidate bounce frame.
MIN_RIGHTWARD_MOVE_PX = 5.0

# Terminal bounce handles cases where the ball lands near the end of a stroke
# and the post-bounce rise is missing, very short, or only a flat plateau.
# Normal bounce still uses TABLE_MARGIN_X_CM / TABLE_MARGIN_Y_CM.
# Terminal bounce allows a slightly larger X tolerance for far-edge projection error.
TERMINAL_LOOKBACK_POINTS = 10
TERMINAL_TABLE_MARGIN_X_CM = 20
TERMINAL_TABLE_MARGIN_Y_CM = TABLE_MARGIN_Y_CM
TERMINAL_MIN_DROP_BEFORE_PX = 18.0
TERMINAL_MIN_PRE_SLOPE = 1.2
TERMINAL_PLATEAU_TOL_PX = 8.0
TERMINAL_POST_RISE_TOL_PX = 8.0
TERMINAL_LAST_RAW_DY_MAX = 8.0
# Do not accept terminal candidates if the ball is still clearly moving downward after the candidate.
TERMINAL_MAX_POST_DOWN_SLOPE = 2.0
NORMAL_FUTURE_REDESCEND_TOL_PX = 20.0

# Clear V-shaped bounces near the far/right table edge can be projected a few
# centimeters outside the table because the ball is above the table plane.
# Use this only for candidates that are already near the far edge and pass the
# normal V-shaped bounce checks.
EDGE_NORMAL_TABLE_MARGIN_X_CM = 20.0
EDGE_NORMAL_MIN_X_CM = TABLE_W - 40.0

# Flat-to-rise bounce handles late bounces where the incoming drop is nearly
# flat in image space, but the post-bounce rise is clear. This happens near
# table edges or under perspective when the ball is already close to the table
# and the pre-bounce Y change is too small for normal V rules.
FLAT_REBOUND_LOOKBACK_POINTS = 12
FLAT_PRE_SLOPE_ABS_MAX = 1.0
FLAT_MAX_DROP_BEFORE_PX = 6.0
FLAT_MIN_RISE_AFTER_PX = 18.0
FLAT_POST_SLOPE_MAX = -1.0
FLAT_FUTURE_REDESCEND_TOL_PX = 10.0


REQUIRED_TABLE_COLS = [
    "table_p1_x", "table_p1_y",
    "table_p2_x", "table_p2_y",
    "table_p3_x", "table_p3_y",
    "table_p4_x", "table_p4_y",
]


def _output_path(filename: str) -> str:
    if CURRENT_BASE_NAME:
        return os.path.join(OUT_DIR, f"{CURRENT_BASE_NAME}_{filename}")
    return os.path.join(OUT_DIR, filename)


def has_valid_table(row) -> bool:
    for col in REQUIRED_TABLE_COLS:
        if col not in row.index:
            return False
        value = row[col]
        if value == "" or pd.isna(value):
            return False
    return True


def pixel_to_cm(ball_px, ball_py, table_corners_px):
    """
    Convert a ball pixel point to table-plane coordinates in cm.
    table_corners order: LT, RT, RB, LB.
    """
    dst_corners = np.array([
        [0, 0],
        [TABLE_W, 0],
        [TABLE_W, TABLE_H],
        [0, TABLE_H],
    ], dtype=np.float32)

    src_corners = np.asarray(table_corners_px, dtype=np.float32)
    H, _ = cv2.findHomography(src_corners, dst_corners)
    if H is None:
        raise ValueError("findHomography failed")

    pt = np.array([[[float(ball_px), float(ball_py)]]], dtype=np.float32)
    dst = cv2.perspectiveTransform(pt, H)
    return float(dst[0, 0, 0]), float(dst[0, 0, 1])


def get_table_corners(row):
    return np.array([
        [row["table_p1_x"], row["table_p1_y"]],
        [row["table_p2_x"], row["table_p2_y"]],
        [row["table_p3_x"], row["table_p3_y"]],
        [row["table_p4_x"], row["table_p4_y"]],
    ], dtype=np.float64)


def assign_zone(x_cm, y_cm):
    col_w = TABLE_W / GRID_COLS
    row_h = TABLE_H / GRID_ROWS
    col = int(np.clip(x_cm // col_w, 0, GRID_COLS - 1))
    row = int(np.clip(y_cm // row_h, 0, GRID_ROWS - 1))
    return col, row


def in_table_relaxed(x_cm, y_cm, margin_x_cm=TABLE_MARGIN_X_CM, margin_y_cm=TABLE_MARGIN_Y_CM) -> bool:
    return (
        -margin_x_cm <= x_cm <= TABLE_W + margin_x_cm and
        -margin_y_cm <= y_cm <= TABLE_H + margin_y_cm
    )


def in_table_strict(x_cm, y_cm) -> bool:
    return 0 <= x_cm <= TABLE_W and 0 <= y_cm <= TABLE_H


def is_right_half_landing(x_cm, margin_cm=RIGHT_HALF_MARGIN_CM) -> bool:
    """Return True if the projected point is on the right side of the net."""
    return x_cm >= (TABLE_W / 2.0 - margin_cm)


def _safe_visible_rows(traj: pd.DataFrame) -> pd.DataFrame:
    visible = traj.copy()
    visible = visible[visible["Visibility"] == 1].copy()
    visible = visible[(visible["X"] > 0) & (visible["Y"] > 0)].copy()
    return visible


def _project_candidates(stroke_row, traj, table_margin_x_cm, table_margin_y_cm, right_half_only, right_half_margin_cm):
    """Project visible stroke points to table plane.

    Do not filter by table margin or right-half here. These points are the
    trajectory used for fitting the bounce shape. The table/right-half status is
    only recorded and will be applied to the final bounce candidate.
    """
    frame_start = int(stroke_row["frame_start"])
    frame_end = int(stroke_row["frame_end"])
    hit_frame = int(stroke_row.get("hit_frame", frame_start))
    if frame_end < frame_start:
        return []

    # Only frames up to frame_end can be selected as bounce candidates.
    # A few frames after frame_end are kept only as support points, so a
    # late bounce can still be verified even when the stroke segment ends
    # before a full FIT_WINDOW is available.
    support_end = frame_end + POST_EXTRA_FRAMES

    stroke_traj = _safe_visible_rows(traj)
    stroke_traj = stroke_traj[(stroke_traj["Frame"] >= hit_frame) & (stroke_traj["Frame"] <= support_end)].copy()
    stroke_traj = stroke_traj.sort_values("Frame").reset_index(drop=True)
    if stroke_traj.empty:
        return []

    table_corners = get_table_corners(stroke_row)
    candidates = []

    for _, r in stroke_traj.iterrows():
        ball_px = float(r["X"])
        ball_py = float(r["Y"])

        try:
            x_cm, y_cm = pixel_to_cm(ball_px, ball_py, table_corners)
        except Exception:
            continue

        strict = in_table_strict(x_cm, y_cm)
        relaxed = in_table_relaxed(
            x_cm,
            y_cm,
            margin_x_cm=table_margin_x_cm,
            margin_y_cm=table_margin_y_cm,
        )
        right_half = is_right_half_landing(x_cm, margin_cm=right_half_margin_cm)

        frame = int(r["Frame"])
        is_candidate_frame = frame <= frame_end

        candidates.append({
            "stroke_id": int(stroke_row["stroke_id"]),
            "bounce_frame": frame,
            "is_candidate_frame": bool(is_candidate_frame),
            "ball_px": ball_px,
            "ball_py": ball_py,
            "x_cm": float(x_cm),
            "y_cm": float(y_cm),
            "in_table": bool(relaxed),
            "in_table_strict": bool(strict),
            "in_table_relaxed": bool(relaxed),
            "edge_bounce": bool(relaxed and not strict),
            "right_half_landing": bool(right_half),
        })

    return candidates


def _is_continuous_frames(frames, max_gap=MAX_FRAME_GAP_IN_FIT):
    if len(frames) < 2:
        return True
    gaps = np.diff(np.asarray(frames, dtype=float))
    return bool(np.all(gaps <= max_gap))


def _same_track_step_ok(df, a, b):
    """Return True when two adjacent support points still look like the same ball track."""
    frame_gap = int(df.loc[b, "bounce_frame"]) - int(df.loc[a, "bounce_frame"])
    if frame_gap > MAX_FRAME_GAP_IN_FIT:
        return False

    dx = float(df.loc[b, "ball_px"]) - float(df.loc[a, "ball_px"])
    dy = float(df.loc[b, "ball_py"]) - float(df.loc[a, "ball_py"])
    if float(np.hypot(dx, dy)) > MAX_LOCAL_STEP_PX:
        return False

    return True


def _local_pre_post_windows(df, i):
    """Return local pre/post windows around candidate index i.

    FIT_WINDOW is the maximum number of points on each side including the
    candidate. The window stops at a large frame gap or a large pixel jump,
    because that usually means the visible points have switched to another ball
    track or a wrong detection. This is a general same-track check, not a
    stroke-specific rule.
    """
    n = len(df)

    pre_start = i
    while pre_start > 0 and (i - pre_start + 1) < FIT_WINDOW:
        if not _same_track_step_ok(df, pre_start - 1, pre_start):
            break
        pre_start -= 1

    post_end = i
    while post_end + 1 < n and (post_end - i + 1) < FIT_WINDOW:
        if not _same_track_step_ok(df, post_end, post_end + 1):
            break
        post_end += 1

    pre = df.iloc[pre_start:i + 1]
    post = df.iloc[i:post_end + 1]
    return pre, post



def _future_same_track_candidate_rows(df, i):
    """Return future candidate rows only until the same-track chain breaks.

    Future re-descend checks should not look across a large pixel jump, because
    that usually means the tracker has switched to another ball segment.
    """
    indices = []
    prev = i
    for j in range(i + 1, len(df)):
        if not _same_track_step_ok(df, prev, j):
            break
        prev = j
        if "is_candidate_frame" not in df.columns or bool(df.loc[j, "is_candidate_frame"]):
            indices.append(j)

    if not indices:
        return df.iloc[0:0]
    return df.loc[indices]

def _line_fit_stats(frames, values):
    """Return slope, rmse, predicted values for a line fit."""
    x = np.asarray(frames, dtype=float)
    y = np.asarray(values, dtype=float)
    if len(x) < 2 or np.ptp(x) < 1e-6:
        return None

    x0 = x - x[0]
    slope, intercept = np.polyfit(x0, y, 1)
    pred = slope * x0 + intercept
    rmse = float(np.sqrt(np.mean((y - pred) ** 2)))
    return float(slope), rmse, pred


def _valid_landing_candidate(row, right_half_only=True, margin_x_cm=TABLE_MARGIN_X_CM, margin_y_cm=TABLE_MARGIN_Y_CM):
    """Check only the candidate bounce point, not the surrounding trajectory."""
    x_cm = float(row["x_cm"])
    y_cm = float(row["y_cm"])
    if not in_table_relaxed(x_cm, y_cm, margin_x_cm=margin_x_cm, margin_y_cm=margin_y_cm):
        return False
    if right_half_only and not is_right_half_landing(x_cm):
        return False
    return True


def _rightward_window_ok(pre, post):
    """The local trajectory around bounce should move left-to-right in image space."""
    pre_x = pre["ball_px"].astype(float).to_numpy()
    post_x = post["ball_px"].astype(float).to_numpy()
    cur_x = float(post.iloc[0]["ball_px"])
    if float(post_x[-1] - pre_x[0]) < MIN_RIGHTWARD_MOVE_PX:
        return False
    if cur_x < float(pre_x[0]) - MIN_RIGHTWARD_MOVE_PX:
        return False
    if float(post_x[-1]) < cur_x - MIN_RIGHTWARD_MOVE_PX:
        return False
    return True


def _make_bounce_result(df, i, bounce_type, drop_before, rise_after, pre_slope, post_slope, min_drop, min_rise, accepted_margin_x_cm=TABLE_MARGIN_X_CM, accepted_margin_y_cm=TABLE_MARGIN_Y_CM):
    best = df.loc[i].to_dict()
    best.update({
        "index": i,
        "drop_before_px": float(drop_before),
        "rise_after_px": float(rise_after),
        "pre_slope_px_per_frame": float(pre_slope),
        "post_slope_px_per_frame": float(post_slope),
        "min_drop_before_px": float(min_drop),
        "min_rise_after_px": float(min_rise),
        "accepted_margin_x_cm": float(accepted_margin_x_cm),
        "accepted_margin_y_cm": float(accepted_margin_y_cm),
        "bounce_type": bounce_type,
    })
    return best


def _select_terminal_bounce(df, right_half_only=True, require_rightward=True):
    """
    Find a late/terminal bounce without requiring a visible V-shaped rebound.

    This covers general cases where the bounce is near frame_end and the ball
    either becomes flat after landing or the stroke ends before enough post-bounce
    frames exist. It is not stroke-id specific.
    """
    candidate_indices = [
        i for i in range(len(df))
        if ("is_candidate_frame" not in df.columns or bool(df.loc[i, "is_candidate_frame"]))
    ]
    if not candidate_indices:
        return None

    last_candidate_i = candidate_indices[-1]
    first_i = max(candidate_indices[0], last_candidate_i - TERMINAL_LOOKBACK_POINTS + 1)

    for i in range(first_i, last_candidate_i + 1):
        if i < MIN_PRE_POINTS - 1:
            continue
        if "is_candidate_frame" in df.columns and not bool(df.loc[i, "is_candidate_frame"]):
            continue

        # Terminal bounce can be slightly outside the far edge because homography
        # projection at the right edge is sensitive to height and corner error.
        if not _valid_landing_candidate(
            df.loc[i],
            right_half_only=right_half_only,
            margin_x_cm=TERMINAL_TABLE_MARGIN_X_CM,
            margin_y_cm=TERMINAL_TABLE_MARGIN_Y_CM,
        ):
            continue

        pre, post = _local_pre_post_windows(df, i)

        if len(pre) < MIN_PRE_POINTS:
            continue
        if not _is_continuous_frames(pre["bounce_frame"].to_numpy()):
            continue
        if len(post) > 1 and not _is_continuous_frames(post["bounce_frame"].to_numpy()):
            continue
        if require_rightward and len(post) > 1 and not _rightward_window_ok(pre, post):
            continue
        if require_rightward and len(post) == 1:
            # No post point exists; at least require pre-to-current rightward motion.
            pre_x = pre["ball_px"].astype(float).to_numpy()
            if float(pre_x[-1] - pre_x[0]) < MIN_RIGHTWARD_MOVE_PX:
                continue

        cur_y = float(df.loc[i, "ball_py_smooth"])
        cur_raw_y = float(df.loc[i, "ball_py"])
        pre_y = pre["ball_py_smooth"].astype(float).to_numpy()
        post_y = post["ball_py_smooth"].astype(float).to_numpy()

        # Must be close to the terminal local maximum, but allow the first frame
        # of a plateau instead of forcing the very last/highest frame.
        terminal_tail = df.iloc[i:min(len(df), last_candidate_i + 1)]
        tail_max_y = float(terminal_tail["ball_py_smooth"].max())
        if cur_y + TERMINAL_PLATEAU_TOL_PX < tail_max_y:
            continue
        if cur_y + LOCAL_PEAK_TOL_PX < float(np.max(pre_y[:-1])):
            continue

        drop_before = cur_y - float(np.min(pre_y[:-1]))
        if drop_before < TERMINAL_MIN_DROP_BEFORE_PX:
            continue

        pre_fit = _line_fit_stats(pre["bounce_frame"].to_numpy(), pre_y)
        pre_slope = float(pre_fit[0]) if pre_fit is not None else 0.0
        if pre_slope < TERMINAL_MIN_PRE_SLOPE:
            continue

        # If post frames exist, accept a short terminal plateau or a short
        # terminal rebound. Near the end of a stroke, a real bounce may not have
        # enough frames to form a full V shape, so do not reject it just because
        # the ball rises more than a small amount after the bounce. The important
        # safety check is that the ball must not continue going lower after the
        # selected candidate.
        if len(post_y) > 1:
            post_after = post_y[1:]
            post_max_extra = float(np.max(post_after) - cur_y)
            post_rise = float(cur_y - np.min(post_after))
            if post_max_extra > TERMINAL_PLATEAU_TOL_PX:
                continue
            post_fit = _line_fit_stats(post["bounce_frame"].to_numpy(), post_y)
            post_slope = float(post_fit[0]) if post_fit is not None else 0.0
            # If the point after the candidate is still clearly moving downward
            # and there is no rebound, this is not a terminal bounce yet.
            if post_slope > TERMINAL_MAX_POST_DOWN_SLOPE and post_rise <= 0.0:
                continue
            rise_after = max(0.0, post_rise)
        else:
            if i <= 0:
                continue
            prev_raw_y = float(df.loc[i - 1, "ball_py"])
            last_raw_dy = cur_raw_y - prev_raw_y
            if last_raw_dy > TERMINAL_LAST_RAW_DY_MAX:
                continue
            post_slope = 0.0
            rise_after = 0.0

        return _make_bounce_result(
            df, i,
            bounce_type="rule_terminal",
            drop_before=drop_before,
            rise_after=rise_after,
            pre_slope=pre_slope,
            post_slope=post_slope,
            min_drop=TERMINAL_MIN_DROP_BEFORE_PX,
            min_rise=0.0,
            accepted_margin_x_cm=TERMINAL_TABLE_MARGIN_X_CM,
            accepted_margin_y_cm=TERMINAL_TABLE_MARGIN_Y_CM,
        )

    return None


def _select_flat_rebound_bounce(df, right_half_only=True, require_rightward=True):
    """
    Find a late flat-to-rise bounce.

    This is for cases where the bounce is near the end of the stroke, the
    pre-bounce image Y is almost flat, but the post-bounce rise is clear.
    It does not use scores and is not stroke-id specific.
    """
    candidate_indices = [
        i for i in range(len(df))
        if ("is_candidate_frame" not in df.columns or bool(df.loc[i, "is_candidate_frame"]))
    ]
    if not candidate_indices:
        return None

    last_candidate_i = candidate_indices[-1]
    first_i = max(candidate_indices[0], last_candidate_i - FLAT_REBOUND_LOOKBACK_POINTS + 1)

    for i in range(first_i, last_candidate_i + 1):
        if i < MIN_PRE_POINTS - 1:
            continue
        if "is_candidate_frame" in df.columns and not bool(df.loc[i, "is_candidate_frame"]):
            continue

        if not _valid_landing_candidate(
            df.loc[i],
            right_half_only=right_half_only,
            margin_x_cm=TABLE_MARGIN_X_CM,
            margin_y_cm=TABLE_MARGIN_Y_CM,
        ):
            continue

        pre, post = _local_pre_post_windows(df, i)

        if len(pre) < MIN_PRE_POINTS:
            continue
        if len(post) < MIN_POST_POINTS:
            continue
        if not _is_continuous_frames(pre["bounce_frame"].to_numpy()):
            continue
        if not _is_continuous_frames(post["bounce_frame"].to_numpy()):
            continue
        if require_rightward and not _rightward_window_ok(pre, post):
            continue

        cur_y = float(df.loc[i, "ball_py_smooth"])
        pre_y = pre["ball_py_smooth"].astype(float).to_numpy()
        post_y = post["ball_py_smooth"].astype(float).to_numpy()

        # Candidate should be at the top of a small plateau / local image bottom.
        if cur_y + LOCAL_PEAK_TOL_PX < float(np.max(pre_y[:-1])):
            continue
        if cur_y + LOCAL_PEAK_TOL_PX < float(np.max(post_y[1:])):
            continue

        drop_before = cur_y - float(np.min(pre_y[:-1]))
        rise_after = cur_y - float(np.min(post_y[1:]))
        if drop_before < 0:
            continue
        if drop_before > FLAT_MAX_DROP_BEFORE_PX:
            continue
        if rise_after < FLAT_MIN_RISE_AFTER_PX:
            continue

        pre_fit = _line_fit_stats(pre["bounce_frame"].to_numpy(), pre_y)
        post_fit = _line_fit_stats(post["bounce_frame"].to_numpy(), post_y)
        pre_slope = float(pre_fit[0]) if pre_fit is not None else 0.0
        post_slope = float(post_fit[0]) if post_fit is not None else 0.0

        # Pre-bounce is allowed to be shallow/flat, but not already strongly rising.
        if abs(pre_slope) > FLAT_PRE_SLOPE_ABS_MAX:
            continue
        if post_slope > FLAT_POST_SLOPE_MAX:
            continue

        # Avoid accepting a mid-flight fake bounce if the ball later goes much lower.
        future = _future_same_track_candidate_rows(df, i)
        if len(future) > 0:
            future_max_y = float(future["ball_py_smooth"].max())
            if future_max_y > cur_y + FLAT_FUTURE_REDESCEND_TOL_PX:
                continue

        return _make_bounce_result(
            df, i,
            bounce_type="rule_flat_rebound",
            drop_before=drop_before,
            rise_after=rise_after,
            pre_slope=pre_slope,
            post_slope=post_slope,
            min_drop=0.0,
            min_rise=FLAT_MIN_RISE_AFTER_PX,
            accepted_margin_x_cm=TABLE_MARGIN_X_CM,
            accepted_margin_y_cm=TABLE_MARGIN_Y_CM,
        )

    return None

def _select_normal_bounce(df, right_half_only=True, require_rightward=True, margin_x_cm=TABLE_MARGIN_X_CM, margin_y_cm=TABLE_MARGIN_Y_CM, require_far_edge=False, bounce_type="rule_normal"):
    """Find a normal V-shaped bounce with visible post-bounce rise."""
    n = len(df)

    for i in range(MIN_PRE_POINTS - 1, n - MIN_POST_POINTS + 1):
        if "is_candidate_frame" in df.columns and not bool(df.loc[i, "is_candidate_frame"]):
            continue
        if not _valid_landing_candidate(
            df.loc[i],
            right_half_only=right_half_only,
            margin_x_cm=margin_x_cm,
            margin_y_cm=margin_y_cm,
        ):
            continue

        if require_far_edge and float(df.loc[i, "x_cm"]) < EDGE_NORMAL_MIN_X_CM:
            continue

        pre, post = _local_pre_post_windows(df, i)

        if len(pre) < MIN_PRE_POINTS:
            continue
        if len(post) < MIN_POST_POINTS:
            continue
        if not _is_continuous_frames(pre["bounce_frame"].to_numpy()):
            continue
        if not _is_continuous_frames(post["bounce_frame"].to_numpy()):
            continue
        if require_rightward and not _rightward_window_ok(pre, post):
            continue

        cur_y = float(df.loc[i, "ball_py_smooth"])
        pre_y = pre["ball_py_smooth"].astype(float).to_numpy()
        post_y = post["ball_py_smooth"].astype(float).to_numpy()

        if cur_y + LOCAL_PEAK_TOL_PX < float(np.max(pre_y[:-1])):
            continue
        if cur_y + LOCAL_PEAK_TOL_PX < float(np.max(post_y[1:])):
            continue

        # Reject early fake V points if the ball later continues downward far
        # below the supposed bounce. This keeps the original normal-bounce behavior
        # while avoiding mid-flight V false positives such as long arcs.
        future = _future_same_track_candidate_rows(df, i)
        if len(future) > 0:
            future_max_y = float(future["ball_py_smooth"].max())
            if future_max_y > cur_y + NORMAL_FUTURE_REDESCEND_TOL_PX:
                continue

        drop_before = cur_y - float(np.min(pre_y[:-1]))
        rise_after = cur_y - float(np.min(post_y[1:]))

        local_y_range = float(max(np.max(pre_y) - np.min(pre_y), np.max(post_y) - np.min(post_y)))
        min_drop = max(MIN_DROP_BEFORE_PX_FLOOR, local_y_range * MIN_DROP_BEFORE_RATIO)
        min_rise = max(MIN_RISE_AFTER_PX_FLOOR, local_y_range * MIN_RISE_AFTER_RATIO)

        if drop_before < min_drop:
            continue
        if rise_after < min_rise:
            continue

        pre_fit = _line_fit_stats(pre["bounce_frame"].to_numpy(), pre_y)
        post_fit = _line_fit_stats(post["bounce_frame"].to_numpy(), post_y)
        pre_slope = float(pre_fit[0]) if pre_fit is not None else 0.0
        post_slope = float(post_fit[0]) if post_fit is not None else 0.0

        if pre_slope < MIN_SLOPE_PX_PER_FRAME:
            continue
        if post_slope > -MIN_SLOPE_PX_PER_FRAME:
            continue

        return _make_bounce_result(
            df, i,
            bounce_type=bounce_type,
            drop_before=drop_before,
            rise_after=rise_after,
            pre_slope=pre_slope,
            post_slope=post_slope,
            min_drop=min_drop,
            min_rise=min_rise,
            accepted_margin_x_cm=margin_x_cm,
            accepted_margin_y_cm=margin_y_cm,
        )

    return None


def _select_bounce_by_piecewise_fit(candidates, right_half_only=True, require_rightward=True):
    """
    Select bounce without scores.

    Two deterministic passes are used:
    1. terminal rule for late landing/plateau/no-post-frame cases;
    2. normal rule for clear V-shaped bounces with visible post-bounce rise.

    Both rules are general pass/fail rules. No stroke id or per-video customization
    is used.
    """
    if len(candidates) < (MIN_PRE_POINTS + MIN_POST_POINTS - 1):
        return None

    df = pd.DataFrame(candidates).sort_values("bounce_frame").reset_index(drop=True)
    df["ball_py_smooth"] = df["ball_py"].rolling(
        window=SMOOTH_IMAGE_Y_WINDOW,
        center=True,
        min_periods=1,
    ).median()

    terminal = _select_terminal_bounce(
        df,
        right_half_only=right_half_only,
        require_rightward=require_rightward,
    )
    if terminal is not None:
        return terminal

    flat_rebound = _select_flat_rebound_bounce(
        df,
        right_half_only=right_half_only,
        require_rightward=require_rightward,
    )
    if flat_rebound is not None:
        return flat_rebound

    normal = _select_normal_bounce(
        df,
        right_half_only=right_half_only,
        require_rightward=require_rightward,
        margin_x_cm=TABLE_MARGIN_X_CM,
        margin_y_cm=TABLE_MARGIN_Y_CM,
        require_far_edge=False,
        bounce_type="rule_normal",
    )
    if normal is not None:
        return normal

    return _select_normal_bounce(
        df,
        right_half_only=right_half_only,
        require_rightward=require_rightward,
        margin_x_cm=EDGE_NORMAL_TABLE_MARGIN_X_CM,
        margin_y_cm=TABLE_MARGIN_Y_CM,
        require_far_edge=True,
        bounce_type="rule_edge_normal",
    )

def find_bounce_by_piecewise_fit(
    stroke_row,
    traj: pd.DataFrame,
    table_margin_x_cm: float = TABLE_MARGIN_X_CM,
    table_margin_y_cm: float = TABLE_MARGIN_Y_CM,
    right_half_only: bool = RIGHT_HALF_ONLY,
    right_half_margin_cm: float = RIGHT_HALF_MARGIN_CM,
):
    """
    Find bounce_frame using table projection + piecewise image-space Y fitting.

    Projection is used only for:
    - keeping candidates near the table;
    - restricting to the target/right half;
    - computing landing x_cm/y_cm and zone label.

    Bounce existence is decided from the image-space Y trajectory. If no valid
    downward-then-upward split exists, returns None.
    """
    if not has_valid_table(stroke_row):
        return None

    candidates = _project_candidates(
        stroke_row,
        traj,
        table_margin_x_cm=table_margin_x_cm,
        table_margin_y_cm=table_margin_y_cm,
        right_half_only=right_half_only,
        right_half_margin_cm=right_half_margin_cm,
    )
    if len(candidates) < (MIN_PRE_POINTS + MIN_POST_POINTS - 1):
        return None

    best = _select_bounce_by_piecewise_fit(
        candidates,
        right_half_only=right_half_only,
        require_rightward=True,
    )
    if best is None:
        return None

    # Final safety check. Re-check with the margin that accepted this rule.
    # Do not use best["in_table_relaxed"] here because it is recorded with the
    # default TABLE_MARGIN_X_CM only; terminal/edge rules may intentionally use
    # a larger far-edge margin.
    final_margin_x = float(best.get("accepted_margin_x_cm", TABLE_MARGIN_X_CM))
    final_margin_y = float(best.get("accepted_margin_y_cm", TABLE_MARGIN_Y_CM))
    if not _valid_landing_candidate(best, right_half_only=right_half_only, margin_x_cm=final_margin_x, margin_y_cm=final_margin_y):
        return None

    col, row = assign_zone(float(best["x_cm"]), float(best["y_cm"]))

    return {
        "stroke_id": int(best["stroke_id"]),
        "bounce_frame": int(best["bounce_frame"]),
        "ball_px": round(float(best["ball_px"]), 1),
        "ball_py": round(float(best["ball_py"]), 1),
        "x_cm": round(float(best["x_cm"]), 1),
        "y_cm": round(float(best["y_cm"]), 1),
        "in_table": True,
        "in_table_strict": bool(best["in_table_strict"]),
        "in_table_relaxed": True,
        "edge_bounce": bool(best["edge_bounce"]),
        "right_half_landing": bool(best["right_half_landing"]),
        "bounce_type": str(best["bounce_type"]),
        "ball_py_smooth": round(float(best["ball_py_smooth"]), 1),
        "drop_before_px": round(float(best["drop_before_px"]), 1),
        "rise_after_px": round(float(best["rise_after_px"]), 1),
        "pre_slope_px_per_frame": round(float(best["pre_slope_px_per_frame"]), 3),
        "post_slope_px_per_frame": round(float(best["post_slope_px_per_frame"]), 3),
        "zone_col": col,
        "zone_row": row,
        "zone_label": f"C{col + 1}R{row + 1}",
    }



def compute_landings_with_bounce(
    strokes: pd.DataFrame,
    traj: pd.DataFrame,
    save_dir: str = None,
    base_name: str = None,
    table_margin_x_cm: float = TABLE_MARGIN_X_CM,
    table_margin_y_cm: float = TABLE_MARGIN_Y_CM,
    min_bounce_score: float = None,
    right_half_only: bool = RIGHT_HALF_ONLY,
    right_half_margin_cm: float = RIGHT_HALF_MARGIN_CM,
) -> pd.DataFrame:
    """
    Compute landing information and bounce_frame for each stroke.
    Returns one row per stroke with a detected table-near bounce.
    """
    global OUT_DIR, CURRENT_BASE_NAME
    CURRENT_BASE_NAME = base_name
    if save_dir is not None:
        OUT_DIR = save_dir
        os.makedirs(OUT_DIR, exist_ok=True)

    if strokes is None or strokes.empty:
        print("[WARN] empty stroke summary, skip landing analysis.")
        return pd.DataFrame()

    missing_cols = [c for c in ["stroke_id", "frame_start", "frame_end"] + REQUIRED_TABLE_COLS if c not in strokes.columns]
    if missing_cols:
        print(f"[WARN] missing columns for landing analysis: {missing_cols}")
        return pd.DataFrame()

    records = []
    skipped_no_geometry = 0
    skipped_no_bounce = 0

    for _, s in strokes.iterrows():
        if not has_valid_table(s):
            skipped_no_geometry += 1
            continue

        note = str(s.get("note", "")).lower()
        if "net_stop" in note or "net_hit" in note:
            skipped_no_bounce += 1
            continue

        result = find_bounce_by_piecewise_fit(
            s,
            traj,
            table_margin_x_cm=table_margin_x_cm,
            table_margin_y_cm=table_margin_y_cm,
            right_half_only=right_half_only,
            right_half_margin_cm=right_half_margin_cm,
        )
        if result is None:
            skipped_no_bounce += 1
            continue
        records.append(result)

    df_land = pd.DataFrame(records)
    if not df_land.empty and base_name:
        df_land.insert(0, "video_name", base_name)

    side_msg = "right-half only" if right_half_only else "all table"
    print(
        f"[landing] detected bounce: {len(df_land)} strokes ({side_msg}), "
        f"no_geometry={skipped_no_geometry}, no_bounce={skipped_no_bounce}"
    )

    if not df_land.empty:
        save_stats(df_land)
        plot_heatmap(df_land)
        plot_scatter(df_land)

    return df_land



def plot_heatmap(df):
    zone_count = np.zeros((GRID_ROWS, GRID_COLS), dtype=int)
    for _, r in df[df["in_table"]].iterrows():
        zone_count[int(r["zone_row"]), int(r["zone_col"])] += 1

    fig, ax = plt.subplots(figsize=(10, 6), dpi=DPI)
    cmap = LinearSegmentedColormap.from_list("tt", ["#f7f7f7", "#f4a261", "#e63946"])

    im = ax.imshow(
        zone_count,
        cmap=cmap,
        aspect="auto",
        vmin=0,
        vmax=max(zone_count.max(), 1),
        extent=[0, TABLE_W, TABLE_H, 0],
    )

    for c in np.linspace(0, TABLE_W, GRID_COLS + 1):
        ax.axvline(c, color="k", lw=1.2)
    for r in np.linspace(0, TABLE_H, GRID_ROWS + 1):
        ax.axhline(r, color="k", lw=1.2)

    ax.axvline(TABLE_W / 2, color="white", lw=2.5, ls="--", label="net")

    col_w = TABLE_W / GRID_COLS
    row_h = TABLE_H / GRID_ROWS
    for ri in range(GRID_ROWS):
        for ci in range(GRID_COLS):
            cnt = zone_count[ri, ci]
            cx = ci * col_w + col_w / 2
            cy = ri * row_h + row_h / 2
            ax.text(cx, cy, str(cnt), ha="center", va="center", fontsize=16, fontweight="bold")
            ax.text(cx, cy + row_h * 0.3, f"C{ci + 1}R{ri + 1}", ha="center", va="center", fontsize=8, color="gray")

    plt.colorbar(im, ax=ax, label="ball count")
    ax.set_xlabel("X (cm) →", fontsize=12)
    ax.set_ylabel("Y (cm) ↓", fontsize=12)
    ax.set_title(f"heatmap({GRID_COLS}×{GRID_ROWS})\n{int(df['in_table'].sum())} ball on table", fontsize=13)
    ax.legend(loc="upper right", fontsize=9)

    out = _output_path("landing_heatmap.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[儲存] {out}")


def plot_scatter(df):
    fig, ax = plt.subplots(figsize=(10, 6), dpi=DPI)
    ax.add_patch(patches.Rectangle((0, 0), TABLE_W, TABLE_H, linewidth=2, edgecolor="k", facecolor="#d4edda"))

    for c in np.linspace(0, TABLE_W, GRID_COLS + 1):
        ax.axvline(c, color="k", lw=0.8, alpha=0.5)
    for r in np.linspace(0, TABLE_H, GRID_ROWS + 1):
        ax.axhline(r, color="k", lw=0.8, alpha=0.5)

    ax.axvline(TABLE_W / 2, color="white", lw=3, ls="--", label="net")

    inside = df[df["in_table"]]
    outside = df[~df["in_table"]]

    ax.scatter(inside["x_cm"], inside["y_cm"], c="#e63946", s=80, zorder=5, label=f"on table ({len(inside)})")
    if len(outside) > 0:
        ax.scatter(outside["x_cm"], outside["y_cm"], c="#adb5bd", s=60, marker="x", zorder=4, label=f"outside ({len(outside)})")

    for _, r in inside.iterrows():
        ax.annotate(str(int(r["stroke_id"])), (r["x_cm"], r["y_cm"]), textcoords="offset points", xytext=(4, 4), fontsize=7)

    ax.set_xlim(-10, TABLE_W + 10)
    ax.set_ylim(-10, TABLE_H + 10)
    ax.invert_yaxis()
    ax.set_xlabel("X (cm) →", fontsize=12)
    ax.set_ylabel("Y (cm) ↓", fontsize=12)
    ax.set_title("landing scatter", fontsize=13)
    ax.legend(loc="best", fontsize=9)

    out = _output_path("landing_zones.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[儲存] {out}")


def save_stats(df):
    os.makedirs(OUT_DIR, exist_ok=True)
    detail_path = _output_path("landing_detail.csv")
    df.to_csv(detail_path, index=False, encoding="utf-8-sig")
    print(f"[儲存] {detail_path}")

    stats = (
        df[df["in_table"]]
        .groupby("zone_label")
        .size()
        .reset_index(name="count")
        .sort_values("zone_label")
    )
    if CURRENT_BASE_NAME:
        stats.insert(0, "video_name", CURRENT_BASE_NAME)

    stats_path = _output_path("zone_stats.csv")
    stats.to_csv(stats_path, index=False, encoding="utf-8-sig")
    print(f"[儲存] {stats_path}")
