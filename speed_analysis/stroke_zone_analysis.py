import argparse
import math
import os
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd

import bounce_landing_analysis as landing
from stroke_analysis import (
    build_frame_to_stroke_map,
    calc_step,
    detect_strokes_from_runs,
    draw_polygon,
    ensure_dir,
    extract_zone_points,
    is_valid_point,
    safe_int,
)
from table_tracker import build_net_front_zone, build_table_from_lines, detect_table


TABLE_W = 274.0
TABLE_H = 152.5
MAX_SPEED_KMH = 115.0

TABLE_OUT_COLS = [
    "table_p1_x", "table_p1_y",
    "table_p2_x", "table_p2_y",
    "table_p3_x", "table_p3_y",
    "table_p4_x", "table_p4_y",
]
NET_OUT_COLS = [
    "net_p1_x", "net_p1_y",
    "net_p2_x", "net_p2_y",
    "net_p3_x", "net_p3_y",
    "net_p4_x", "net_p4_y",
    "net_p5_x", "net_p5_y",
    "net_p6_x", "net_p6_y",
]
ZONE_OUT_COLS = TABLE_OUT_COLS + NET_OUT_COLS


class FrameReader:
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if self.fps <= 0:
            self.fps = 120.0

    def read_frame(self, frame_id: int):
        if frame_id < 0 or frame_id >= self.total_frames:
            return None
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_id))
        ok, frame = self.cap.read()
        return frame if ok and frame is not None else None

    def release(self):
        self.cap.release()


class CsvFrameInfo:
    def __init__(self, df: pd.DataFrame, fps: float = 120.0, frame_w: int = 1920, frame_h: int = 1080):
        self.video_path = None
        self.cap = None
        self.total_frames = int(df["Frame"].max()) + 1 if "Frame" in df.columns and len(df) > 0 else 0
        self.fps = float(fps)
        self.width = int(frame_w)
        self.height = int(frame_h)

    def read_frame(self, frame_id: int):
        return None

    def release(self):
        pass


def collect_predict_videos(video_root: str) -> List[str]:
    video_files = []
    for root, _, files in os.walk(video_root):
        for fname in files:
            if fname.lower().endswith(".mp4") and fname.endswith("_predict.mp4"):
                video_files.append(os.path.join(root, fname))
    return sorted(video_files)


def find_ball_csv_for_video(video_path: str, csv_suffixes=("_ball.csv", "_bass.csv")):
    video_name = os.path.basename(video_path)
    if not video_name.endswith("_predict.mp4"):
        return None

    stem = video_name[:-len("_predict.mp4")]
    video_dir = os.path.dirname(video_path)

    for suffix in csv_suffixes:
        csv_path = os.path.join(video_dir, f"{stem}{suffix}")
        if os.path.exists(csv_path):
            return csv_path
    return None


def collect_ball_csvs(root_dir: str, csv_suffixes=("_ball.csv", "_bass.csv")) -> List[str]:
    csv_files = []
    for root, _, files in os.walk(root_dir):
        for fname in files:
            if any(fname.endswith(suffix) for suffix in csv_suffixes):
                csv_files.append(os.path.join(root, fname))
    return sorted(csv_files)


def strip_csv_suffix(csv_path: str, csv_suffixes=("_ball.csv", "_bass.csv")) -> str:
    name = os.path.basename(csv_path)
    for suffix in csv_suffixes:
        if name.endswith(suffix):
            return name[:-len(suffix)]
    return os.path.splitext(name)[0]


def find_video_for_ball_csv(ball_csv: str, video_root: Optional[str] = None, csv_suffixes=("_ball.csv", "_bass.csv")) -> Optional[str]:
    stem = strip_csv_suffix(ball_csv, csv_suffixes)

    # Prefer original input video. The analysis CSV is usually under save_root,
    # while the original mp4 stays under video_root.
    search_dirs = []
    if video_root is not None:
        search_dirs.append(video_root)
    search_dirs.append(os.path.dirname(ball_csv))

    for search_dir in search_dirs:
        if search_dir is None or not os.path.exists(search_dir):
            continue

        direct_candidates = [
            os.path.join(search_dir, f"{stem}.mp4"),
            os.path.join(search_dir, f"{stem}.MP4"),
            os.path.join(search_dir, f"{stem}_predict.mp4"),
        ]
        for video_path in direct_candidates:
            if os.path.exists(video_path):
                return video_path

        for root, _, files in os.walk(search_dir):
            for fname in files:
                lower = fname.lower()
                if not lower.endswith(".mp4"):
                    continue
                name_no_ext = os.path.splitext(fname)[0]
                if name_no_ext == stem or name_no_ext == f"{stem}_predict":
                    return os.path.join(root, fname)

    return None


def polygon_to_result_dict(table_corners, net_zone) -> Dict:
    result = {col: "" for col in ZONE_OUT_COLS}

    if table_corners is not None:
        for i in range(4):
            result[f"table_p{i + 1}_x"] = float(table_corners[i, 0])
            result[f"table_p{i + 1}_y"] = float(table_corners[i, 1])

    if net_zone is not None:
        for i in range(6):
            result[f"net_p{i + 1}_x"] = float(net_zone[i, 0])
            result[f"net_p{i + 1}_y"] = float(net_zone[i, 1])

    return result


def detect_geometry_from_frame(frame, up_px: int = 140, left_shift_px: int = 160):
    horiz_lines, net_lines, edge_lines = detect_table(frame, debug=False)
    corners, net_line, top_line, bottom_line = build_table_from_lines(
        horiz_lines=horiz_lines,
        edge_lines=edge_lines,
        net_lines=net_lines,
        frame=frame,
        debug=False,
    )
    net_zone = build_net_front_zone(net_line, top_line, bottom_line, up_px=up_px, left_shift_px=left_shift_px)
    return {"corners": corners, "net_zone": net_zone, "valid": corners is not None and net_zone is not None}


def choose_anchor_frame(stroke: Dict) -> int:
    hit_frame = stroke.get("hit_frame")
    if hit_frame not in [None, ""]:
        return int(hit_frame)
    return int(round((int(stroke["frame_start"]) + int(stroke["frame_end"])) / 2.0))


def build_candidate_frames(stroke: Dict, window: int = 2) -> List[int]:
    frame_start = int(stroke["frame_start"])
    frame_end = int(stroke["frame_end"])
    anchor = choose_anchor_frame(stroke)

    candidates = []
    for d in range(window + 1):
        if d == 0 and frame_start <= anchor <= frame_end:
            candidates.append(anchor)
        elif d > 0:
            for frame_id in (anchor - d, anchor + d):
                if frame_start <= frame_id <= frame_end:
                    candidates.append(frame_id)

    return list(dict.fromkeys(candidates))


def average_geometries(geometry_list: List[Dict]) -> Optional[Dict]:
    valid_items = [g for g in geometry_list if g.get("valid")]
    if not valid_items:
        return None

    corner_stack = np.stack([g["corners"] for g in valid_items], axis=0).astype(np.float32)
    net_zone_stack = np.stack([g["net_zone"] for g in valid_items], axis=0).astype(np.float32)
    return {"corners": np.mean(corner_stack, axis=0), "net_zone": np.mean(net_zone_stack, axis=0)}


def select_zone_for_stroke_from_video(
    stroke: Dict,
    frame_reader,
    geometry_cache: Dict[int, Dict],
    window: int = 2,
    up_px: int = 140,
    left_shift_px: int = 160,
) -> Dict:
    geometries = []

    for frame_id in build_candidate_frames(stroke, window=window):
        if frame_id not in geometry_cache:
            frame = frame_reader.read_frame(frame_id)
            geometry_cache[frame_id] = {"valid": False} if frame is None else detect_geometry_from_frame(frame, up_px, left_shift_px)
        geometries.append(geometry_cache[frame_id])

    avg_geom = average_geometries(geometries)
    if avg_geom is None:
        return {col: "" for col in ZONE_OUT_COLS}
    return polygon_to_result_dict(avg_geom["corners"], avg_geom["net_zone"])


def point_in_net_zone(px: float, py: float, net_zone_points) -> bool:
    if net_zone_points is None:
        return False
    poly = np.asarray(net_zone_points, dtype=np.float32)
    if poly.shape != (6, 2):
        return False
    return cv2.pointPolygonTest(poly, (float(px), float(py)), False) >= 0


def compute_fixed_scales(table_corners):
    """Compute fixed cm-per-pixel scales from table corners ordered as LT, RT, RB, LB."""
    p1, p2, p3, p4 = np.asarray(table_corners, dtype=np.float32)

    top_sx = TABLE_W / calc_step(p1[0], p1[1], p2[0], p2[1])
    bottom_sx = TABLE_W / calc_step(p4[0], p4[1], p3[0], p3[1])
    left_sy = TABLE_H / calc_step(p1[0], p1[1], p4[0], p4[1])
    right_sy = TABLE_H / calc_step(p2[0], p2[1], p3[0], p3[1])

    return float(max(top_sx, bottom_sx)), float(max(left_sy, right_sy))


def calc_segment_speed_basic_kmh(x1, y1, x2, y2, fps, sx, sy, dt_frames):
    # Use a conservative mixed scale instead of the pure max scale.
    scale = sx + 0.25 * (sy - sx)
    dx_cm = (x2 - x1) * scale
    dy_cm = (y2 - y1) * scale
    v_cm_s = math.hypot(dx_cm, dy_cm) / (dt_frames / fps)
    return float(v_cm_s * 0.036)


def make_speed_segment(df, i, j, fps, sx, sy, dt_frames, speed_end_frame, run_start_idx, run_end_idx):
    if i < run_start_idx or j > run_end_idx:
        return None

    r1 = df.iloc[i]
    r2 = df.iloc[j]
    f1 = int(r1["Frame"])
    f2 = int(r2["Frame"])

    if f2 - f1 != dt_frames or f2 > speed_end_frame:
        return None
    if not is_valid_point(r1) or not is_valid_point(r2):
        return None

    x1 = float(r1["X"])
    y1 = float(r1["Y"])
    x2 = float(r2["X"])
    y2 = float(r2["Y"])
    if min(x1, y1, x2, y2) <= 0:
        return None

    speed = calc_segment_speed_basic_kmh(x1, y1, x2, y2, fps, sx, sy, dt_frames)
    if not np.isfinite(speed) or speed > MAX_SPEED_KMH:
        return None

    return float(speed), f1, f2


def compute_speed_metrics_for_stroke(df: pd.DataFrame, stroke: Dict, fps: float, table_corners, net_zone_points):
    hit_frame = stroke.get("hit_frame")
    if hit_frame in [None, ""] or table_corners is None:
        return None

    hit_frame = int(hit_frame)
    bounce_frame = stroke.get("bounce_frame", 0)
    frame_end = int(stroke.get("frame_end", 0))
    speed_end_frame = int(bounce_frame) if bounce_frame not in [None, 0, ""] else frame_end
    run_start_idx = int(stroke["run_start_idx"])
    run_end_idx = int(stroke["run_end_idx"])
    sx, sy = compute_fixed_scales(table_corners)

    all_segments = []

    for i in range(run_start_idx, run_end_idx + 1):
        row = df.iloc[i]
        frame_id = int(row["Frame"])

        if frame_id < hit_frame or frame_id > speed_end_frame or not is_valid_point(row):
            continue

        x = float(row["X"])
        y = float(row["Y"])
        if min(x, y) <= 0:
            continue

        speed_candidates = []
        for speed_type, start_i, end_i, dt_frames in (
            ("1f", i - 1, i, 1),
            ("2f", i, i + 2, 2),
            ("c2f", i - 1, i + 1, 2),
        ):
            seg = make_speed_segment(df, start_i, end_i, fps, sx, sy, dt_frames, speed_end_frame, run_start_idx, run_end_idx)
            if seg is not None and seg[1] >= hit_frame:
                speed_candidates.append((speed_type, *seg))

        if not speed_candidates:
            continue

        speed_map = {speed_type: speed for speed_type, speed, _, _ in speed_candidates}
        best_type, best_speed, best_start, best_end = max(speed_candidates, key=lambda item: item[1])

        all_segments.append({
            "frame": frame_id,
            "best_type": best_type,
            "best_speed": float(best_speed),
            "best_start": int(best_start),
            "best_end": int(best_end),
            "speed_1f": speed_map.get("1f"),
            "speed_2f": speed_map.get("2f"),
            "speed_c2f": speed_map.get("c2f"),
            "in_net": point_in_net_zone(x, y, net_zone_points),
        })

    if not all_segments:
        return None

    best_speeds = [seg["best_speed"] for seg in all_segments]

    expanded_net_idx = set()
    for idx, seg in enumerate(all_segments):
        if seg["in_net"]:
            expanded_net_idx.update(j for j in (idx - 1, idx, idx + 1) if 0 <= j < len(all_segments))

    net_segments = [all_segments[j] for j in sorted(expanded_net_idx)]
    best_net = max(net_segments, key=lambda seg: seg["best_speed"]) if net_segments else None

    return {
        "avg_speed_kmh": float(np.mean(best_speeds)),
        "max_speed_kmh": float(np.max(best_speeds)),
        "net_zone_max_speed_kmh": best_net["best_speed"] if best_net else None,
        "net_zone_max_speed_type": best_net["best_type"] if best_net else "",
        "net_zone_max_speed_start_frame": best_net["best_start"] if best_net else None,
        "net_zone_max_speed_end_frame": best_net["best_end"] if best_net else None,
        "net_zone_max_speed_1f_kmh": max((seg["speed_1f"] for seg in net_segments if seg["speed_1f"] is not None), default=None),
        "net_zone_max_speed_2f_kmh": max((seg["speed_2f"] for seg in net_segments if seg["speed_2f"] is not None), default=None),
        "net_zone_max_speed_c2f_kmh": max((seg["speed_c2f"] for seg in net_segments if seg["speed_c2f"] is not None), default=None),
        "sx_cm_per_px": sx,
        "sy_cm_per_px": sy,
    }


def zone_info_to_arrays(zone_info: Dict):
    table_corners = None
    net_zone_points = None

    try:
        if zone_info.get("table_p1_x", "") != "":
            table_corners = np.array([
                [zone_info["table_p1_x"], zone_info["table_p1_y"]],
                [zone_info["table_p2_x"], zone_info["table_p2_y"]],
                [zone_info["table_p3_x"], zone_info["table_p3_y"]],
                [zone_info["table_p4_x"], zone_info["table_p4_y"]],
            ], dtype=np.float32)

        if zone_info.get("net_p1_x", "") != "":
            net_zone_points = np.array([
                [zone_info[f"net_p{i}_x"], zone_info[f"net_p{i}_y"]] for i in range(1, 7)
            ], dtype=np.float32)
    except Exception:
        return None, None

    return table_corners, net_zone_points


def append_note(note: str, value: str) -> str:
    if value in str(note).split(";"):
        return note
    return value if note == "" else f"{note};{value}"


def update_net_note(df: pd.DataFrame, stroke: Dict, net_zone_points, note: str) -> str:
    if net_zone_points is None:
        return note

    bounce_frame = stroke.get("bounce_frame", 0)
    if bounce_frame not in [None, 0, ""]:
        bounce_row = df[df["Frame"] == int(bounce_frame)]
        if len(bounce_row) > 0 and int(bounce_row.iloc[0]["Visibility"]) == 1:
            bx = float(bounce_row.iloc[0]["X"])
            by = float(bounce_row.iloc[0]["Y"])
            if point_in_net_zone(bx, by, net_zone_points):
                note = append_note(note, "net_hit")

    end_row = df[df["Frame"] == int(stroke["frame_end"])]
    if len(end_row) > 0 and int(end_row.iloc[0]["Visibility"]) == 1:
        ex = float(end_row.iloc[0]["X"])
        ey = float(end_row.iloc[0]["Y"])
        if point_in_net_zone(ex, ey, net_zone_points):
            note = append_note(note, "net_stop")

    return note


def value_or_blank(value):
    return value if value is not None else ""


def build_stroke_summary_csv(
    df: pd.DataFrame,
    strokes: List[Dict],
    fps: float,
    frame_reader,
    zone_window: int = 2,
    up_px: int = 140,
    left_shift_px: int = 160,
) -> pd.DataFrame:
    rows = []
    geometry_cache: Dict[int, Dict] = {}

    for stroke in strokes:
        zone_info = select_zone_for_stroke_from_video(stroke, frame_reader, geometry_cache, zone_window, up_px, left_shift_px)
        table_corners, net_zone_points = zone_info_to_arrays(zone_info)

        valid = stroke["valid"]
        note = stroke["note"]
        speed_metrics = None

        if stroke.get("hit_frame") is not None and table_corners is not None and fps is not None and fps > 0:
            speed_metrics = compute_speed_metrics_for_stroke(df, stroke, fps, table_corners, net_zone_points)
            if speed_metrics is None:
                valid = 0
                note = "hit_found_but_no_clean_speed_segment"
        elif stroke.get("hit_frame") is not None and table_corners is None:
            note = append_note(note, "no_video_or_table_geometry")

        note = update_net_note(df, stroke, net_zone_points, note)

        row_out = {
            "stroke_id": stroke["stroke_id"],
            "frame_start": safe_int(stroke["frame_start"]),
            "hit_frame": safe_int(stroke["hit_frame"]),
            "frame_end": safe_int(stroke["frame_end"]),
            "bounce_frame": int(stroke.get("bounce_frame", 0) or 0),
            "avg_speed_kmh": value_or_blank(speed_metrics["avg_speed_kmh"] if speed_metrics else None),
            "max_speed_kmh": value_or_blank(speed_metrics["max_speed_kmh"] if speed_metrics else None),
            "net_zone_max_speed_kmh": value_or_blank(speed_metrics["net_zone_max_speed_kmh"] if speed_metrics else None),
            "net_zone_max_speed_1f_kmh": value_or_blank(speed_metrics["net_zone_max_speed_1f_kmh"] if speed_metrics else None),
            "net_zone_max_speed_2f_kmh": value_or_blank(speed_metrics["net_zone_max_speed_2f_kmh"] if speed_metrics else None),
            "net_zone_max_speed_c2f_kmh": value_or_blank(speed_metrics["net_zone_max_speed_c2f_kmh"] if speed_metrics else None),
            "net_zone_max_speed_type": speed_metrics["net_zone_max_speed_type"] if speed_metrics else "",
            "net_zone_max_speed_start_frame": value_or_blank(speed_metrics["net_zone_max_speed_start_frame"] if speed_metrics else None),
            "net_zone_max_speed_end_frame": value_or_blank(speed_metrics["net_zone_max_speed_end_frame"] if speed_metrics else None),
            "sx_cm_per_px": value_or_blank(speed_metrics["sx_cm_per_px"] if speed_metrics else None),
            "sy_cm_per_px": value_or_blank(speed_metrics["sy_cm_per_px"] if speed_metrics else None),
            "valid": valid,
            "note": note,
        }
        row_out.update(zone_info)
        rows.append(row_out)

    return pd.DataFrame(rows)


def draw_visual_video(video_path: str, df: pd.DataFrame, strokes: List[Dict], summary_df: pd.DataFrame, output_video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not out.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open output video: {output_video_path}")

    frame_to_stroke = build_frame_to_stroke_map(strokes)
    summary_map = {int(row["stroke_id"]): row for _, row in summary_df.iterrows()}
    frame_to_row = {int(df.iloc[idx]["Frame"]): idx for idx in range(len(df))}

    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id in frame_to_stroke and frame_id in frame_to_row:
            stroke = frame_to_stroke[frame_id]
            row = df.iloc[frame_to_row[frame_id]]
            if int(row["Visibility"]) == 1:
                draw_stroke_overlay(frame, df, frame_to_row, frame_id, stroke, summary_map)

        out.write(frame)
        frame_id += 1

    cap.release()
    out.release()


def draw_stroke_overlay(frame, df, frame_to_row, frame_id: int, stroke: Dict, summary_map: Dict):
    x = int(df.iloc[frame_to_row[frame_id]]["X"])
    y = int(df.iloc[frame_to_row[frame_id]]["Y"])
    color = (0, 255, 0) if int(stroke["valid"]) == 1 else (0, 0, 255)

    cv2.circle(frame, (x, y), 6, color, -1)

    sid = int(stroke["stroke_id"])
    f_start = int(stroke["frame_start"])
    f_end = int(stroke["frame_end"])
    hit_frame = stroke["hit_frame"]
    bounce_frame = stroke.get("bounce_frame", 0)

    points = []
    for f in range(max(f_start, frame_id - 25), min(f_end, frame_id) + 1):
        if f not in frame_to_row:
            continue
        r = df.iloc[frame_to_row[f]]
        if int(r["Visibility"]) == 1:
            points.append((int(r["X"]), int(r["Y"])))

    for p in points:
        cv2.circle(frame, p, 3, color, -1)
    for k in range(1, len(points)):
        cv2.line(frame, points[k - 1], points[k], color, 2)

    text = f"stroke={sid} start={f_start} hit={hit_frame if hit_frame is not None else 'NA'} end={f_end} bounce={bounce_frame} valid={stroke['valid']}"
    cv2.putText(frame, text, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

    summary_row = summary_map.get(sid)
    if summary_row is not None:
        table_pts = extract_zone_points(summary_row, "table", 4)
        net_pts = extract_zone_points(summary_row, "net", 6)
        if table_pts is not None:
            draw_polygon(frame, table_pts, (255, 0, 0), thickness=2, fill=False)
        if net_pts is not None:
            draw_polygon(frame, net_pts, (0, 255, 255), thickness=2, fill=True, alpha=0.18)

    if frame_id == f_start:
        cv2.putText(frame, "START", (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
    if hit_frame is not None and frame_id == int(hit_frame):
        cv2.putText(frame, "HIT", (x + 10, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    if bounce_frame and frame_id == int(bounce_frame):
        cv2.putText(frame, "BOUNCE", (x + 10, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2, cv2.LINE_AA)
    if frame_id == f_end:
        cv2.putText(frame, "END", (x + 10, y + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2, cv2.LINE_AA)


def build_export_stroke_csv(summary_with_landing_df: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [
        "stroke_id", "frame_start", "hit_frame", "frame_end", "bounce_frame",
        "avg_speed_kmh", "max_speed_kmh", "net_zone_max_speed_kmh",
        "valid", "note", "zone_label", "in_table",
    ]
    return keep_columns(summary_with_landing_df, keep_cols)


def build_export_zone_detail_csv(summary_df_full: pd.DataFrame) -> pd.DataFrame:
    return keep_columns(summary_df_full, ["stroke_id"] + ZONE_OUT_COLS)


def build_export_speed_compare_csv(summary_df_full: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [
        "stroke_id", "frame_start", "hit_frame", "frame_end", "bounce_frame", "valid", "note",
        "avg_speed_kmh", "max_speed_kmh", "net_zone_max_speed_kmh",
        "net_zone_max_speed_type", "net_zone_max_speed_start_frame", "net_zone_max_speed_end_frame",
        "net_zone_max_speed_1f_kmh", "net_zone_max_speed_2f_kmh", "net_zone_max_speed_c2f_kmh",
        "sx_cm_per_px", "sy_cm_per_px",
    ]
    return keep_columns(summary_df_full, keep_cols)


def keep_columns(df: pd.DataFrame, keep_cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for col in keep_cols:
        if col not in out.columns:
            out[col] = ""
    return out[keep_cols]


def has_valid_zone_geometry(summary_df_full: pd.DataFrame) -> bool:
    if summary_df_full.empty:
        return False

    required_cols = TABLE_OUT_COLS + NET_OUT_COLS
    for col in required_cols:
        if col not in summary_df_full.columns:
            return False

    zone_df = summary_df_full[required_cols].replace("", np.nan)
    return zone_df.notna().any(axis=1).any()


def run_landing_analysis_with_module(summary_df_full: pd.DataFrame, traj_df: pd.DataFrame, save_dir: str) -> pd.DataFrame:
    if not has_valid_zone_geometry(summary_df_full):
        print("[WARN] no valid table/net geometry, skip landing analysis.")
        return pd.DataFrame()

    landing.OUT_DIR = save_dir
    safe_summary = summary_df_full.copy()
    safe_summary[ZONE_OUT_COLS] = safe_summary[ZONE_OUT_COLS].replace("", np.nan)
    df_land = landing.compute_landings(safe_summary, traj_df)

    if not df_land.empty:
        landing.save_stats(df_land)
        landing.plot_heatmap(df_land)
        landing.plot_scatter(df_land)

    return df_land


def process_single_video(
    video_file,
    ball_csv,
    save_dir,
    min_left_segments=5,
    min_candidate_frames=50,
    max_step_th=130.0,
    max_abs_dy_th=45.0,
    left_half_ratio=0.5,
    right_side_ratio=0.6,
    zone_window=2,
    up_px=140,
    left_shift_px=160,
    fps=120.0,
    frame_w=1920,
    frame_h=1080,
    save_video=False,
):
    ensure_dir(save_dir)
    df = pd.read_csv(ball_csv)

    for col in ["Frame", "Visibility", "X", "Y"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    has_video = video_file is not None and os.path.exists(video_file)
    frame_reader = FrameReader(video_file) if has_video else CsvFrameInfo(df, fps=fps, frame_w=frame_w, frame_h=frame_h)
    if not has_video:
        print(f"[WARN] no original/predict mp4 found for csv, run csv-only mode: {ball_csv}")

    try:
        strokes = detect_strokes_from_runs(
            df=df,
            frame_w=frame_reader.width,
            min_left_segments=min_left_segments,
            min_candidate_frames=min_candidate_frames,
            max_step_th=max_step_th,
            max_abs_dy_th=max_abs_dy_th,
            left_half_ratio=left_half_ratio,
            right_side_ratio=right_side_ratio,
        )

        summary_df_full = build_stroke_summary_csv(
            df=df,
            strokes=strokes,
            fps=frame_reader.fps,
            frame_reader=frame_reader,
            zone_window=zone_window,
            up_px=up_px,
            left_shift_px=left_shift_px,
        )

        df_land = run_landing_analysis_with_module(summary_df_full, df, save_dir)
        if not df_land.empty:
            summary_df_full = summary_df_full.merge(
                df_land[["stroke_id", "in_table", "zone_label", "zone_col", "zone_row"]],
                on="stroke_id",
                how="left",
            )

        base = os.path.splitext(os.path.basename(video_file))[0] if has_video else strip_csv_suffix(ball_csv)
        csv_path = os.path.join(save_dir, f"{base}_stroke_zone.csv")
        zone_detail_csv_path = os.path.join(save_dir, f"{base}_zone_detail.csv")
        speed_compare_csv_path = os.path.join(save_dir, f"{base}_stroke_speed_compare.csv")
        video_path = os.path.join(save_dir, f"{base}_stroke_zone_visualize.mp4")

        build_export_stroke_csv(summary_df_full).to_csv(csv_path, index=False, encoding="utf-8-sig")
        build_export_zone_detail_csv(summary_df_full).to_csv(zone_detail_csv_path, index=False, encoding="utf-8-sig")
        build_export_speed_compare_csv(summary_df_full).to_csv(speed_compare_csv_path, index=False, encoding="utf-8-sig")
        if save_video and has_video:
            draw_visual_video(video_file, df, strokes, summary_df_full, video_path)
        elif save_video and not has_video:
            print("[WARN] --save_video was set, but no mp4 was found. Skip visual video output.")
    finally:
        frame_reader.release()

    print(f"saved csv   : {csv_path}")
    print(f"saved zone  : {zone_detail_csv_path}")
    print(f"saved speed : {speed_compare_csv_path}")
    if save_video and has_video:
        print(f"saved video : {video_path}")
    elif save_video and not has_video:
        print("saved video : skipped (no matching mp4)")
    else:
        print("saved video : skipped (--save_video not set)")
    print(f"num strokes : {len(summary_df_full)}")


def process_video_root(
    video_root,
    save_root=None,
    csv_suffixes=("_ball.csv", "_bass.csv"),
    min_left_segments=5,
    min_candidate_frames=50,
    max_step_th=130.0,
    max_abs_dy_th=45.0,
    left_half_ratio=0.5,
    right_side_ratio=0.5,
    zone_window=2,
    up_px=140,
    left_shift_px=160,
    fps=120.0,
    frame_w=1920,
    frame_h=1080,
    save_video=False,
):
    search_root = save_root if save_root is not None else video_root
    ball_csv_files = collect_ball_csvs(search_root, csv_suffixes=csv_suffixes)
    if not ball_csv_files:
        raise RuntimeError(f"No ball csv files found under: {search_root}")

    print(f"[INFO] video_root : {video_root}")
    print(f"[INFO] search_root: {search_root}")
    print(f"[INFO] found {len(ball_csv_files)} csv files under {search_root}")

    for i, ball_csv in enumerate(ball_csv_files, 1):
        video_file = find_video_for_ball_csv(ball_csv, video_root=video_root, csv_suffixes=csv_suffixes)
        save_dir = os.path.dirname(ball_csv)
        print("=" * 80)
        print(f"[BATCH] ({i}/{len(ball_csv_files)})")
        print(f"[BATCH] video    : {video_file if video_file else 'None (csv-only)'}")
        print(f"[BATCH] ball csv : {ball_csv}")
        print(f"[BATCH] save dir : {save_dir}")

        try:
            process_single_video(
                video_file=video_file,
                ball_csv=ball_csv,
                save_dir=save_dir,
                min_left_segments=min_left_segments,
                min_candidate_frames=min_candidate_frames,
                max_step_th=max_step_th,
                max_abs_dy_th=max_abs_dy_th,
                left_half_ratio=left_half_ratio,
                right_side_ratio=right_side_ratio,
                zone_window=zone_window,
                up_px=up_px,
                left_shift_px=left_shift_px,
                fps=fps,
                frame_w=frame_w,
                frame_h=frame_h,
                save_video=save_video,
            )
        except Exception as e:
            print(f"[ERROR] failed on {video_file}")
            print(f"[ERROR] {e}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_file", type=str, default=None)
    parser.add_argument("--ball_csv", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--video_root", type=str, default=None)
    parser.add_argument("--save_root", type=str, default=None)
    parser.add_argument("--save_video", action="store_true")
    parser.add_argument("--csv_suffixes", type=str, nargs="+", default=["_ball.csv", "_bass.csv"])
    parser.add_argument("--min_left_segments", type=int, default=5)
    parser.add_argument("--min_candidate_frames", type=int, default=50)
    parser.add_argument("--max_step_th", type=float, default=130.0)
    parser.add_argument("--max_abs_dy_th", type=float, default=45.0)
    parser.add_argument("--left_half_ratio", type=float, default=0.5)
    parser.add_argument("--right_side_ratio", type=float, default=0.5)
    parser.add_argument("--zone_window", type=int, default=2)
    parser.add_argument("--up_px", type=int, default=140)
    parser.add_argument("--left_shift_px", type=int, default=160)
    parser.add_argument("--fps", type=float, default=120.0)
    parser.add_argument("--frame_w", type=int, default=1920)
    parser.add_argument("--frame_h", type=int, default=1080)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.video_root is not None:
        process_video_root(
            video_root=args.video_root,
            save_root=args.save_root,
            csv_suffixes=tuple(args.csv_suffixes),
            min_left_segments=args.min_left_segments,
            min_candidate_frames=args.min_candidate_frames,
            max_step_th=args.max_step_th,
            max_abs_dy_th=args.max_abs_dy_th,
            left_half_ratio=args.left_half_ratio,
            right_side_ratio=args.right_side_ratio,
            zone_window=args.zone_window,
            up_px=args.up_px,
            left_shift_px=args.left_shift_px,
            fps=args.fps,
            frame_w=args.frame_w,
            frame_h=args.frame_h,
            save_video=args.save_video,
        )
        return

    if args.video_file is not None or args.ball_csv is not None:
        if args.ball_csv is None or args.save_dir is None:
            raise ValueError("single mode requires --ball_csv --save_dir. --video_file is optional.")

        process_single_video(
            video_file=args.video_file,
            ball_csv=args.ball_csv,
            save_dir=args.save_dir,
            min_left_segments=args.min_left_segments,
            min_candidate_frames=args.min_candidate_frames,
            max_step_th=args.max_step_th,
            max_abs_dy_th=args.max_abs_dy_th,
            left_half_ratio=args.left_half_ratio,
            right_side_ratio=args.right_side_ratio,
            zone_window=args.zone_window,
            up_px=args.up_px,
            left_shift_px=args.left_shift_px,
            fps=args.fps,
            frame_w=args.frame_w,
            frame_h=args.frame_h,
            save_video=args.save_video,
        )
        return

    raise ValueError("Please provide either --video_root or --ball_csv")


if __name__ == "__main__":
    main()
