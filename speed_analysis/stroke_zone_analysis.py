"""Stroke-zone analysis entry point.

This file detects strokes from a TrackNet ball CSV, computes net-zone speed,
merges landing/bounce results, and optionally writes a visualized MP4.
"""

import argparse
import math
import os
import subprocess
from typing import Dict, List, Optional

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

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
from helper_table import BOX_EDGES, BOX_HEIGHT, NEAR_NET_DIST, NearNetRegion, load_table_corners


TABLE_W = 274.0
TABLE_H = 152.5
MAX_SPEED_KMH = 130.0
SPEED_GT_SCALE_FACTOR = 0.75

# Height correction before local table-width scale lookup.
# Positive value means the observed ball center is moved toward the near side
# before using its y position to find the corresponding table width.
BALL_HEIGHT_Y_OFFSET_PX = 110.0

# Net-zone speed selection.
# Directly choose the largest valid net-zone segment speed.
ROBUST_NET_TOP_N = 1

# Draw the speed segment and local table-width lookup line in visualize video.
DRAW_SPEED_SCALE_DEBUG = True

# Method 2: estimate a rough camera model from the table corners, then compare
# the same image segment on z=0 and z=h parallel planes. The resulting ratio
# is used to scale the current y-based speed, instead of directly guessing a
# global multiplier.
DEFAULT_PLANE_HEIGHT_CM = 26.0
DEFAULT_CAMERA_FOCAL_SCALE = 1.0
DEFAULT_PLANE_DEBUG_MAX_HEIGHT_CM = 50.0

TABLE_WORLD_CORNERS = np.array([
    [0.0, 0.0, 0.0],
    [TABLE_W, 0.0, 0.0],
    [TABLE_W, TABLE_H, 0.0],
    [0.0, TABLE_H, 0.0],
], dtype=np.float32)

# Rectified orange-plane debug image scale.
# 4 px/cm -> 274 cm x 152.5 cm becomes about 1096 x 610 px,
# which is large enough for visual checking but not too heavy to save.
ORANGE_RECT_SCALE_PX_PER_CM = 4.0
ORANGE_SY_PROBE_PX = 20.0

class FFmpegWriter:
    """
    FFmpeg subprocess writer, compatible with cv2.VideoWriter-style write/release.
    Input frame must be BGR uint8 ndarray. Output is mp4.

    codec:
        h264_nvenc: NVIDIA NVENC hardware encoder
        libx264   : CPU encoder fallback
    """
    def __init__(self, save_file, width, height, fps, codec="h264_nvenc", preset=None, cq=23):
        self.save_file = save_file
        self.width = int(width)
        self.height = int(height)
        self.closed = False

        common_in = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{self.width}x{self.height}",
            "-r", f"{fps}",
            "-i", "-",
            "-an",
        ]
        common_out = [
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            save_file,
        ]

        if codec == "h264_nvenc":
            preset = preset or "p4"
            enc = ["-c:v", "h264_nvenc", "-preset", preset, "-cq", str(cq)]
        elif codec == "libx264":
            preset = preset or "veryfast"
            enc = ["-c:v", "libx264", "-preset", preset, "-crf", str(cq)]
        else:
            raise ValueError(f"Unsupported codec: {codec}")

        cmd = common_in + enc + common_out
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    def write(self, frame_bgr):
        if frame_bgr.shape[0] != self.height or frame_bgr.shape[1] != self.width:
            frame_bgr = cv2.resize(frame_bgr, (self.width, self.height))
        if frame_bgr.dtype != np.uint8:
            frame_bgr = frame_bgr.astype(np.uint8)
        if not frame_bgr.flags["C_CONTIGUOUS"]:
            frame_bgr = np.ascontiguousarray(frame_bgr)
        try:
            self.proc.stdin.write(frame_bgr.tobytes())
        except BrokenPipeError:
            err = self.proc.stderr.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"ffmpeg pipe broken:\n{err}")

    def release(self):
        if self.closed:
            return
        self.closed = True
        try:
            self.proc.stdin.close()
        except Exception:
            pass
        try:
            ret = self.proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            ret = -1
        if ret != 0:
            err = self.proc.stderr.read().decode("utf-8", errors="ignore")
            print(f"[FFmpegWriter] ffmpeg exit={ret}\n{err}")

    def isOpened(self):
        return not self.closed and self.proc.poll() is None


TABLE_OUT_COLS = [
    "table_p1_x", "table_p1_y",
    "table_p2_x", "table_p2_y",
    "table_p3_x", "table_p3_y",
    "table_p4_x", "table_p4_y",
]
NET_OUT_COLS = [
    # helper_table Near Box 原始 8 個投影點，順序完全保留 helper_table.get_box_vertices_2d():
    # 1~4: bottom face, 5~8: top face
    "net_p1_x", "net_p1_y",
    "net_p2_x", "net_p2_y",
    "net_p3_x", "net_p3_y",
    "net_p4_x", "net_p4_y",
    "net_p5_x", "net_p5_y",
    "net_p6_x", "net_p6_y",
    "net_p7_x", "net_p7_y",
    "net_p8_x", "net_p8_y",
]
ORANGE_OUT_COLS = [
    "orange_p1_x", "orange_p1_y",
    "orange_p2_x", "orange_p2_y",
    "orange_p3_x", "orange_p3_y",
    "orange_p4_x", "orange_p4_y",
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
        net_zone = np.asarray(net_zone, dtype=np.float32)
        max_net_points = min(len(net_zone), len(NET_OUT_COLS) // 2)
        for i in range(max_net_points):
            result[f"net_p{i + 1}_x"] = float(net_zone[i, 0])
            result[f"net_p{i + 1}_y"] = float(net_zone[i, 1])

    return result


def helper_corners_to_table_corners(corners_lf_rf_rb_lb):
    """
    helper_table 點選順序:
        LF -> RF -> RB -> LB
        左前 -> 右前 -> 右後 -> 左後

    table/world 座標定義:
        LB 左後 = (0, 0)
        RB 右後 = (274, 0)
        RF 右前 = (274, 152.5)
        LF 左前 = (0, 152.5)

    bounce_landing_analysis 需要:
        table_p1 = (0, 0)
        table_p2 = (274, 0)
        table_p3 = (274, 152.5)
        table_p4 = (0, 152.5)
    """
    lf, rf, rb, lb = corners_lf_rf_rb_lb
    return np.array([lb, rb, rf, lf], dtype=np.float32)


def helper_box_vertices_to_net_points(box_vertices_2d):
    """Return helper_table Near Box 8 projected vertices without resampling or truncation.

    This intentionally keeps the exact output of NearNetRegion.get_box_vertices_2d():
      0~3: bottom face, 4~7: top face.
    No polygon hull conversion, no 6-point sampling, no bounding-box approximation.
    """
    if box_vertices_2d is None:
        return None

    pts = np.asarray(box_vertices_2d, dtype=np.float32)
    if pts.ndim != 2 or pts.shape != (8, 2):
        raise ValueError(f"helper_table box vertices must have shape (8, 2), got {pts.shape}")
    return pts.copy()

def build_zone_info_from_helper_table(corners_lf_rf_rb_lb, frame_shape, near_dist=NEAR_NET_DIST, box_height=BOX_HEIGHT):
    """Build table_p* and raw helper_table Near Box net_p1~net_p8.

    Net zone points are NOT converted to hull, NOT resampled, and NOT compressed to 6 points.
    They are exactly NearNetRegion.get_box_vertices_2d() in helper_table order.
    """
    region = NearNetRegion(
        image_corners=corners_lf_rf_rb_lb,
        frame_shape=frame_shape,
        near_dist=near_dist,
        box_height=box_height,
    )
    table_corners = helper_corners_to_table_corners(corners_lf_rf_rb_lb)
    net_zone = helper_box_vertices_to_net_points(region.get_box_vertices_2d(as_int=False))
    return polygon_to_result_dict(table_corners, net_zone)


def point_in_net_zone(px: float, py: float, net_zone_points) -> bool:
    if net_zone_points is None:
        return False

    pts = np.asarray(net_zone_points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape != (8, 2):
        return False

    # 視覺上的 8 點框外輪廓，用固定六邊形：
    # N1 -> N4 -> N3 -> N7 -> N6 -> N5
    # 1-based: 1, 4, 3, 7, 6, 5
    # 0-based: 0, 3, 2, 6, 5, 4
    poly = pts[[0, 3, 2, 6, 5, 4]].astype(np.float32)

    return cv2.pointPolygonTest(poly, (float(px), float(py)), False) >= 0


def draw_helper_box(frame, box_pts, color=(0, 255, 255), thickness=2, fill_alpha=0.0):
    """Draw helper_table Near Box with the original 8 vertices and BOX_EDGES. No point conversion."""
    if box_pts is None:
        return
    pts = np.asarray(box_pts, dtype=np.float32)
    if pts.ndim != 2 or pts.shape != (8, 2):
        return

    pts_i = pts.astype(np.int32)
    if fill_alpha and fill_alpha > 0:
        overlay = frame.copy()
        # Fill only for readability; edges still use the original 8 helper_table points.
        hull = cv2.convexHull(pts_i)
        cv2.fillPoly(overlay, [hull], color)
        cv2.addWeighted(overlay, fill_alpha, frame, 1 - fill_alpha, 0, frame)

    for i, j in BOX_EDGES:
        cv2.line(frame, tuple(pts_i[i]), tuple(pts_i[j]), color, thickness, cv2.LINE_AA)

    for idx, p in enumerate(pts_i):
        cv2.circle(frame, tuple(p), 4, color, -1)
        cv2.putText(frame, f"N{idx + 1}", (int(p[0]) + 5, int(p[1]) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def build_approx_camera_matrix(frame_w, frame_h, focal_scale=DEFAULT_CAMERA_FOCAL_SCALE):
    focal = float(max(frame_w, frame_h)) * float(focal_scale)
    cx = float(frame_w) / 2.0
    cy = float(frame_h) / 2.0
    return np.array([[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def estimate_camera_model_from_table(table_corners, frame_w, frame_h, focal_scale=DEFAULT_CAMERA_FOCAL_SCALE):
    """Method 2: use a rough intrinsic matrix and solvePnP from table corners.

    This does NOT require an explicit camera calibration session. It is still an
    approximation, but the height ratio comes from geometry rather than an
    arbitrary speed multiplier.
    """
    img_pts = np.asarray(table_corners, dtype=np.float32).reshape(-1, 2)
    if img_pts.shape[0] < 4:
        return None

    obj_pts = TABLE_WORLD_CORNERS.astype(np.float32)
    K = build_approx_camera_matrix(frame_w, frame_h, focal_scale=focal_scale)
    dist = np.zeros((4, 1), dtype=np.float64)

    flags_try = [cv2.SOLVEPNP_IPPE, cv2.SOLVEPNP_ITERATIVE]
    for flags in flags_try:
        try:
            ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist, flags=flags)
            if ok:
                R, _ = cv2.Rodrigues(rvec)
                cam_center_world = -R.T @ tvec.reshape(3, 1)
                return {
                    'K': K,
                    'dist': dist,
                    'rvec': rvec,
                    'tvec': tvec,
                    'R': R,
                    'cam_center_world': cam_center_world.reshape(3),
                    'focal_scale': float(focal_scale),
                }
        except Exception:
            pass
    return None


def pixel_to_world_on_plane(px, py, camera_model, plane_z_cm=0.0):
    if camera_model is None:
        return None

    K = camera_model['K']
    R = camera_model['R']
    cam_center_world = camera_model['cam_center_world']

    ray_cam = np.linalg.inv(K) @ np.array([float(px), float(py), 1.0], dtype=np.float64)
    ray_world = R.T @ ray_cam

    if abs(ray_world[2]) < 1e-8:
        return None

    lam = (float(plane_z_cm) - float(cam_center_world[2])) / float(ray_world[2])
    if lam <= 0:
        return None

    pt_world = cam_center_world + lam * ray_world
    return pt_world.astype(np.float64)


def compute_plane_scale_ratio_for_segment(x1, y1, x2, y2, camera_model, plane_height_cm):
    effective_z = resolve_effective_plane_z(camera_model, plane_height_cm)

    p1_z0 = pixel_to_world_on_plane(x1, y1, camera_model, plane_z_cm=0.0)
    p2_z0 = pixel_to_world_on_plane(x2, y2, camera_model, plane_z_cm=0.0)
    p1_zh = pixel_to_world_on_plane(x1, y1, camera_model, plane_z_cm=effective_z)
    p2_zh = pixel_to_world_on_plane(x2, y2, camera_model, plane_z_cm=effective_z)

    if p1_z0 is None or p2_z0 is None or p1_zh is None or p2_zh is None:
        return None, None, None

    d0 = float(np.linalg.norm(p2_z0[:2] - p1_z0[:2]))
    dh = float(np.linalg.norm(p2_zh[:2] - p1_zh[:2]))
    if d0 < 1e-8 or not np.isfinite(d0) or not np.isfinite(dh):
        return None, d0, dh

    return float(dh / d0), d0, dh


def project_world_points(world_points, camera_model):
    if camera_model is None:
        return None
    pts, _ = cv2.projectPoints(
        np.asarray(world_points, dtype=np.float32),
        camera_model['rvec'],
        camera_model['tvec'],
        camera_model['K'],
        camera_model['dist'],
    )
    return pts.reshape(-1, 2)


def resolve_effective_plane_z(camera_model, plane_height_cm):
    """Return the signed z value that makes the raised plane move upward in image y.

    The user-facing plane_height_cm is a positive physical height above the table.
    Because solvePnP can choose either normal direction for the table plane, +z is
    not guaranteed to be visually upward. We project both +h and -h, then choose
    the one whose projected table polygon has a smaller mean image-y value.
    """
    if camera_model is None:
        return float(plane_height_cm)

    h = abs(float(plane_height_cm))
    if h < 1e-9:
        return 0.0

    base = project_world_points(TABLE_WORLD_CORNERS, camera_model)
    if base is None or base.shape[0] != 4:
        return h

    base_mean_y = float(np.mean(base[:, 1]))
    best_z = h
    best_mean_y = float('inf')

    for signed_z in (h, -h):
        raised_world = TABLE_WORLD_CORNERS.copy()
        raised_world[:, 2] = float(signed_z)
        raised = project_world_points(raised_world, camera_model)
        if raised is None or raised.shape[0] != 4:
            continue
        mean_y = float(np.mean(raised[:, 1]))
        if mean_y < best_mean_y:
            best_mean_y = mean_y
            best_z = float(signed_z)

    # In image coordinates, upward means y is smaller.
    # If neither candidate is above the original plane, still return the one with
    # smaller y so the debug image exposes the least-wrong direction.
    return float(best_z)


def get_display_plane_height_cm(plane_height_cm):
    return abs(float(plane_height_cm))


def save_depth_ratio_curve(save_path, speed_start_xy, speed_end_xy, camera_model, plane_height_cm, max_height_cm=DEFAULT_PLANE_DEBUG_MAX_HEIGHT_CM):
    if plt is None or camera_model is None:
        return

    x1, y1 = speed_start_xy
    x2, y2 = speed_end_xy
    display_height = get_display_plane_height_cm(plane_height_cm)
    heights = np.arange(0.0, float(max(display_height, max_height_cm)) + 0.001, 2.0)
    ratios = []
    for h in heights:
        ratio, d0, dh = compute_plane_scale_ratio_for_segment(x1, y1, x2, y2, camera_model, plane_height_cm=h)
        ratios.append(1.0 if h == 0 else (ratio if ratio is not None else np.nan))

    plt.figure(figsize=(8, 5))
    plt.plot(heights, ratios, marker='o', markersize=3)
    plt.axvline(float(display_height), linestyle='--')
    if display_height in heights:
        idx = list(heights).index(float(display_height))
        if np.isfinite(ratios[idx]):
            plt.scatter([display_height], [ratios[idx]], s=60)
    plt.xlabel('Plane height above table (cm)')
    plt.ylabel('Scale ratio = distance(z=h) / distance(z=0)')
    plt.title('Height-plane scale ratio for selected speed segment')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def save_height_debug_overlay(save_path, frame, table_corners, camera_model, plane_height_cm, speed_start_xy, speed_end_xy, table_speed, corrected_speed, ratio):
    if frame is None or camera_model is None:
        return

    canvas = frame.copy()
    base_poly = np.asarray(table_corners, dtype=np.int32).reshape(-1, 2)
    cv2.polylines(canvas, [base_poly], True, (255, 0, 0), 2, cv2.LINE_AA)

    effective_z = resolve_effective_plane_z(camera_model, plane_height_cm)
    display_height = get_display_plane_height_cm(plane_height_cm)

    raised_world = TABLE_WORLD_CORNERS.copy()
    raised_world[:, 2] = float(effective_z)
    raised_poly = project_world_points(raised_world, camera_model)
    if raised_poly is not None and raised_poly.shape[0] == 4:
        raised_poly_i = np.round(raised_poly).astype(np.int32)
        cv2.polylines(canvas, [raised_poly_i], True, (0, 165, 255), 2, cv2.LINE_AA)
        for p0, ph in zip(base_poly, raised_poly_i):
            cv2.line(canvas, tuple(p0), tuple(ph), (180, 180, 180), 1, cv2.LINE_AA)

    raw_x1, raw_y1 = speed_start_xy
    raw_x2, raw_y2 = speed_end_xy
    x1, y1 = int(round(raw_x1)), int(round(raw_y1))
    x2, y2 = int(round(raw_x2)), int(round(raw_y2))

    mid_x = (float(raw_x1) + float(raw_x2)) / 2.0
    mid_y = (float(raw_y1) + float(raw_y2)) / 2.0
    table_y = correct_ball_y_to_table_plane(
        ball_y=mid_y,
        table_corners=table_corners,
        y_offset_px=BALL_HEIGHT_Y_OFFSET_PX,
    )

    # Purple: observed ball midpoint -> corrected table-plane y used for scale lookup.
    p_mid = (int(round(mid_x)), int(round(mid_y)))
    p_table = (int(round(mid_x)), int(round(table_y)))
    cv2.circle(canvas, p_mid, 6, (255, 0, 255), -1)
    cv2.circle(canvas, p_table, 6, (255, 0, 255), 2)
    cv2.line(canvas, p_mid, p_table, (255, 0, 255), 2, cv2.LINE_AA)

    # Cyan: the local table-width line actually used on the blue table plane z=0.
    blue_width_px = None
    blue_width_line = compute_local_table_width_line(table_corners, table_y)
    if blue_width_line is not None:
        bx1, by1, bx2, by2, blue_width_px = blue_width_line
        p_b1 = (int(round(bx1)), int(round(by1)))
        p_b2 = (int(round(bx2)), int(round(by2)))
        cv2.line(canvas, p_b1, p_b2, (255, 255, 0), 4, cv2.LINE_AA)
        cv2.circle(canvas, p_b1, 7, (255, 255, 0), -1)
        cv2.circle(canvas, p_b2, 7, (255, 255, 0), -1)
        cv2.putText(canvas, f'blue table-width line: {blue_width_px:.1f}px',
                    (p_b1[0] + 10, p_b1[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2, cv2.LINE_AA)

    # Orange: raised-plane width line uses observed mid_y directly, without offset.
    # This makes the orange line pass through the speed-segment center height.
    orange_width_px = None
    orange_transform_info = None
    if raised_poly is not None and raised_poly.shape[0] == 4:
        orange_transform_info = compute_blue_orange_mixed_scale_info(table_corners, raised_poly, raw_x1, raw_y1, raw_x2, raw_y2, table_y)
        if orange_transform_info is not None:
            _, sy_table_for_debug = compute_y_based_scales(table_corners, table_y)
            if orange_transform_info.get('blue_depth_ratio') is not None:
                orange_transform_info['blue_sy_cm_per_px'] = float(sy_table_for_debug * orange_transform_info['blue_depth_ratio'])
            ox1, oy1, ox2, oy2, orange_width_px = orange_transform_info['orange_width_line']
            p_o1 = (int(round(ox1)), int(round(oy1)))
            p_o2 = (int(round(ox2)), int(round(oy2)))
            cv2.line(canvas, p_o1, p_o2, (0, 165, 255), 4, cv2.LINE_AA)
            cv2.circle(canvas, p_o1, 7, (0, 165, 255), -1)
            cv2.circle(canvas, p_o2, 7, (0, 165, 255), -1)
            cv2.putText(canvas, f'orange table-width line: {orange_width_px:.1f}px',
                        (p_o1[0] + 10, p_o1[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255), 2, cv2.LINE_AA)

            # Green short vertical probe: image-y step used to estimate sy
            # after rectifying the blue table plane.
            p_probe = (int(round(mid_x)), int(round(mid_y + ORANGE_SY_PROBE_PX)))
            cv2.line(canvas, p_mid, p_probe, (0, 180, 0), 3, cv2.LINE_AA)
            cv2.circle(canvas, p_probe, 6, (0, 180, 0), 2)
            cv2.putText(canvas, f'blue sy probe +{ORANGE_SY_PROBE_PX:.0f}px image-y',
                        (p_probe[0] + 10, p_probe[1] + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 180, 0), 2, cv2.LINE_AA)

            rect_debug_path = save_path.replace('_plane_debug.png', '_orange_rect_debug.png')
            if rect_debug_path == save_path:
                root, ext = os.path.splitext(save_path)
                rect_debug_path = f'{root}_orange_rect_debug{ext}'
            save_orange_rectified_debug(rect_debug_path, raised_poly, speed_start_xy, speed_end_xy)
            blue_rect_debug_path = save_path.replace('_plane_debug.png', '_blue_sy_rect_debug.png')
            if blue_rect_debug_path == save_path:
                root, ext = os.path.splitext(save_path)
                blue_rect_debug_path = f'{root}_blue_sy_rect_debug{ext}'
            save_blue_rectified_sy_debug(blue_rect_debug_path, table_corners, speed_start_xy, speed_end_xy)

    # Red: selected speed segment.
    cv2.line(canvas, (x1, y1), (x2, y2), (0, 0, 255), 4, cv2.LINE_AA)
    cv2.circle(canvas, (x1, y1), 6, (0, 0, 255), -1)
    cv2.circle(canvas, (x2, y2), 6, (0, 0, 255), -1)

    lines = [
        f'blue: table plane z=0',
        f'orange: raised plane height={display_height:.1f} cm, effective_z={effective_z:.1f}',
        f'table speed={table_speed:.1f} km/h',
        f'ratio={ratio:.4f}',
        f'corrected speed={corrected_speed:.1f} km/h',
        f'mid_y={mid_y:.1f} -> table_y={table_y:.1f}',
    ]
    if blue_width_px is not None:
        lines.append(f'blue width line={blue_width_px:.1f} px')
    if orange_width_px is not None:
        lines.append(f'orange width line={orange_width_px:.1f} px')
    if orange_transform_info is not None:
        final_sx_vis = (TABLE_W / orange_transform_info["blue_local_width_px"]) * orange_transform_info["orange_blue_ratio_x"]
        lines.append(f'final sx=blue sx*orange/blue={final_sx_vis:.4f} cm/img-px')
        lines.append(f'blue sy={orange_transform_info["blue_sy_cm_per_px"]:.4f} cm/img-px')
        lines.append(f'height ratio={orange_transform_info["blue_depth_ratio"]:.4f} = rect_h/orig_h')
        lines.append(f'orig_h={orange_transform_info["blue_original_table_height_px"]:.1f}px, rect_h={orange_transform_info["blue_rectified_table_height_px"]:.1f}px')

    x_text, y_text = 30, 40
    for line in lines:
        cv2.putText(canvas, line, (x_text, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(canvas, line, (x_text, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (20, 20, 20), 1, cv2.LINE_AA)
        y_text += 30

    cv2.imwrite(save_path, canvas)


def save_height_debug_artifacts(video_file, df, summary_df, save_dir, base_name, camera_model, plane_height_cm):
    if video_file is None or not os.path.exists(video_file) or camera_model is None or summary_df is None or summary_df.empty:
        return

    reader = FrameReader(video_file)
    try:
        frame_to_row = {int(df.iloc[idx]['Frame']): idx for idx in range(len(df))}
        for _, row in summary_df.iterrows():
            speed_start = _to_int_or_none(row.get('net_zone_max_speed_start_frame', None))
            speed_end = _to_int_or_none(row.get('net_zone_max_speed_end_frame', None))
            ratio = row.get('plane_scale_ratio', '')
            corrected_speed = row.get('net_zone_max_speed_kmh', '')
            table_speed = row.get('net_zone_max_speed_table_kmh', '')
            stroke_id = _to_int_or_none(row.get('stroke_id', None))
            if speed_start is None or speed_end is None or stroke_id is None:
                continue
            if speed_start not in frame_to_row or speed_end not in frame_to_row:
                continue
            if ratio in ('', None) or table_speed in ('', None) or corrected_speed in ('', None):
                continue

            r1 = df.iloc[frame_to_row[speed_start]]
            r2 = df.iloc[frame_to_row[speed_end]]
            if not is_valid_point(r1) or not is_valid_point(r2):
                continue

            frame_mid = int(round((speed_start + speed_end) / 2.0))
            frame = reader.read_frame(frame_mid)
            if frame is None:
                continue

            table_corners = extract_zone_points(row, 'table', 4)
            if table_corners is None:
                continue

            x1, y1 = float(r1['X']), float(r1['Y'])
            x2, y2 = float(r2['X']), float(r2['Y'])

            overlay_path = os.path.join(save_dir, f'{base_name}_stroke_{stroke_id:02d}_plane_debug.png')
            curve_path = os.path.join(save_dir, f'{base_name}_stroke_{stroke_id:02d}_plane_ratio_curve.png')

            save_height_debug_overlay(
                overlay_path,
                frame,
                np.asarray(table_corners, dtype=np.float32),
                camera_model,
                plane_height_cm,
                (x1, y1),
                (x2, y2),
                float(table_speed),
                float(corrected_speed),
                float(ratio),
            )
            save_depth_ratio_curve(curve_path, (x1, y1), (x2, y2), camera_model, plane_height_cm)
    finally:
        reader.release()


def compute_fixed_scales(table_corners):
    """Compute fixed fallback cm-per-pixel scales from table corners ordered as LT, RT, RB, LB."""
    p1, p2, p3, p4 = np.asarray(table_corners, dtype=np.float32)

    top_sx = TABLE_W / calc_step(p1[0], p1[1], p2[0], p2[1])
    bottom_sx = TABLE_W / calc_step(p4[0], p4[1], p3[0], p3[1])
    left_sy = TABLE_H / calc_step(p1[0], p1[1], p4[0], p4[1])
    right_sy = TABLE_H / calc_step(p2[0], p2[1], p3[0], p3[1])

    return float((top_sx + bottom_sx) / 2.0), float((left_sy + right_sy) / 2.0)

def _table_y_ratio_from_y(table_corners, target_y):
    """Convert corrected image y to table ratio t.

    t = 0 means back/far edge p1->p2
    t = 1 means front/near edge p4->p3

    The ratio is estimated from the average y position of the left/right table edges,
    then clipped to the table range.
    """
    p1, p2, p3, p4 = np.asarray(table_corners, dtype=np.float32)[:4]

    back_y = (float(p1[1]) + float(p2[1])) / 2.0
    front_y = (float(p4[1]) + float(p3[1])) / 2.0

    denom = front_y - back_y
    if abs(denom) < 1e-6:
        return 0.5

    t = (float(target_y) - back_y) / denom
    return float(np.clip(t, 0.0, 1.0))


def compute_y_based_scales(table_corners, ball_y):
    """Use corrected table-plane y to compute the local table width directly.

    This version does NOT cut the table with an image-horizontal line.
    Instead, it finds the y-based position ratio on the table, then interpolates
    a line parallel to the table's front/back edges:

        left_pt  = p1 -> p4 at ratio t
        right_pt = p2 -> p3 at ratio t

    sx = TABLE_W / local_table_width_px
    """
    pts = np.asarray(table_corners, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] < 4:
        return compute_fixed_scales(table_corners)

    p1, p2, p3, p4 = pts[:4]

    t = _table_y_ratio_from_y(pts, ball_y)

    left_pt = p1 + t * (p4 - p1)
    right_pt = p2 + t * (p3 - p2)

    local_width_px = calc_step(left_pt[0], left_pt[1], right_pt[0], right_pt[1])

    if local_width_px < 1e-6:
        sx, sy = compute_fixed_scales(table_corners)
        return float(sx), float(sy)

    sx = float(TABLE_W / local_width_px)

    left_sy = TABLE_H / calc_step(p1[0], p1[1], p4[0], p4[1])
    right_sy = TABLE_H / calc_step(p2[0], p2[1], p3[0], p3[1])
    sy = float((left_sy + right_sy) / 2.0)

    return float(sx), float(sy)


def compute_local_table_width_line(table_corners, ball_y):
    """Return the local table-width line used for sx visualization.

    Output:
        left_x, left_y, right_x, right_y, local_width_px

    This line is parallel to the table front/back edges, not image-horizontal.
    """
    pts = np.asarray(table_corners, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] < 4:
        return None

    p1, p2, p3, p4 = pts[:4]

    t = _table_y_ratio_from_y(pts, ball_y)

    left_pt = p1 + t * (p4 - p1)
    right_pt = p2 + t * (p3 - p2)

    local_width_px = calc_step(left_pt[0], left_pt[1], right_pt[0], right_pt[1])

    return (
        float(left_pt[0]),
        float(left_pt[1]),
        float(right_pt[0]),
        float(right_pt[1]),
        float(local_width_px),
    )



def compute_local_width_line_by_ratio(table_corners, t):
    """Return local width line at table ratio t using p1->p4 and p2->p3."""
    pts = np.asarray(table_corners, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] < 4:
        return None

    t = float(np.clip(t, 0.0, 1.0))
    p1, p2, p3, p4 = pts[:4]
    left_pt = p1 + t * (p4 - p1)
    right_pt = p2 + t * (p3 - p2)
    local_width_px = calc_step(left_pt[0], left_pt[1], right_pt[0], right_pt[1])
    return (
        float(left_pt[0]), float(left_pt[1]),
        float(right_pt[0]), float(right_pt[1]),
        float(local_width_px),
    )


def polygon_area_px(points):
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] < 3:
        return None
    return float(abs(cv2.contourArea(pts.reshape(-1, 1, 2))))


def compute_average_table_height_px(table_corners):
    pts = np.asarray(table_corners, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] < 4:
        return None
    p1, p2, p3, p4 = pts[:4]
    left_h = calc_step(p1[0], p1[1], p4[0], p4[1])
    right_h = calc_step(p2[0], p2[1], p3[0], p3[1])
    return float((left_h + right_h) / 2.0)


def get_raised_plane_corners(camera_model, plane_height_cm):
    """Project the orange raised plane into image coordinates."""
    if camera_model is None or plane_height_cm is None or abs(float(plane_height_cm)) <= 0:
        return None

    effective_z = resolve_effective_plane_z(camera_model, plane_height_cm)
    raised_world = TABLE_WORLD_CORNERS.copy()
    raised_world[:, 2] = float(effective_z)
    raised = project_world_points(raised_world, camera_model)
    if raised is None or raised.shape[0] != 4:
        return None
    return np.asarray(raised, dtype=np.float32)



def get_orange_rect_size_px(scale_px_per_cm=ORANGE_RECT_SCALE_PX_PER_CM):
    rect_w = int(round(TABLE_W * float(scale_px_per_cm)))
    rect_h = int(round(TABLE_H * float(scale_px_per_cm)))
    return rect_w, rect_h


def build_plane_homography_to_cm(plane_corners):
    """Build homography from image pixels on a projected table-like plane to table cm coordinates."""
    src = np.asarray(plane_corners, dtype=np.float32)
    if src.ndim != 2 or src.shape[0] < 4:
        return None
    src = src[:4].astype(np.float32)
    dst = np.array([
        [0.0, 0.0],
        [TABLE_W, 0.0],
        [TABLE_W, TABLE_H],
        [0.0, TABLE_H],
    ], dtype=np.float32)
    return cv2.getPerspectiveTransform(src, dst)


def build_plane_homography_to_rect_px(plane_corners, scale_px_per_cm=ORANGE_RECT_SCALE_PX_PER_CM):
    """Build homography from image pixels on a projected plane to a rectified debug image."""
    src = np.asarray(plane_corners, dtype=np.float32)
    if src.ndim != 2 or src.shape[0] < 4:
        return None
    src = src[:4].astype(np.float32)
    rect_w, rect_h = get_orange_rect_size_px(scale_px_per_cm)
    dst = np.array([
        [0.0, 0.0],
        [float(rect_w), 0.0],
        [float(rect_w), float(rect_h)],
        [0.0, float(rect_h)],
    ], dtype=np.float32)
    return cv2.getPerspectiveTransform(src, dst)


def apply_homography_points(points, H):
    if H is None:
        return None
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
    return out.astype(np.float64)


def compute_orange_plane_transform_info(orange_corners, x1, y1, x2, y2, sy_probe_px=ORANGE_SY_PROBE_PX):
    """Compute direct orange-plane sx and local sy from the raised plane only.

    sx:
        Use the orange local table-width line at the speed segment midpoint y.
        sx_orange = TABLE_W / orange_local_width_px

    sy:
        Rectify the orange plane to true table cm coordinates and measure how
        far an image-space vertical step at the segment midpoint moves on that
        plane.  This gives a local cm-per-pixel value for image y displacement:
        sy_orange = distance_cm( H(mid_x, mid_y), H(mid_x, mid_y + probe_px) ) / probe_px

    The rectified-pixel values are saved only for debug/visualization.
    """
    orange = np.asarray(orange_corners, dtype=np.float32)
    if orange.ndim != 2 or orange.shape[0] < 4:
        return None

    mid_x = (float(x1) + float(x2)) / 2.0
    mid_y = (float(y1) + float(y2)) / 2.0

    t_orange = _table_y_ratio_from_y(orange, mid_y)
    orange_width_line = compute_local_width_line_by_ratio(orange, t_orange)
    if orange_width_line is None:
        return None

    orange_width_px = float(orange_width_line[4])
    if orange_width_px < 1e-6:
        return None
    sx_orange = float(TABLE_W / orange_width_px)

    H_cm = build_plane_homography_to_cm(orange)
    H_rect = build_plane_homography_to_rect_px(orange)
    if H_cm is None or H_rect is None:
        return None

    sy_probe_px = float(max(abs(float(sy_probe_px)), 1.0))
    img_pts = np.array([
        [float(x1), float(y1)],
        [float(x2), float(y2)],
        [mid_x, mid_y],
        [mid_x, mid_y + sy_probe_px],
    ], dtype=np.float32)

    cm_pts = apply_homography_points(img_pts, H_cm)
    rect_pts = apply_homography_points(img_pts, H_rect)
    if cm_pts is None or rect_pts is None:
        return None

    p1_cm, p2_cm, mid_cm, probe_cm = cm_pts
    p1_rect, p2_rect, mid_rect, probe_rect = rect_pts

    sy_probe_cm = float(np.linalg.norm(probe_cm - mid_cm))
    sy_orange = float(sy_probe_cm / sy_probe_px) if sy_probe_px > 1e-8 else None
    if sy_orange is None or not np.isfinite(sy_orange) or sy_orange <= 0:
        return None

    dx_cm_h = float(p2_cm[0] - p1_cm[0])
    dy_cm_h = float(p2_cm[1] - p1_cm[1])
    dist_cm_h = float(np.linalg.norm(p2_cm - p1_cm))

    dx_rect_px = float(p2_rect[0] - p1_rect[0])
    dy_rect_px = float(p2_rect[1] - p1_rect[1])
    dist_rect_px = float(np.linalg.norm(p2_rect - p1_rect))
    sy_probe_rect_px = float(np.linalg.norm(probe_rect - mid_rect))

    return {
        'orange_width_line': orange_width_line,
        'orange_local_width_px': orange_width_px,
        'orange_sx_cm_per_px': sx_orange,
        'orange_sy_cm_per_px': sy_orange,
        'orange_sy_probe_px': sy_probe_px,
        'orange_sy_probe_cm': sy_probe_cm,
        'orange_sy_probe_rect_px': sy_probe_rect_px,
        'orange_rect_scale_px_per_cm': float(ORANGE_RECT_SCALE_PX_PER_CM),
        'orange_rect_w_px': float(get_orange_rect_size_px()[0]),
        'orange_rect_h_px': float(get_orange_rect_size_px()[1]),
        'orange_rect_p1_x': float(p1_rect[0]),
        'orange_rect_p1_y': float(p1_rect[1]),
        'orange_rect_p2_x': float(p2_rect[0]),
        'orange_rect_p2_y': float(p2_rect[1]),
        'orange_rect_mid_x': float(mid_rect[0]),
        'orange_rect_mid_y': float(mid_rect[1]),
        'orange_rect_probe_x': float(probe_rect[0]),
        'orange_rect_probe_y': float(probe_rect[1]),
        'orange_rect_dx_px': dx_rect_px,
        'orange_rect_dy_px': dy_rect_px,
        'orange_rect_dist_px': dist_rect_px,
        'orange_homography_p1_cm_x': float(p1_cm[0]),
        'orange_homography_p1_cm_y': float(p1_cm[1]),
        'orange_homography_p2_cm_x': float(p2_cm[0]),
        'orange_homography_p2_cm_y': float(p2_cm[1]),
        'orange_homography_mid_cm_x': float(mid_cm[0]),
        'orange_homography_mid_cm_y': float(mid_cm[1]),
        'orange_homography_dx_cm': dx_cm_h,
        'orange_homography_dy_cm': dy_cm_h,
        'orange_homography_dist_cm': dist_cm_h,
        'table_ratio_t_orange': float(t_orange),
    }


def save_orange_rectified_debug(save_path, orange_corners, speed_start_xy, speed_end_xy, sy_probe_px=ORANGE_SY_PROBE_PX):
    """Save a rectified orange-plane view to explain how sy is estimated."""
    if orange_corners is None:
        return

    x1, y1 = speed_start_xy
    x2, y2 = speed_end_xy
    info = compute_orange_plane_transform_info(orange_corners, x1, y1, x2, y2, sy_probe_px=sy_probe_px)
    if info is None:
        return

    rect_w, rect_h = get_orange_rect_size_px()
    canvas = np.full((rect_h, rect_w, 3), 255, dtype=np.uint8)

    # Draw table boundary and coarse real-world grid.
    cv2.rectangle(canvas, (0, 0), (rect_w - 1, rect_h - 1), (0, 165, 255), 2, cv2.LINE_AA)
    grid_cm = 26.0
    for x_cm in np.arange(grid_cm, TABLE_W, grid_cm):
        x_px = int(round(x_cm * ORANGE_RECT_SCALE_PX_PER_CM))
        cv2.line(canvas, (x_px, 0), (x_px, rect_h - 1), (220, 220, 220), 1, cv2.LINE_AA)
    for y_cm in np.arange(grid_cm, TABLE_H, grid_cm):
        y_px = int(round(y_cm * ORANGE_RECT_SCALE_PX_PER_CM))
        cv2.line(canvas, (0, y_px), (rect_w - 1, y_px), (220, 220, 220), 1, cv2.LINE_AA)

    p1 = (int(round(info['orange_rect_p1_x'])), int(round(info['orange_rect_p1_y'])))
    p2 = (int(round(info['orange_rect_p2_x'])), int(round(info['orange_rect_p2_y'])))
    pm = (int(round(info['orange_rect_mid_x'])), int(round(info['orange_rect_mid_y'])))
    pp = (int(round(info['orange_rect_probe_x'])), int(round(info['orange_rect_probe_y'])))

    # Red: selected speed segment after orange-plane rectification.
    cv2.line(canvas, p1, p2, (0, 0, 255), 4, cv2.LINE_AA)
    cv2.circle(canvas, p1, 6, (0, 0, 255), -1)
    cv2.circle(canvas, p2, 6, (0, 0, 255), -1)

    # Purple: where image-space +Y probe maps on the rectified orange plane.
    cv2.line(canvas, pm, pp, (255, 0, 255), 3, cv2.LINE_AA)
    cv2.circle(canvas, pm, 6, (255, 0, 255), -1)
    cv2.circle(canvas, pp, 6, (255, 0, 255), 2)

    lines = [
        f'orange rectified plane: {rect_w}x{rect_h}px ({ORANGE_RECT_SCALE_PX_PER_CM:.1f}px/cm)',
        f'red rect dx={info["orange_rect_dx_px"]:.1f}px, dy={info["orange_rect_dy_px"]:.1f}px, dist={info["orange_rect_dist_px"]:.1f}px',
        f'red cm dx={info["orange_homography_dx_cm"]:.2f}, dy={info["orange_homography_dy_cm"]:.2f}, dist={info["orange_homography_dist_cm"]:.2f}',
        f'sx from orange width={info["orange_sx_cm_per_px"]:.4f} cm/img-px',
        f'sy probe: image +{info["orange_sy_probe_px"]:.0f}px y -> {info["orange_sy_probe_rect_px"]:.1f} rect-px = {info["orange_sy_probe_cm"]:.2f}cm',
        f'sy={info["orange_sy_cm_per_px"]:.4f} cm/img-px',
    ]

    x_text, y_text = 20, 35
    for line in lines:
        cv2.putText(canvas, line, (x_text, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(canvas, line, (x_text, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (20, 20, 20), 1, cv2.LINE_AA)
        y_text += 26


    cv2.imwrite(save_path, canvas)


def compute_blue_orange_mixed_scale_info(table_corners, orange_corners, x1, y1, x2, y2, table_y, sy_probe_px=ORANGE_SY_PROBE_PX):
    """Use blue table scale as baseline, orange plane only for x-height compensation.

    Final idea:
      sx_final = sx_blue * (orange_width_px / blue_width_px)
      sy_final = sy_blue * (rectified_table_height_px / original_table_height_px)

    Orange is used for the x-direction height ratio only.  The y-direction scale
    uses a table-height ratio: the rectified table height in pixels divided by
    the original slanted table height in pixels.  This intentionally treats
    the perspective-flattened height as an extra depth compensation factor.
    """
    blue = np.asarray(table_corners, dtype=np.float32)
    orange = np.asarray(orange_corners, dtype=np.float32)
    if blue.ndim != 2 or blue.shape[0] < 4 or orange.ndim != 2 or orange.shape[0] < 4:
        return None

    mid_x = (float(x1) + float(x2)) / 2.0
    mid_y = (float(y1) + float(y2)) / 2.0

    blue_width_line = compute_local_table_width_line(blue, table_y)
    if blue_width_line is None:
        return None
    blue_width_px = float(blue_width_line[4])
    if blue_width_px < 1e-6:
        return None

    t_orange = _table_y_ratio_from_y(orange, mid_y)
    orange_width_line = compute_local_width_line_by_ratio(orange, t_orange)
    if orange_width_line is None:
        return None
    orange_width_px = float(orange_width_line[4])
    if orange_width_px < 1e-6:
        return None

    ratio_x = float(orange_width_px / blue_width_px)

    H_blue_cm = build_plane_homography_to_cm(blue)
    H_blue_rect = build_plane_homography_to_rect_px(blue)
    if H_blue_cm is None or H_blue_rect is None:
        return None

    sy_probe_px = float(max(abs(float(sy_probe_px)), 1.0))
    img_pts = np.array([
        [float(x1), float(y1)],
        [float(x2), float(y2)],
        [mid_x, mid_y],
        [mid_x, mid_y + sy_probe_px],
    ], dtype=np.float32)

    blue_cm_pts = apply_homography_points(img_pts, H_blue_cm)
    blue_rect_pts = apply_homography_points(img_pts, H_blue_rect)
    if blue_cm_pts is None or blue_rect_pts is None:
        return None

    p1_cm, p2_cm, mid_cm, probe_cm = blue_cm_pts
    p1_rect, p2_rect, mid_rect, probe_rect = blue_rect_pts

    blue_sy_probe_cm = float(np.linalg.norm(probe_cm - mid_cm))
    blue_sy = float(blue_sy_probe_cm / sy_probe_px) if sy_probe_px > 1e-8 else None
    if blue_sy is None or not np.isfinite(blue_sy) or blue_sy <= 0:
        return None

    blue_dx_cm = float(p2_cm[0] - p1_cm[0])
    blue_dy_cm = float(p2_cm[1] - p1_cm[1])
    blue_dist_cm = float(np.linalg.norm(p2_cm - p1_cm))

    blue_rect_dx_px = float(p2_rect[0] - p1_rect[0])
    blue_rect_dy_px = float(p2_rect[1] - p1_rect[1])
    blue_rect_dist_px = float(np.linalg.norm(p2_rect - p1_rect))
    blue_sy_probe_rect_px = float(np.linalg.norm(probe_rect - mid_rect))

    blue_original_table_height_px = compute_average_table_height_px(blue)
    if blue_original_table_height_px is None or blue_original_table_height_px < 1e-6:
        return None

    # Depth compensation should not depend on the debug image scale
    # (ORANGE_RECT_SCALE_PX_PER_CM), and should not follow every local_y jitter.
    # Use the average of far/back and near/front table widths as a fixed
    # rectified-plane pixel scale.
    # p1->p2 is the far/back edge; p4->p3 is the near/front edge.
    p1_b, p2_b, p3_b, p4_b = blue[:4]
    far_width_px = calc_step(p1_b[0], p1_b[1], p2_b[0], p2_b[1])
    near_width_px = calc_step(p4_b[0], p4_b[1], p3_b[0], p3_b[1])
    avg_width_px = float((far_width_px + near_width_px) / 2.0)
    if avg_width_px < 1e-6:
        return None

    # avg_width_px corresponds to TABLE_W cm, so it defines a stable px/cm
    # scale for the rectified table height.
    avg_px_per_cm = float(avg_width_px / TABLE_W)
    blue_rectified_table_height_px = float(TABLE_H * avg_px_per_cm)
    blue_depth_ratio = float(blue_rectified_table_height_px / blue_original_table_height_px)

    return {
        'blue_width_line': blue_width_line,
        'orange_width_line': orange_width_line,
        'blue_local_width_px': blue_width_px,
        'orange_local_width_px': orange_width_px,
        'orange_blue_ratio_x': ratio_x,
        'orange_blue_ratio_y': None,
        'orange_blue_ratio_len': None,
        'orange_blue_ratio_area_sqrt': None,
        'blue_avg_height_px': blue_original_table_height_px,
        'orange_avg_height_px': compute_average_table_height_px(orange),
        'blue_area_px2': polygon_area_px(blue),
        'orange_area_px2': polygon_area_px(orange),
        'blue_original_table_height_px': blue_original_table_height_px,
        'blue_rectified_table_height_px': blue_rectified_table_height_px,
        'blue_depth_ratio': blue_depth_ratio,
        'blue_sy_cm_per_px': None,
        'blue_sy_probe_cm_per_px': blue_sy,
        'blue_sy_probe_px': sy_probe_px,
        'blue_sy_probe_cm': blue_sy_probe_cm,
        'blue_sy_probe_rect_px': blue_sy_probe_rect_px,
        'blue_rect_scale_px_per_cm': float(ORANGE_RECT_SCALE_PX_PER_CM),
        'blue_rect_w_px': float(get_orange_rect_size_px()[0]),
        'blue_rect_h_px': float(get_orange_rect_size_px()[1]),
        'blue_rect_p1_x': float(p1_rect[0]),
        'blue_rect_p1_y': float(p1_rect[1]),
        'blue_rect_p2_x': float(p2_rect[0]),
        'blue_rect_p2_y': float(p2_rect[1]),
        'blue_rect_mid_x': float(mid_rect[0]),
        'blue_rect_mid_y': float(mid_rect[1]),
        'blue_rect_probe_x': float(probe_rect[0]),
        'blue_rect_probe_y': float(probe_rect[1]),
        'blue_rect_dx_px': blue_rect_dx_px,
        'blue_rect_dy_px': blue_rect_dy_px,
        'blue_rect_dist_px': blue_rect_dist_px,
        'blue_homography_p1_cm_x': float(p1_cm[0]),
        'blue_homography_p1_cm_y': float(p1_cm[1]),
        'blue_homography_p2_cm_x': float(p2_cm[0]),
        'blue_homography_p2_cm_y': float(p2_cm[1]),
        'blue_homography_mid_cm_x': float(mid_cm[0]),
        'blue_homography_mid_cm_y': float(mid_cm[1]),
        'blue_homography_dx_cm': blue_dx_cm,
        'blue_homography_dy_cm': blue_dy_cm,
        'blue_homography_dist_cm': blue_dist_cm,
        'table_ratio_t_blue': float(_table_y_ratio_from_y(blue, table_y)),
        'table_ratio_t_orange': float(t_orange),
    }


def save_blue_rectified_sy_debug(save_path, table_corners, speed_start_xy, speed_end_xy, sy_probe_px=ORANGE_SY_PROBE_PX):
    """Save a rectified blue-table view to explain how the local sy is estimated."""
    if table_corners is None:
        return

    x1, y1 = speed_start_xy
    x2, y2 = speed_end_xy
    table_y = correct_ball_y_to_table_plane(
        ball_y=(float(y1) + float(y2)) / 2.0,
        table_corners=table_corners,
        y_offset_px=BALL_HEIGHT_Y_OFFSET_PX,
    )
    info = compute_blue_orange_mixed_scale_info(table_corners, table_corners, x1, y1, x2, y2, table_y, sy_probe_px=sy_probe_px)
    if info is None:
        return

    rect_w, rect_h = get_orange_rect_size_px()
    canvas = np.full((rect_h, rect_w, 3), 255, dtype=np.uint8)

    cv2.rectangle(canvas, (0, 0), (rect_w - 1, rect_h - 1), (255, 255, 0), 2, cv2.LINE_AA)
    grid_cm = 26.0
    for x_cm in np.arange(grid_cm, TABLE_W, grid_cm):
        x_px = int(round(x_cm * ORANGE_RECT_SCALE_PX_PER_CM))
        cv2.line(canvas, (x_px, 0), (x_px, rect_h - 1), (220, 220, 220), 1, cv2.LINE_AA)
    for y_cm in np.arange(grid_cm, TABLE_H, grid_cm):
        y_px = int(round(y_cm * ORANGE_RECT_SCALE_PX_PER_CM))
        cv2.line(canvas, (0, y_px), (rect_w - 1, y_px), (220, 220, 220), 1, cv2.LINE_AA)

    p1 = (int(round(info['blue_rect_p1_x'])), int(round(info['blue_rect_p1_y'])))
    p2 = (int(round(info['blue_rect_p2_x'])), int(round(info['blue_rect_p2_y'])))
    pm = (int(round(info['blue_rect_mid_x'])), int(round(info['blue_rect_mid_y'])))
    pp = (int(round(info['blue_rect_probe_x'])), int(round(info['blue_rect_probe_y'])))

    cv2.line(canvas, p1, p2, (0, 0, 255), 4, cv2.LINE_AA)
    cv2.circle(canvas, p1, 6, (0, 0, 255), -1)
    cv2.circle(canvas, p2, 6, (0, 0, 255), -1)
    cv2.line(canvas, pm, pp, (0, 180, 0), 3, cv2.LINE_AA)
    cv2.circle(canvas, pm, 6, (0, 180, 0), -1)
    cv2.circle(canvas, pp, 6, (0, 180, 0), 2)

    lines = [
        f'blue rectified table: {rect_w}x{rect_h}px ({ORANGE_RECT_SCALE_PX_PER_CM:.1f}px/cm)',
        f'red rect dx={info["blue_rect_dx_px"]:.1f}px, dy={info["blue_rect_dy_px"]:.1f}px, dist={info["blue_rect_dist_px"]:.1f}px',
        f'red cm dx={info["blue_homography_dx_cm"]:.2f}, dy={info["blue_homography_dy_cm"]:.2f}, dist={info["blue_homography_dist_cm"]:.2f}',
        f'depth ratio = avg_width_rect_h / original_h = {info["blue_rectified_table_height_px"]:.1f} / {info["blue_original_table_height_px"]:.1f} = {info["blue_depth_ratio"]:.4f}',
        f'sx_final = sx_blue * ratio_x; sy_final = sy_blue * depth_ratio; probe sy reference = {info["blue_sy_probe_cm_per_px"]:.4f}',
    ]
    x_text, y_text = 20, 35
    for line in lines:
        cv2.putText(canvas, line, (x_text, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(canvas, line, (x_text, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (20, 20, 20), 1, cv2.LINE_AA)
        y_text += 26

    cv2.imwrite(save_path, canvas)


def orange_corners_to_result_dict(orange_corners):
    result = {col: "" for col in ORANGE_OUT_COLS}
    if orange_corners is None:
        return result
    pts = np.asarray(orange_corners, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] < 4:
        return result
    for i in range(4):
        result[f"orange_p{i + 1}_x"] = float(pts[i, 0])
        result[f"orange_p{i + 1}_y"] = float(pts[i, 1])
    return result


def compute_orange_blue_scale_info(table_corners, camera_model, plane_height_cm, table_y):
    """Compute orange/blue scale ratios from projected table widths/heights.

    The original blue table scale is 274/local_width_px.  In this project the
    orange raised plane is used only as a correction ratio.  When orange is
    larger than blue, ratio_x > 1 and the final x scale becomes larger:
        sx_final = sx_blue * ratio_x
    This matches the user's intended compensation direction.
    """
    orange_corners = get_raised_plane_corners(camera_model, plane_height_cm)
    if orange_corners is None:
        return None

    blue = np.asarray(table_corners, dtype=np.float32)
    if blue.ndim != 2 or blue.shape[0] < 4:
        return None

    t = _table_y_ratio_from_y(blue, table_y)
    blue_width_line = compute_local_width_line_by_ratio(blue, t)
    orange_width_line = compute_local_width_line_by_ratio(orange_corners, t)
    if blue_width_line is None or orange_width_line is None:
        return None

    blue_width = float(blue_width_line[4])
    orange_width = float(orange_width_line[4])
    blue_height = compute_average_table_height_px(blue)
    orange_height = compute_average_table_height_px(orange_corners)
    blue_area = polygon_area_px(blue)
    orange_area = polygon_area_px(orange_corners)

    if blue_width < 1e-6 or orange_width < 1e-6 or blue_height is None or orange_height is None or blue_height < 1e-6 or orange_height < 1e-6:
        return None

    ratio_x = float(orange_width / blue_width)
    ratio_y = float(orange_height / blue_height)
    area_scale_ratio = None
    if blue_area is not None and orange_area is not None and blue_area > 1e-6 and orange_area > 1e-6:
        area_scale_ratio = float(math.sqrt(orange_area / blue_area))

    # A single display ratio only for detail/debug.  Actual speed uses x/y ratios separately.
    length_scale_ratio = float((ratio_x + ratio_y) / 2.0)

    return {
        'orange_corners': orange_corners,
        'blue_width_line': blue_width_line,
        'orange_width_line': orange_width_line,
        'blue_local_width_px': blue_width,
        'orange_local_width_px': orange_width,
        'blue_avg_height_px': float(blue_height),
        'orange_avg_height_px': float(orange_height),
        'blue_area_px2': float(blue_area) if blue_area is not None else None,
        'orange_area_px2': float(orange_area) if orange_area is not None else None,
        'orange_blue_ratio_x': ratio_x,
        'orange_blue_ratio_y': ratio_y,
        'orange_blue_ratio_len': length_scale_ratio,
        'orange_blue_ratio_area_sqrt': area_scale_ratio,
        'table_ratio_t': float(t),
    }

def correct_ball_y_to_table_plane(ball_y, table_corners, y_offset_px=BALL_HEIGHT_Y_OFFSET_PX):
    """Move observed ball y toward its estimated table-plane position before sx lookup.

    In image coordinates, larger y is closer to the camera/near side.
    A positive y_offset_px means the ball center is corrected toward the near side.
    The result is clipped to the table's near/far y range before local table-width lookup.
    """
    pts = np.asarray(table_corners, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] < 4:
        return float(ball_y)

    p1, p2, p3, p4 = pts[:4]
    edge12_y = (float(p1[1]) + float(p2[1])) / 2.0
    edge43_y = (float(p4[1]) + float(p3[1])) / 2.0
    far_y = min(edge12_y, edge43_y)
    near_y = max(edge12_y, edge43_y)

    corrected_y = float(ball_y) + float(y_offset_px)
    return float(np.clip(corrected_y, far_y, near_y))


def calc_segment_speed_basic_kmh(x1, y1, x2, y2, fps, sx, sy, dt_frames):
    # Keep the current project rule: x/y scales are blended differently.
    #sy = sx + 0.8 * (sy - sx)
    #sx = sx + 0.2 * (sy - sx)
    #scale = sx + 0.2 * (sy - sx)

    dx_cm = (x2 - x1) * sx
    dy_cm = (y2 - y1) * sy
    v_cm_s = math.hypot(dx_cm, dy_cm) / (dt_frames / fps)
    return float(v_cm_s * 0.036)


def make_speed_segment(
    df, i, j, fps, sx, sy, dt_frames, speed_end_frame, run_start_idx, run_end_idx,
    table_corners=None, camera_model=None, plane_height_cm=DEFAULT_PLANE_HEIGHT_CM,
    use_height_plane_scale=False,
):
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

    if x2 <= x1:
        return None

    mid_y = (y1 + y2) / 2.0
    if table_corners is not None:
        table_y = correct_ball_y_to_table_plane(
            ball_y=mid_y,
            table_corners=table_corners,
            y_offset_px=BALL_HEIGHT_Y_OFFSET_PX,
        )
        sx_use, sy_use = compute_y_based_scales(table_corners, ball_y=table_y)
    else:
        table_y = mid_y
        sx_use, sy_use = sx, sy

    speed_table = calc_segment_speed_basic_kmh(x1, y1, x2, y2, fps, sx_use, sy_use, dt_frames)
    if not np.isfinite(speed_table) or speed_table > MAX_SPEED_KMH:
        return None

    speed_selected = speed_table
    speed_plane = None
    plane_scale_ratio = None
    camera_d0_cm = None
    camera_dh_cm = None
    orange_info = None
    sx_final = sx_use
    sy_final = sy_use

    if use_height_plane_scale and table_corners is not None and camera_model is not None and plane_height_cm is not None and abs(float(plane_height_cm)) > 0:
        orange_corners = get_raised_plane_corners(camera_model, plane_height_cm)
        if orange_corners is not None:
            orange_info = compute_blue_orange_mixed_scale_info(table_corners, orange_corners, x1, y1, x2, y2, table_y)

        if orange_info is not None:
            ratio_x = orange_info['orange_blue_ratio_x']
            depth_ratio = orange_info.get('blue_depth_ratio')
            sx_final = float(sx_use * ratio_x * depth_ratio) if depth_ratio is not None else float(sx_use * ratio_x)
            sy_final = float(sy_use * depth_ratio) if depth_ratio is not None else sy_use
            speed_plane = calc_segment_speed_basic_kmh(x1, y1, x2, y2, fps, sx_final, sy_final, dt_frames)

            orange_info['blue_sy_cm_per_px'] = float(sy_final)
            orange_info['orange_blue_ratio_y'] = float(sy_final / sy_use) if sy_use > 1e-8 else None
            if orange_info['orange_blue_ratio_y'] is not None:
                orange_info['orange_blue_ratio_len'] = float((orange_info['orange_blue_ratio_x'] + orange_info['orange_blue_ratio_y']) / 2.0)

            # Keep the previous direct-orange numbers for comparison only.
            orange_direct_info = compute_orange_plane_transform_info(orange_corners, x1, y1, x2, y2)
            if orange_direct_info is not None:
                orange_info['orange_direct_speed_kmh'] = calc_segment_speed_basic_kmh(
                    x1, y1, x2, y2, fps,
                    float(orange_direct_info['orange_sx_cm_per_px']),
                    float(orange_direct_info['orange_sy_cm_per_px']),
                    dt_frames,
                )
                homography_dist_cm = orange_direct_info.get('orange_homography_dist_cm')
                if homography_dist_cm is not None and np.isfinite(homography_dist_cm):
                    orange_info['orange_homography_speed_kmh'] = float((homography_dist_cm / (dt_frames / fps)) * 0.036)
                for cmp_key in (
                    'orange_sx_cm_per_px', 'orange_sy_cm_per_px',
                    'orange_sy_probe_px', 'orange_sy_probe_cm', 'orange_sy_probe_rect_px',
                    'orange_rect_scale_px_per_cm', 'orange_rect_w_px', 'orange_rect_h_px',
                    'orange_rect_p1_x', 'orange_rect_p1_y', 'orange_rect_p2_x', 'orange_rect_p2_y',
                    'orange_rect_mid_x', 'orange_rect_mid_y', 'orange_rect_probe_x', 'orange_rect_probe_y',
                    'orange_rect_dx_px', 'orange_rect_dy_px', 'orange_rect_dist_px',
                    'orange_homography_p1_cm_x', 'orange_homography_p1_cm_y',
                    'orange_homography_p2_cm_x', 'orange_homography_p2_cm_y',
                    'orange_homography_mid_cm_x', 'orange_homography_mid_cm_y',
                    'orange_homography_dx_cm', 'orange_homography_dy_cm', 'orange_homography_dist_cm'
                ):
                    orange_info[cmp_key] = orange_direct_info.get(cmp_key)

            if np.isfinite(speed_plane):
                speed_selected = float(speed_plane)
                plane_scale_ratio = float(speed_selected / speed_table) if speed_table > 1e-8 else None

    # Final GT-based calibration scale.
    # The current setup is scaled by 0.75 from GT comparison (30 / 40).
    speed_table = float(speed_table) * SPEED_GT_SCALE_FACTOR
    if speed_plane is not None:
        speed_plane = float(speed_plane) * SPEED_GT_SCALE_FACTOR
    speed_selected = float(speed_selected) * SPEED_GT_SCALE_FACTOR

    if not np.isfinite(speed_selected) or speed_selected > MAX_SPEED_KMH:
        return None

    result = {
        'speed': float(speed_selected),
        'speed_table': float(speed_table),
        'speed_plane': float(speed_plane) if speed_plane is not None else None,
        'plane_scale_ratio': float(plane_scale_ratio) if plane_scale_ratio is not None else None,
        'camera_dist_z0_cm': float(camera_d0_cm) if camera_d0_cm is not None else None,
        'camera_dist_zh_cm': float(camera_dh_cm) if camera_dh_cm is not None else None,
        'start_frame': int(f1),
        'end_frame': int(f2),
        'sx_cm_per_px': float(sx_final),
        'sy_cm_per_px': float(sy_final),
        'sx_table_cm_per_px': float(sx_use),
        'sy_table_cm_per_px': float(sy_use),
        'table_y': float(table_y),
        'orange_blue_ratio_x': None,
        'orange_blue_ratio_y': None,
        'orange_blue_ratio_len': None,
        'orange_blue_ratio_area_sqrt': None,
        'blue_local_width_px': None,
        'orange_local_width_px': None,
        'blue_avg_height_px': None,
        'orange_avg_height_px': None,
        'blue_area_px2': None,
        'orange_area_px2': None,
        'blue_sy_cm_per_px': None,
        'blue_sy_probe_cm_per_px': None,
        'blue_sy_probe_px': None,
        'blue_sy_probe_cm': None,
        'blue_sy_probe_rect_px': None,
        'blue_original_table_height_px': None,
        'blue_rectified_table_height_px': None,
        'blue_depth_ratio': None,
        'blue_rect_scale_px_per_cm': None,
        'blue_rect_w_px': None,
        'blue_rect_h_px': None,
        'blue_rect_p1_x': None,
        'blue_rect_p1_y': None,
        'blue_rect_p2_x': None,
        'blue_rect_p2_y': None,
        'blue_rect_mid_x': None,
        'blue_rect_mid_y': None,
        'blue_rect_probe_x': None,
        'blue_rect_probe_y': None,
        'blue_rect_dx_px': None,
        'blue_rect_dy_px': None,
        'blue_rect_dist_px': None,
        'blue_homography_p1_cm_x': None,
        'blue_homography_p1_cm_y': None,
        'blue_homography_p2_cm_x': None,
        'blue_homography_p2_cm_y': None,
        'blue_homography_mid_cm_x': None,
        'blue_homography_mid_cm_y': None,
        'blue_homography_dx_cm': None,
        'blue_homography_dy_cm': None,
        'blue_homography_dist_cm': None,
        'orange_direct_speed_kmh': None,
        'orange_homography_speed_kmh': None,
        'orange_sx_cm_per_px': None,
        'orange_sy_cm_per_px': None,
        'orange_sy_probe_px': None,
        'orange_sy_probe_cm': None,
        'orange_sy_probe_rect_px': None,
        'orange_rect_scale_px_per_cm': None,
        'orange_rect_w_px': None,
        'orange_rect_h_px': None,
        'orange_rect_p1_x': None,
        'orange_rect_p1_y': None,
        'orange_rect_p2_x': None,
        'orange_rect_p2_y': None,
        'orange_rect_mid_x': None,
        'orange_rect_mid_y': None,
        'orange_rect_probe_x': None,
        'orange_rect_probe_y': None,
        'orange_rect_dx_px': None,
        'orange_rect_dy_px': None,
        'orange_rect_dist_px': None,
        'orange_homography_p1_cm_x': None,
        'orange_homography_p1_cm_y': None,
        'orange_homography_p2_cm_x': None,
        'orange_homography_p2_cm_y': None,
        'orange_homography_mid_cm_x': None,
        'orange_homography_mid_cm_y': None,
        'orange_homography_dx_cm': None,
        'orange_homography_dy_cm': None,
        'orange_homography_dist_cm': None,
    }
    if orange_info is not None:
        for key in (
            'orange_blue_ratio_x', 'orange_blue_ratio_y', 'orange_blue_ratio_len',
            'orange_blue_ratio_area_sqrt', 'blue_local_width_px', 'orange_local_width_px',
            'blue_avg_height_px', 'orange_avg_height_px', 'blue_area_px2', 'orange_area_px2',
            'blue_sy_cm_per_px', 'blue_sy_probe_cm_per_px', 'blue_sy_probe_px', 'blue_sy_probe_cm', 'blue_sy_probe_rect_px',
            'blue_original_table_height_px', 'blue_rectified_table_height_px', 'blue_depth_ratio',
            'blue_rect_scale_px_per_cm', 'blue_rect_w_px', 'blue_rect_h_px',
            'blue_rect_p1_x', 'blue_rect_p1_y', 'blue_rect_p2_x', 'blue_rect_p2_y',
            'blue_rect_mid_x', 'blue_rect_mid_y', 'blue_rect_probe_x', 'blue_rect_probe_y',
            'blue_rect_dx_px', 'blue_rect_dy_px', 'blue_rect_dist_px',
            'blue_homography_p1_cm_x', 'blue_homography_p1_cm_y',
            'blue_homography_p2_cm_x', 'blue_homography_p2_cm_y',
            'blue_homography_mid_cm_x', 'blue_homography_mid_cm_y',
            'blue_homography_dx_cm', 'blue_homography_dy_cm', 'blue_homography_dist_cm',
            'orange_direct_speed_kmh', 'orange_homography_speed_kmh',
            'orange_sx_cm_per_px', 'orange_sy_cm_per_px',
            'orange_sy_probe_px', 'orange_sy_probe_cm', 'orange_sy_probe_rect_px',
            'orange_rect_scale_px_per_cm', 'orange_rect_w_px', 'orange_rect_h_px',
            'orange_rect_p1_x', 'orange_rect_p1_y', 'orange_rect_p2_x', 'orange_rect_p2_y',
            'orange_rect_mid_x', 'orange_rect_mid_y', 'orange_rect_probe_x', 'orange_rect_probe_y',
            'orange_rect_dx_px', 'orange_rect_dy_px', 'orange_rect_dist_px',
            'orange_homography_p1_cm_x', 'orange_homography_p1_cm_y',
            'orange_homography_p2_cm_x', 'orange_homography_p2_cm_y',
            'orange_homography_mid_cm_x', 'orange_homography_mid_cm_y',
            'orange_homography_dx_cm', 'orange_homography_dy_cm', 'orange_homography_dist_cm'
        ):
            result[key] = orange_info.get(key)
    return result


def select_robust_net_speed(net_segments, top_n=ROBUST_NET_TOP_N):
    """Select the maximum valid speed segment inside the net zone.

    top_n is kept only for backward compatibility with older code, but it is
    intentionally not used. The user's current requirement is to use the direct
    maximum speed instead of the previous top-N median selection.
    """
    clean_segments = [
        seg for seg in net_segments
        if seg.get("best_speed") is not None and np.isfinite(seg.get("best_speed"))
    ]
    if not clean_segments:
        return None

    return max(clean_segments, key=lambda seg: float(seg["best_speed"]))

def compute_net_zone_speed_for_stroke(
    df: pd.DataFrame,
    stroke: Dict,
    fps: float,
    table_corners,
    net_zone_points,
    camera_model=None,
    plane_height_cm=DEFAULT_PLANE_HEIGHT_CM,
    use_height_plane_scale=False,
):
    """Compute only net-zone max speed.

    Baseline speed still comes from the current y-based local-table-width method.
    If use_height_plane_scale=True, we additionally compute a geometry ratio
    using the approximate camera model:

        ratio(h) = distance on z=h / distance on z=0

    and use corrected_speed = baseline_speed * ratio(h).
    """
    if table_corners is None or net_zone_points is None:
        return None

    frame_start = int(stroke["frame_start"])
    frame_end = int(stroke["frame_end"])
    run_start_idx = int(stroke["run_start_idx"])
    run_end_idx = int(stroke["run_end_idx"])
    sx, sy = compute_fixed_scales(table_corners)

    all_segments = []

    for i in range(run_start_idx, run_end_idx + 1):
        row = df.iloc[i]
        frame_id = int(row["Frame"])

        if frame_id < frame_start or frame_id > frame_end or not is_valid_point(row):
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
            seg = make_speed_segment(
                df, start_i, end_i, fps, sx, sy, dt_frames,
                speed_end_frame=frame_end,
                run_start_idx=run_start_idx,
                run_end_idx=run_end_idx,
                table_corners=table_corners,
                camera_model=camera_model,
                plane_height_cm=plane_height_cm,
                use_height_plane_scale=use_height_plane_scale,
            )
            if seg is not None:
                speed_candidates.append((speed_type, seg))

        if not speed_candidates:
            continue

        speed_map = {speed_type: seg['speed'] for speed_type, seg in speed_candidates}
        best_type, best_seg = max(speed_candidates, key=lambda item: item[1]['speed'])

        all_segments.append({
            'frame': frame_id,
            'best_type': best_type,
            'best_speed': float(best_seg['speed']),
            'best_speed_table': float(best_seg['speed_table']),
            'best_speed_plane': float(best_seg['speed_plane']) if best_seg['speed_plane'] is not None else None,
            'best_start': int(best_seg['start_frame']),
            'best_end': int(best_seg['end_frame']),
            'sx_cm_per_px': float(best_seg['sx_cm_per_px']),
            'sy_cm_per_px': float(best_seg['sy_cm_per_px']),
            'sx_table_cm_per_px': float(best_seg.get('sx_table_cm_per_px', best_seg['sx_cm_per_px'])),
            'sy_table_cm_per_px': float(best_seg.get('sy_table_cm_per_px', best_seg['sy_cm_per_px'])),
            'plane_scale_ratio': best_seg['plane_scale_ratio'],
            'orange_blue_ratio_x': best_seg.get('orange_blue_ratio_x'),
            'orange_blue_ratio_y': best_seg.get('orange_blue_ratio_y'),
            'orange_blue_ratio_len': best_seg.get('orange_blue_ratio_len'),
            'orange_blue_ratio_area_sqrt': best_seg.get('orange_blue_ratio_area_sqrt'),
            'blue_local_width_px': best_seg.get('blue_local_width_px'),
            'orange_local_width_px': best_seg.get('orange_local_width_px'),
            'blue_avg_height_px': best_seg.get('blue_avg_height_px'),
            'orange_avg_height_px': best_seg.get('orange_avg_height_px'),
            'blue_area_px2': best_seg.get('blue_area_px2'),
            'orange_area_px2': best_seg.get('orange_area_px2'),
            'blue_sy_cm_per_px': best_seg.get('blue_sy_cm_per_px'),
            'blue_sy_probe_cm_per_px': best_seg.get('blue_sy_probe_cm_per_px'),
            'blue_sy_probe_px': best_seg.get('blue_sy_probe_px'),
            'blue_sy_probe_cm': best_seg.get('blue_sy_probe_cm'),
            'blue_sy_probe_rect_px': best_seg.get('blue_sy_probe_rect_px'),
            'blue_original_table_height_px': best_seg.get('blue_original_table_height_px'),
            'blue_rectified_table_height_px': best_seg.get('blue_rectified_table_height_px'),
            'blue_depth_ratio': best_seg.get('blue_depth_ratio'),
            'blue_rect_scale_px_per_cm': best_seg.get('blue_rect_scale_px_per_cm'),
            'blue_rect_w_px': best_seg.get('blue_rect_w_px'),
            'blue_rect_h_px': best_seg.get('blue_rect_h_px'),
            'blue_rect_p1_x': best_seg.get('blue_rect_p1_x'),
            'blue_rect_p1_y': best_seg.get('blue_rect_p1_y'),
            'blue_rect_p2_x': best_seg.get('blue_rect_p2_x'),
            'blue_rect_p2_y': best_seg.get('blue_rect_p2_y'),
            'blue_rect_mid_x': best_seg.get('blue_rect_mid_x'),
            'blue_rect_mid_y': best_seg.get('blue_rect_mid_y'),
            'blue_rect_probe_x': best_seg.get('blue_rect_probe_x'),
            'blue_rect_probe_y': best_seg.get('blue_rect_probe_y'),
            'blue_rect_dx_px': best_seg.get('blue_rect_dx_px'),
            'blue_rect_dy_px': best_seg.get('blue_rect_dy_px'),
            'blue_rect_dist_px': best_seg.get('blue_rect_dist_px'),
            'blue_homography_p1_cm_x': best_seg.get('blue_homography_p1_cm_x'),
            'blue_homography_p1_cm_y': best_seg.get('blue_homography_p1_cm_y'),
            'blue_homography_p2_cm_x': best_seg.get('blue_homography_p2_cm_x'),
            'blue_homography_p2_cm_y': best_seg.get('blue_homography_p2_cm_y'),
            'blue_homography_mid_cm_x': best_seg.get('blue_homography_mid_cm_x'),
            'blue_homography_mid_cm_y': best_seg.get('blue_homography_mid_cm_y'),
            'blue_homography_dx_cm': best_seg.get('blue_homography_dx_cm'),
            'blue_homography_dy_cm': best_seg.get('blue_homography_dy_cm'),
            'blue_homography_dist_cm': best_seg.get('blue_homography_dist_cm'),
            'orange_direct_speed_kmh': best_seg.get('orange_direct_speed_kmh'),
            'orange_homography_speed_kmh': best_seg.get('orange_homography_speed_kmh'),
            'orange_sx_cm_per_px': best_seg.get('orange_sx_cm_per_px'),
            'orange_sy_cm_per_px': best_seg.get('orange_sy_cm_per_px'),
            'orange_sy_probe_px': best_seg.get('orange_sy_probe_px'),
            'orange_sy_probe_cm': best_seg.get('orange_sy_probe_cm'),
            'orange_sy_probe_rect_px': best_seg.get('orange_sy_probe_rect_px'),
            'orange_rect_scale_px_per_cm': best_seg.get('orange_rect_scale_px_per_cm'),
            'orange_rect_w_px': best_seg.get('orange_rect_w_px'),
            'orange_rect_h_px': best_seg.get('orange_rect_h_px'),
            'orange_rect_p1_x': best_seg.get('orange_rect_p1_x'),
            'orange_rect_p1_y': best_seg.get('orange_rect_p1_y'),
            'orange_rect_p2_x': best_seg.get('orange_rect_p2_x'),
            'orange_rect_p2_y': best_seg.get('orange_rect_p2_y'),
            'orange_rect_mid_x': best_seg.get('orange_rect_mid_x'),
            'orange_rect_mid_y': best_seg.get('orange_rect_mid_y'),
            'orange_rect_probe_x': best_seg.get('orange_rect_probe_x'),
            'orange_rect_probe_y': best_seg.get('orange_rect_probe_y'),
            'orange_rect_dx_px': best_seg.get('orange_rect_dx_px'),
            'orange_rect_dy_px': best_seg.get('orange_rect_dy_px'),
            'orange_rect_dist_px': best_seg.get('orange_rect_dist_px'),
            'orange_homography_p1_cm_x': best_seg.get('orange_homography_p1_cm_x'),
            'orange_homography_p1_cm_y': best_seg.get('orange_homography_p1_cm_y'),
            'orange_homography_p2_cm_x': best_seg.get('orange_homography_p2_cm_x'),
            'orange_homography_p2_cm_y': best_seg.get('orange_homography_p2_cm_y'),
            'orange_homography_mid_cm_x': best_seg.get('orange_homography_mid_cm_x'),
            'orange_homography_mid_cm_y': best_seg.get('orange_homography_mid_cm_y'),
            'orange_homography_dx_cm': best_seg.get('orange_homography_dx_cm'),
            'orange_homography_dy_cm': best_seg.get('orange_homography_dy_cm'),
            'orange_homography_dist_cm': best_seg.get('orange_homography_dist_cm'),
            'camera_dist_z0_cm': best_seg['camera_dist_z0_cm'],
            'camera_dist_zh_cm': best_seg['camera_dist_zh_cm'],
            'speed_1f': speed_map.get('1f'),
            'speed_2f': speed_map.get('2f'),
            'speed_c2f': speed_map.get('c2f'),
            'in_net': point_in_net_zone(x, y, net_zone_points),
        })

    if not all_segments:
        return None

    net_segments = [seg for seg in all_segments if seg['in_net']]
    best_net = select_robust_net_speed(net_segments) if net_segments else None

    return {
        'net_zone_max_speed_kmh': best_net['best_speed'] if best_net else None,
        'net_zone_max_speed_table_kmh': best_net['best_speed_table'] if best_net else None,
        'net_zone_max_speed_plane_kmh': best_net['best_speed_plane'] if best_net else None,
        'net_zone_max_speed_type': best_net['best_type'] if best_net else '',
        'net_zone_max_speed_start_frame': best_net['best_start'] if best_net else None,
        'net_zone_max_speed_end_frame': best_net['best_end'] if best_net else None,
        'net_zone_max_speed_1f_kmh': max((seg['speed_1f'] for seg in net_segments if seg['speed_1f'] is not None), default=None),
        'net_zone_max_speed_2f_kmh': max((seg['speed_2f'] for seg in net_segments if seg['speed_2f'] is not None), default=None),
        'net_zone_max_speed_c2f_kmh': max((seg['speed_c2f'] for seg in net_segments if seg['speed_c2f'] is not None), default=None),
        'sx_cm_per_px': best_net.get('sx_cm_per_px', sx) if best_net else sx,
        'sy_cm_per_px': best_net.get('sy_cm_per_px', sy) if best_net else sy,
        'sx_table_cm_per_px': best_net.get('sx_table_cm_per_px', sx) if best_net else sx,
        'sy_table_cm_per_px': best_net.get('sy_table_cm_per_px', sy) if best_net else sy,
        'plane_height_cm': get_display_plane_height_cm(plane_height_cm) if use_height_plane_scale else 0.0,
        'plane_scale_ratio': best_net.get('plane_scale_ratio', None) if best_net else None,
        'orange_blue_ratio_x': best_net.get('orange_blue_ratio_x', None) if best_net else None,
        'orange_blue_ratio_y': best_net.get('orange_blue_ratio_y', None) if best_net else None,
        'orange_blue_ratio_len': best_net.get('orange_blue_ratio_len', None) if best_net else None,
        'orange_blue_ratio_area_sqrt': best_net.get('orange_blue_ratio_area_sqrt', None) if best_net else None,
        'blue_local_width_px': best_net.get('blue_local_width_px', None) if best_net else None,
        'orange_local_width_px': best_net.get('orange_local_width_px', None) if best_net else None,
        'blue_avg_height_px': best_net.get('blue_avg_height_px', None) if best_net else None,
        'orange_avg_height_px': best_net.get('orange_avg_height_px', None) if best_net else None,
        'blue_area_px2': best_net.get('blue_area_px2', None) if best_net else None,
        'orange_area_px2': best_net.get('orange_area_px2', None) if best_net else None,
        'blue_sy_cm_per_px': best_net.get('blue_sy_cm_per_px', None) if best_net else None,
        'blue_sy_probe_cm_per_px': best_net.get('blue_sy_probe_cm_per_px', None) if best_net else None,
        'blue_sy_probe_px': best_net.get('blue_sy_probe_px', None) if best_net else None,
        'blue_sy_probe_cm': best_net.get('blue_sy_probe_cm', None) if best_net else None,
        'blue_sy_probe_rect_px': best_net.get('blue_sy_probe_rect_px', None) if best_net else None,
        'blue_original_table_height_px': best_net.get('blue_original_table_height_px', None) if best_net else None,
        'blue_rectified_table_height_px': best_net.get('blue_rectified_table_height_px', None) if best_net else None,
        'blue_depth_ratio': best_net.get('blue_depth_ratio', None) if best_net else None,
        'blue_rect_scale_px_per_cm': best_net.get('blue_rect_scale_px_per_cm', None) if best_net else None,
        'blue_rect_w_px': best_net.get('blue_rect_w_px', None) if best_net else None,
        'blue_rect_h_px': best_net.get('blue_rect_h_px', None) if best_net else None,
        'blue_rect_p1_x': best_net.get('blue_rect_p1_x', None) if best_net else None,
        'blue_rect_p1_y': best_net.get('blue_rect_p1_y', None) if best_net else None,
        'blue_rect_p2_x': best_net.get('blue_rect_p2_x', None) if best_net else None,
        'blue_rect_p2_y': best_net.get('blue_rect_p2_y', None) if best_net else None,
        'blue_rect_mid_x': best_net.get('blue_rect_mid_x', None) if best_net else None,
        'blue_rect_mid_y': best_net.get('blue_rect_mid_y', None) if best_net else None,
        'blue_rect_probe_x': best_net.get('blue_rect_probe_x', None) if best_net else None,
        'blue_rect_probe_y': best_net.get('blue_rect_probe_y', None) if best_net else None,
        'blue_rect_dx_px': best_net.get('blue_rect_dx_px', None) if best_net else None,
        'blue_rect_dy_px': best_net.get('blue_rect_dy_px', None) if best_net else None,
        'blue_rect_dist_px': best_net.get('blue_rect_dist_px', None) if best_net else None,
        'blue_homography_p1_cm_x': best_net.get('blue_homography_p1_cm_x', None) if best_net else None,
        'blue_homography_p1_cm_y': best_net.get('blue_homography_p1_cm_y', None) if best_net else None,
        'blue_homography_p2_cm_x': best_net.get('blue_homography_p2_cm_x', None) if best_net else None,
        'blue_homography_p2_cm_y': best_net.get('blue_homography_p2_cm_y', None) if best_net else None,
        'blue_homography_mid_cm_x': best_net.get('blue_homography_mid_cm_x', None) if best_net else None,
        'blue_homography_mid_cm_y': best_net.get('blue_homography_mid_cm_y', None) if best_net else None,
        'blue_homography_dx_cm': best_net.get('blue_homography_dx_cm', None) if best_net else None,
        'blue_homography_dy_cm': best_net.get('blue_homography_dy_cm', None) if best_net else None,
        'blue_homography_dist_cm': best_net.get('blue_homography_dist_cm', None) if best_net else None,
        'orange_direct_speed_kmh': best_net.get('orange_direct_speed_kmh', None) if best_net else None,
        'orange_homography_speed_kmh': best_net.get('orange_homography_speed_kmh', None) if best_net else None,
        'orange_sx_cm_per_px': best_net.get('orange_sx_cm_per_px', None) if best_net else None,
        'orange_sy_cm_per_px': best_net.get('orange_sy_cm_per_px', None) if best_net else None,
        'orange_sy_probe_px': best_net.get('orange_sy_probe_px', None) if best_net else None,
        'orange_sy_probe_cm': best_net.get('orange_sy_probe_cm', None) if best_net else None,
        'orange_sy_probe_rect_px': best_net.get('orange_sy_probe_rect_px', None) if best_net else None,
        'orange_rect_scale_px_per_cm': best_net.get('orange_rect_scale_px_per_cm', None) if best_net else None,
        'orange_rect_w_px': best_net.get('orange_rect_w_px', None) if best_net else None,
        'orange_rect_h_px': best_net.get('orange_rect_h_px', None) if best_net else None,
        'orange_rect_p1_x': best_net.get('orange_rect_p1_x', None) if best_net else None,
        'orange_rect_p1_y': best_net.get('orange_rect_p1_y', None) if best_net else None,
        'orange_rect_p2_x': best_net.get('orange_rect_p2_x', None) if best_net else None,
        'orange_rect_p2_y': best_net.get('orange_rect_p2_y', None) if best_net else None,
        'orange_rect_mid_x': best_net.get('orange_rect_mid_x', None) if best_net else None,
        'orange_rect_mid_y': best_net.get('orange_rect_mid_y', None) if best_net else None,
        'orange_rect_probe_x': best_net.get('orange_rect_probe_x', None) if best_net else None,
        'orange_rect_probe_y': best_net.get('orange_rect_probe_y', None) if best_net else None,
        'orange_rect_dx_px': best_net.get('orange_rect_dx_px', None) if best_net else None,
        'orange_rect_dy_px': best_net.get('orange_rect_dy_px', None) if best_net else None,
        'orange_rect_dist_px': best_net.get('orange_rect_dist_px', None) if best_net else None,
        'orange_homography_p1_cm_x': best_net.get('orange_homography_p1_cm_x', None) if best_net else None,
        'orange_homography_p1_cm_y': best_net.get('orange_homography_p1_cm_y', None) if best_net else None,
        'orange_homography_p2_cm_x': best_net.get('orange_homography_p2_cm_x', None) if best_net else None,
        'orange_homography_p2_cm_y': best_net.get('orange_homography_p2_cm_y', None) if best_net else None,
        'orange_homography_mid_cm_x': best_net.get('orange_homography_mid_cm_x', None) if best_net else None,
        'orange_homography_mid_cm_y': best_net.get('orange_homography_mid_cm_y', None) if best_net else None,
        'orange_homography_dx_cm': best_net.get('orange_homography_dx_cm', None) if best_net else None,
        'orange_homography_dy_cm': best_net.get('orange_homography_dy_cm', None) if best_net else None,
        'orange_homography_dist_cm': best_net.get('orange_homography_dist_cm', None) if best_net else None,
        'camera_dist_z0_cm': best_net.get('camera_dist_z0_cm', None) if best_net else None,
        'camera_dist_zh_cm': best_net.get('camera_dist_zh_cm', None) if best_net else None,
        'effective_plane_z_cm': resolve_effective_plane_z(camera_model, plane_height_cm) if use_height_plane_scale and camera_model is not None else 0.0,
        'height_plane_enabled': int(bool(use_height_plane_scale)),
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
                [zone_info[f"net_p{i}_x"], zone_info[f"net_p{i}_y"]] for i in range(1, 9)
            ], dtype=np.float32)
    except Exception:
        return None, None

    return table_corners, net_zone_points


def append_note(note: str, value: str) -> str:
    if note is None:
        note = ""
    try:
        if pd.isna(note):
            note = ""
    except Exception:
        pass

    note = str(note).strip()
    if note == "":
        return value
    if value in note.split(";"):
        return note
    return f"{note};{value}"


def update_net_note(df: pd.DataFrame, stroke: Dict, net_zone_points, note: str) -> str:
    if net_zone_points is None:
        return note

    # 保留 net_stop：最後一幀停在 net zone
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
    helper_zone_info: Dict,
    camera_model=None,
    plane_height_cm=DEFAULT_PLANE_HEIGHT_CM,
    use_height_plane_scale=False,
) -> pd.DataFrame:
    rows = []

    if helper_zone_info is None:
        raise ValueError("helper_zone_info is required. Run helper_table.py first to create *_helper_table.json.")

    orange_info = orange_corners_to_result_dict(None)
    if use_height_plane_scale and camera_model is not None:
        table_for_orange, _ = zone_info_to_arrays(helper_zone_info)
        orange_info = orange_corners_to_result_dict(get_raised_plane_corners(camera_model, plane_height_cm))

    for stroke in strokes:
        zone_info = helper_zone_info.copy()
        table_corners, net_zone_points = zone_info_to_arrays(zone_info)

        valid = stroke["valid"]
        note = stroke["note"]
        speed_metrics = None

        is_no_hit = int(valid) == 0 or "no_hit" in str(note).split(";")

        if is_no_hit:
            # no_hit means the stroke did not turn into a left-to-right hit.
            # Keep the row, but do not compute any speed.
            speed_metrics = None
            valid = 0
            note = append_note(note, "no_hit")
        elif table_corners is not None and net_zone_points is not None and fps is not None and fps > 0:
            speed_metrics = compute_net_zone_speed_for_stroke(
                df, stroke, fps, table_corners, net_zone_points,
                camera_model=camera_model,
                plane_height_cm=plane_height_cm,
                use_height_plane_scale=use_height_plane_scale,
            )
            if speed_metrics is None:
                valid = 0
                note = append_note(note, "no_clean_speed_segment")
            elif speed_metrics["net_zone_max_speed_kmh"] is None:
                note = append_note(note, "no_ball_in_net_zone")
        else:
            note = append_note(note, "no_video_or_table_geometry")

        note = update_net_note(df, stroke, net_zone_points, note)

        row_out = {
            "stroke_id": stroke["stroke_id"],
            "frame_start": safe_int(stroke["frame_start"]),
            "frame_end": safe_int(stroke["frame_end"]),
            "bounce_frame": int(stroke.get("bounce_frame", 0) or 0),
            "net_zone_max_speed_kmh": value_or_blank(speed_metrics["net_zone_max_speed_kmh"] if speed_metrics else None),
            "net_zone_max_speed_table_kmh": value_or_blank(speed_metrics["net_zone_max_speed_table_kmh"] if speed_metrics else None),
            "net_zone_max_speed_plane_kmh": value_or_blank(speed_metrics["net_zone_max_speed_plane_kmh"] if speed_metrics else None),
            "net_zone_max_speed_1f_kmh": value_or_blank(speed_metrics["net_zone_max_speed_1f_kmh"] if speed_metrics else None),
            "net_zone_max_speed_2f_kmh": value_or_blank(speed_metrics["net_zone_max_speed_2f_kmh"] if speed_metrics else None),
            "net_zone_max_speed_c2f_kmh": value_or_blank(speed_metrics["net_zone_max_speed_c2f_kmh"] if speed_metrics else None),
            "net_zone_max_speed_type": speed_metrics["net_zone_max_speed_type"] if speed_metrics else "",
            "net_zone_max_speed_start_frame": value_or_blank(speed_metrics["net_zone_max_speed_start_frame"] if speed_metrics else None),
            "net_zone_max_speed_end_frame": value_or_blank(speed_metrics["net_zone_max_speed_end_frame"] if speed_metrics else None),
            "sx_cm_per_px": value_or_blank(speed_metrics["sx_cm_per_px"] if speed_metrics else None),
            "sy_cm_per_px": value_or_blank(speed_metrics["sy_cm_per_px"] if speed_metrics else None),
            "sx_table_cm_per_px": value_or_blank(speed_metrics["sx_table_cm_per_px"] if speed_metrics else None),
            "sy_table_cm_per_px": value_or_blank(speed_metrics["sy_table_cm_per_px"] if speed_metrics else None),
            "plane_height_cm": value_or_blank(speed_metrics["plane_height_cm"] if speed_metrics else None),
            "plane_scale_ratio": value_or_blank(speed_metrics["plane_scale_ratio"] if speed_metrics else None),
            "orange_blue_ratio_x": value_or_blank(speed_metrics["orange_blue_ratio_x"] if speed_metrics else None),
            "orange_blue_ratio_y": value_or_blank(speed_metrics["orange_blue_ratio_y"] if speed_metrics else None),
            "orange_blue_ratio_len": value_or_blank(speed_metrics["orange_blue_ratio_len"] if speed_metrics else None),
            "orange_blue_ratio_area_sqrt": value_or_blank(speed_metrics["orange_blue_ratio_area_sqrt"] if speed_metrics else None),
            "blue_local_width_px": value_or_blank(speed_metrics["blue_local_width_px"] if speed_metrics else None),
            "orange_local_width_px": value_or_blank(speed_metrics["orange_local_width_px"] if speed_metrics else None),
            "blue_avg_height_px": value_or_blank(speed_metrics["blue_avg_height_px"] if speed_metrics else None),
            "orange_avg_height_px": value_or_blank(speed_metrics["orange_avg_height_px"] if speed_metrics else None),
            "blue_area_px2": value_or_blank(speed_metrics["blue_area_px2"] if speed_metrics else None),
            "orange_area_px2": value_or_blank(speed_metrics["orange_area_px2"] if speed_metrics else None),
            "blue_sy_cm_per_px": value_or_blank(speed_metrics["blue_sy_cm_per_px"] if speed_metrics else None),
            "blue_sy_probe_cm_per_px": value_or_blank(speed_metrics["blue_sy_probe_cm_per_px"] if speed_metrics else None),
            "blue_sy_probe_px": value_or_blank(speed_metrics["blue_sy_probe_px"] if speed_metrics else None),
            "blue_sy_probe_cm": value_or_blank(speed_metrics["blue_sy_probe_cm"] if speed_metrics else None),
            "blue_sy_probe_rect_px": value_or_blank(speed_metrics["blue_sy_probe_rect_px"] if speed_metrics else None),
            "blue_original_table_height_px": value_or_blank(speed_metrics["blue_original_table_height_px"] if speed_metrics else None),
            "blue_rectified_table_height_px": value_or_blank(speed_metrics["blue_rectified_table_height_px"] if speed_metrics else None),
            "blue_depth_ratio": value_or_blank(speed_metrics["blue_depth_ratio"] if speed_metrics else None),
            "blue_rect_scale_px_per_cm": value_or_blank(speed_metrics["blue_rect_scale_px_per_cm"] if speed_metrics else None),
            "blue_rect_w_px": value_or_blank(speed_metrics["blue_rect_w_px"] if speed_metrics else None),
            "blue_rect_h_px": value_or_blank(speed_metrics["blue_rect_h_px"] if speed_metrics else None),
            "blue_rect_p1_x": value_or_blank(speed_metrics["blue_rect_p1_x"] if speed_metrics else None),
            "blue_rect_p1_y": value_or_blank(speed_metrics["blue_rect_p1_y"] if speed_metrics else None),
            "blue_rect_p2_x": value_or_blank(speed_metrics["blue_rect_p2_x"] if speed_metrics else None),
            "blue_rect_p2_y": value_or_blank(speed_metrics["blue_rect_p2_y"] if speed_metrics else None),
            "blue_rect_mid_x": value_or_blank(speed_metrics["blue_rect_mid_x"] if speed_metrics else None),
            "blue_rect_mid_y": value_or_blank(speed_metrics["blue_rect_mid_y"] if speed_metrics else None),
            "blue_rect_probe_x": value_or_blank(speed_metrics["blue_rect_probe_x"] if speed_metrics else None),
            "blue_rect_probe_y": value_or_blank(speed_metrics["blue_rect_probe_y"] if speed_metrics else None),
            "blue_rect_dx_px": value_or_blank(speed_metrics["blue_rect_dx_px"] if speed_metrics else None),
            "blue_rect_dy_px": value_or_blank(speed_metrics["blue_rect_dy_px"] if speed_metrics else None),
            "blue_rect_dist_px": value_or_blank(speed_metrics["blue_rect_dist_px"] if speed_metrics else None),
            "blue_homography_p1_cm_x": value_or_blank(speed_metrics["blue_homography_p1_cm_x"] if speed_metrics else None),
            "blue_homography_p1_cm_y": value_or_blank(speed_metrics["blue_homography_p1_cm_y"] if speed_metrics else None),
            "blue_homography_p2_cm_x": value_or_blank(speed_metrics["blue_homography_p2_cm_x"] if speed_metrics else None),
            "blue_homography_p2_cm_y": value_or_blank(speed_metrics["blue_homography_p2_cm_y"] if speed_metrics else None),
            "blue_homography_mid_cm_x": value_or_blank(speed_metrics["blue_homography_mid_cm_x"] if speed_metrics else None),
            "blue_homography_mid_cm_y": value_or_blank(speed_metrics["blue_homography_mid_cm_y"] if speed_metrics else None),
            "blue_homography_dx_cm": value_or_blank(speed_metrics["blue_homography_dx_cm"] if speed_metrics else None),
            "blue_homography_dy_cm": value_or_blank(speed_metrics["blue_homography_dy_cm"] if speed_metrics else None),
            "blue_homography_dist_cm": value_or_blank(speed_metrics["blue_homography_dist_cm"] if speed_metrics else None),
            "orange_direct_speed_kmh": value_or_blank(speed_metrics["orange_direct_speed_kmh"] if speed_metrics else None),
            "orange_homography_speed_kmh": value_or_blank(speed_metrics["orange_homography_speed_kmh"] if speed_metrics else None),
            "orange_sx_cm_per_px": value_or_blank(speed_metrics["orange_sx_cm_per_px"] if speed_metrics else None),
            "orange_sy_cm_per_px": value_or_blank(speed_metrics["orange_sy_cm_per_px"] if speed_metrics else None),
            "orange_sy_probe_px": value_or_blank(speed_metrics["orange_sy_probe_px"] if speed_metrics else None),
            "orange_sy_probe_cm": value_or_blank(speed_metrics["orange_sy_probe_cm"] if speed_metrics else None),
            "orange_sy_probe_rect_px": value_or_blank(speed_metrics["orange_sy_probe_rect_px"] if speed_metrics else None),
            "orange_rect_scale_px_per_cm": value_or_blank(speed_metrics["orange_rect_scale_px_per_cm"] if speed_metrics else None),
            "orange_rect_w_px": value_or_blank(speed_metrics["orange_rect_w_px"] if speed_metrics else None),
            "orange_rect_h_px": value_or_blank(speed_metrics["orange_rect_h_px"] if speed_metrics else None),
            "orange_rect_p1_x": value_or_blank(speed_metrics["orange_rect_p1_x"] if speed_metrics else None),
            "orange_rect_p1_y": value_or_blank(speed_metrics["orange_rect_p1_y"] if speed_metrics else None),
            "orange_rect_p2_x": value_or_blank(speed_metrics["orange_rect_p2_x"] if speed_metrics else None),
            "orange_rect_p2_y": value_or_blank(speed_metrics["orange_rect_p2_y"] if speed_metrics else None),
            "orange_rect_mid_x": value_or_blank(speed_metrics["orange_rect_mid_x"] if speed_metrics else None),
            "orange_rect_mid_y": value_or_blank(speed_metrics["orange_rect_mid_y"] if speed_metrics else None),
            "orange_rect_probe_x": value_or_blank(speed_metrics["orange_rect_probe_x"] if speed_metrics else None),
            "orange_rect_probe_y": value_or_blank(speed_metrics["orange_rect_probe_y"] if speed_metrics else None),
            "orange_rect_dx_px": value_or_blank(speed_metrics["orange_rect_dx_px"] if speed_metrics else None),
            "orange_rect_dy_px": value_or_blank(speed_metrics["orange_rect_dy_px"] if speed_metrics else None),
            "orange_rect_dist_px": value_or_blank(speed_metrics["orange_rect_dist_px"] if speed_metrics else None),
            "orange_homography_p1_cm_x": value_or_blank(speed_metrics["orange_homography_p1_cm_x"] if speed_metrics else None),
            "orange_homography_p1_cm_y": value_or_blank(speed_metrics["orange_homography_p1_cm_y"] if speed_metrics else None),
            "orange_homography_p2_cm_x": value_or_blank(speed_metrics["orange_homography_p2_cm_x"] if speed_metrics else None),
            "orange_homography_p2_cm_y": value_or_blank(speed_metrics["orange_homography_p2_cm_y"] if speed_metrics else None),
            "orange_homography_mid_cm_x": value_or_blank(speed_metrics["orange_homography_mid_cm_x"] if speed_metrics else None),
            "orange_homography_mid_cm_y": value_or_blank(speed_metrics["orange_homography_mid_cm_y"] if speed_metrics else None),
            "orange_homography_dx_cm": value_or_blank(speed_metrics["orange_homography_dx_cm"] if speed_metrics else None),
            "orange_homography_dy_cm": value_or_blank(speed_metrics["orange_homography_dy_cm"] if speed_metrics else None),
            "orange_homography_dist_cm": value_or_blank(speed_metrics["orange_homography_dist_cm"] if speed_metrics else None),
            "camera_dist_z0_cm": value_or_blank(speed_metrics["camera_dist_z0_cm"] if speed_metrics else None),
            "camera_dist_zh_cm": value_or_blank(speed_metrics["camera_dist_zh_cm"] if speed_metrics else None),
            "effective_plane_z_cm": value_or_blank(speed_metrics["effective_plane_z_cm"] if speed_metrics else None),
            "height_plane_enabled": value_or_blank(speed_metrics["height_plane_enabled"] if speed_metrics else None),
            "valid": valid,
            "note": note,
        }
        row_out.update(zone_info)
        row_out.update(orange_info)
        rows.append(row_out)

    return pd.DataFrame(rows)


def _to_int_or_none(value):
    try:
        if value is None or pd.isna(value) or value == "":
            return None
        return int(float(value))
    except Exception:
        return None


def _draw_speed_scale_debug(frame, df, frame_to_row, stroke, summary_map):
    """Draw the selected speed segment and the corrected-y table width line."""
    sid = int(stroke.get("stroke_id", -1))
    summary_row = summary_map.get(sid)
    if summary_row is None:
        return

    speed_start = _to_int_or_none(summary_row.get("net_zone_max_speed_start_frame", None))
    speed_end = _to_int_or_none(summary_row.get("net_zone_max_speed_end_frame", None))
    if speed_start is None or speed_end is None:
        return
    if speed_start not in frame_to_row or speed_end not in frame_to_row:
        return

    r1 = df.iloc[frame_to_row[speed_start]]
    r2 = df.iloc[frame_to_row[speed_end]]
    if not is_valid_point(r1) or not is_valid_point(r2):
        return

    x1, y1 = float(r1["X"]), float(r1["Y"])
    x2, y2 = float(r2["X"]), float(r2["Y"])
    if min(x1, y1, x2, y2) <= 0:
        return

    table_corners = extract_zone_points(summary_row, "table", 4)
    if table_corners is None:
        return
    table_corners = np.asarray(table_corners, dtype=np.float32)

    mid_x = (x1 + x2) / 2.0
    mid_y = (y1 + y2) / 2.0
    table_y = correct_ball_y_to_table_plane(
        ball_y=mid_y,
        table_corners=table_corners,
        y_offset_px=BALL_HEIGHT_Y_OFFSET_PX,
    )
    width_line = compute_local_table_width_line(table_corners, table_y)
    if width_line is None:
        return

    orange_corners = extract_zone_points(summary_row, "orange", 4)
    orange_width_line = None
    orange_local_width_px = None
    orange_sy_text = ""
    if orange_corners is not None:
        orange_corners = np.asarray(orange_corners, dtype=np.float32)
        orange_transform_info = compute_blue_orange_mixed_scale_info(table_corners, orange_corners, x1, y1, x2, y2, table_y)
        if orange_transform_info is not None:
            _, sy_table_for_debug = compute_y_based_scales(table_corners, table_y)
            if orange_transform_info.get('blue_depth_ratio') is not None:
                orange_transform_info['blue_sy_cm_per_px'] = float(sy_table_for_debug * orange_transform_info['blue_depth_ratio'])
            orange_width_line = orange_transform_info['orange_width_line']
            orange_local_width_px = float(orange_transform_info['orange_local_width_px'])
            orange_sy_text = f"blue sy={orange_transform_info.get('blue_sy_cm_per_px', 0):.4f}, h_ratio={orange_transform_info.get('blue_depth_ratio', 0):.3f}"

    left_x, left_y, right_x, right_y, local_width_px = width_line

    p_start = (int(round(x1)), int(round(y1)))
    p_end = (int(round(x2)), int(round(y2)))
    p_mid = (int(round(mid_x)), int(round(mid_y)))
    p_table = (int(round(mid_x)), int(round(table_y)))
    p_left = (int(round(left_x)), int(round(left_y)))
    p_right = (int(round(right_x)), int(round(right_y)))

    # Red: selected speed segment.
    cv2.line(frame, p_start, p_end, (0, 0, 255), 4, cv2.LINE_AA)
    cv2.circle(frame, p_start, 7, (0, 0, 255), -1)
    cv2.circle(frame, p_end, 7, (0, 0, 255), -1)

    # Purple: y correction from observed segment midpoint to table-plane y.
    cv2.circle(frame, p_mid, 7, (255, 0, 255), -1)
    cv2.circle(frame, p_table, 7, (255, 0, 255), 2)
    cv2.line(frame, p_mid, p_table, (255, 0, 255), 2, cv2.LINE_AA)

    # Green: image-y probe used for blue rectified local sy.
    p_sy_probe = (int(round(mid_x)), int(round(mid_y + ORANGE_SY_PROBE_PX)))
    cv2.line(frame, p_mid, p_sy_probe, (0, 180, 0), 3, cv2.LINE_AA)
    cv2.circle(frame, p_sy_probe, 6, (0, 180, 0), 2)

    # Cyan: blue table local width used by the original table scale.
    cv2.line(frame, p_left, p_right, (255, 255, 0), 3, cv2.LINE_AA)
    cv2.circle(frame, p_left, 5, (255, 255, 0), -1)
    cv2.circle(frame, p_right, 5, (255, 255, 0), -1)

    if orange_width_line is not None:
        olx, oly, orx, ory, _ = orange_width_line
        p_ol = (int(round(olx)), int(round(oly)))
        p_or = (int(round(orx)), int(round(ory)))
        cv2.line(frame, p_ol, p_or, (0, 165, 255), 3, cv2.LINE_AA)
        cv2.circle(frame, p_ol, 5, (0, 165, 255), -1)
        cv2.circle(frame, p_or, 5, (0, 165, 255), -1)

    speed = summary_row.get("net_zone_max_speed_kmh", "")
    ratio_value = summary_row.get("plane_scale_ratio", "")
    table_speed_value = summary_row.get("net_zone_max_speed_table_kmh", "")
    sx_value = summary_row.get("sx_cm_per_px", "")
    try:
        speed_text = f"{float(speed):.1f} km/h"
    except Exception:
        speed_text = str(speed)
    try:
        sx_text = f"sx={float(sx_value):.4f} cm/px"
    except Exception:
        sx_text = f"sx={sx_value}"

    text_x = min(max(int(round(mid_x)) + 12, 20), frame.shape[1] - 420)
    text_y = min(max(int(round(table_y)) - 45, 120), frame.shape[0] - 80)

    cv2.putText(frame, f"speed seg {speed_start}->{speed_end}: {speed_text}",
                (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"mid_y {mid_y:.1f} -> table_y {table_y:.1f}",
                (text_x, text_y + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"blue width={local_width_px:.1f}px, {sx_text}",
                (text_x, text_y + 52), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2, cv2.LINE_AA)
    if ratio_value not in ("", None):
        try:
            ratio_text = f"mixed/table ratio={float(ratio_value):.4f}"
        except Exception:
            ratio_text = f"mixed/table ratio={ratio_value}"
        try:
            table_speed_text = f"base table speed={float(table_speed_value):.1f} km/h"
        except Exception:
            table_speed_text = f"base table speed={table_speed_value}"
        orange_width_text = ""
        if orange_local_width_px is not None:
            orange_width_text = f"orange width={orange_local_width_px:.1f}px"
        cv2.putText(frame, ratio_text,
                    (text_x, text_y + 78), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, table_speed_text,
                    (text_x, text_y + 104), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255), 2, cv2.LINE_AA)
        if orange_width_text:
            cv2.putText(frame, orange_width_text,
                        (text_x, text_y + 130), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255), 2, cv2.LINE_AA)
        if orange_sy_text:
            cv2.putText(frame, orange_sy_text,
                        (text_x, text_y + 156), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255), 2, cv2.LINE_AA)


def draw_visual_video(
    video_path: str,
    df: pd.DataFrame,
    strokes: List[Dict],
    summary_df: pd.DataFrame,
    output_video_path: str,
    video_codec: str = "h264_nvenc",
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    try:
        out = FFmpegWriter(output_video_path, width, height, fps, codec=video_codec)
        used_codec = video_codec
    except Exception as e:
        print(f"[draw_visual_video] {video_codec} init failed ({e}), fallback to libx264")
        out = FFmpegWriter(output_video_path, width, height, fps, codec="libx264")
        used_codec = "libx264"

    print(f"FFmpegWriter opened: codec={used_codec}, {width}x{height}@{fps:.2f}fps, save={output_video_path}")

    frame_to_stroke = build_frame_to_stroke_map(strokes)
    summary_map = {int(row["stroke_id"]): row for _, row in summary_df.iterrows()}
    frame_to_row = {int(df.iloc[idx]["Frame"]): idx for idx in range(len(df))}

    default_summary_row = summary_df.iloc[0] if summary_df is not None and len(summary_df) > 0 else None

    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Always draw the helper_table geometry, so the visualize video clearly shows
        # the fixed table and near-net zone read from *_helper_table.json.
        if default_summary_row is not None:
            table_pts = extract_zone_points(default_summary_row, "table", 4)
            net_pts = extract_zone_points(default_summary_row, "net", 8)
            if table_pts is not None:
                draw_polygon(frame, table_pts, (255, 0, 0), thickness=2, fill=False)
            orange_pts = extract_zone_points(default_summary_row, "orange", 4)
            if orange_pts is not None:
                draw_polygon(frame, orange_pts, (0, 165, 255), thickness=2, fill=False)
            if net_pts is not None:
                draw_helper_box(frame, net_pts, color=(0, 255, 255), thickness=2, fill_alpha=0.12)

        if frame_id in frame_to_stroke and frame_id in frame_to_row:
            stroke = frame_to_stroke[frame_id]
            row = df.iloc[frame_to_row[frame_id]]
            if int(row["Visibility"]) == 1:
                draw_stroke_overlay(frame, df, frame_to_row, frame_id, stroke, summary_map)

        # Draw current frame number on every frame
        cv2.putText(frame,f"Frame: {frame_id}", (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3, cv2.LINE_AA,)

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

    note = stroke.get("note", "")
    text = f"stroke={sid} start={f_start} end={f_end} bounce={bounce_frame} valid={stroke['valid']} {note}"
    cv2.putText(frame, text, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

    if DRAW_SPEED_SCALE_DEBUG:
        _draw_speed_scale_debug(frame, df, frame_to_row, stroke, summary_map)

    # Table / net zone is drawn once per frame in draw_visual_video().
    # Do not draw it again here, otherwise stroke frames get a double-filled zone.

    if frame_id == f_start:
        cv2.putText(frame, "START", (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
    if bounce_frame and frame_id == int(bounce_frame):
        cv2.putText(frame, "BOUNCE", (x + 10, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2, cv2.LINE_AA)
    if frame_id == f_end:
        cv2.putText(frame, "END", (x + 10, y + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2, cv2.LINE_AA)


def build_export_stroke_csv(summary_df: pd.DataFrame) -> pd.DataFrame:
    # Final/public CSV stays clean: only the final selected speed is exported here.
    # Intermediate speed variants and correction ratios are kept in *_net_zone_speed_detail.csv.
    keep_cols = [
        "stroke_id",
        "frame_start",
        "frame_end",
        "bounce_frame",
        "net_zone_max_speed_kmh",
        "net_zone_max_speed_type",
        "net_zone_max_speed_start_frame",
        "net_zone_max_speed_end_frame",
        "zone_label",
        "in_table",
        "valid",
        "note",
    ]
    return keep_columns(summary_df, keep_cols)


def build_export_zone_detail_csv(summary_df_full: pd.DataFrame) -> pd.DataFrame:
    return keep_columns(summary_df_full, ["stroke_id"] + ZONE_OUT_COLS + ORANGE_OUT_COLS)


def build_export_speed_detail_csv(summary_df_full: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [
        "stroke_id", "frame_start", "frame_end", "bounce_frame", "valid", "note",
        "net_zone_max_speed_kmh", "net_zone_max_speed_table_kmh", "net_zone_max_speed_plane_kmh", "net_zone_max_speed_type",
        "net_zone_max_speed_start_frame", "net_zone_max_speed_end_frame",
        "net_zone_max_speed_1f_kmh", "net_zone_max_speed_2f_kmh", "net_zone_max_speed_c2f_kmh",
        "sx_cm_per_px", "sy_cm_per_px", "sx_table_cm_per_px", "sy_table_cm_per_px",
        "plane_height_cm", "plane_scale_ratio", "orange_blue_ratio_x", "orange_blue_ratio_y", "orange_blue_ratio_len", "orange_blue_ratio_area_sqrt",
        "blue_local_width_px", "orange_local_width_px", "blue_avg_height_px", "orange_avg_height_px", "blue_area_px2", "orange_area_px2",
        "blue_sy_cm_per_px", "blue_sy_probe_cm_per_px", "blue_sy_probe_px", "blue_sy_probe_cm", "blue_sy_probe_rect_px",
        "blue_original_table_height_px", "blue_rectified_table_height_px", "blue_depth_ratio",
        "blue_rect_scale_px_per_cm", "blue_rect_w_px", "blue_rect_h_px",
        "blue_rect_p1_x", "blue_rect_p1_y", "blue_rect_p2_x", "blue_rect_p2_y",
        "blue_rect_mid_x", "blue_rect_mid_y", "blue_rect_probe_x", "blue_rect_probe_y",
        "blue_rect_dx_px", "blue_rect_dy_px", "blue_rect_dist_px",
        "blue_homography_p1_cm_x", "blue_homography_p1_cm_y",
        "blue_homography_p2_cm_x", "blue_homography_p2_cm_y",
        "blue_homography_mid_cm_x", "blue_homography_mid_cm_y",
        "blue_homography_dx_cm", "blue_homography_dy_cm", "blue_homography_dist_cm",
        "orange_direct_speed_kmh", "orange_homography_speed_kmh",
        "orange_sx_cm_per_px", "orange_sy_cm_per_px",
        "orange_sy_probe_px", "orange_sy_probe_cm", "orange_sy_probe_rect_px",
        "orange_rect_scale_px_per_cm", "orange_rect_w_px", "orange_rect_h_px",
        "orange_rect_p1_x", "orange_rect_p1_y", "orange_rect_p2_x", "orange_rect_p2_y",
        "orange_rect_mid_x", "orange_rect_mid_y", "orange_rect_probe_x", "orange_rect_probe_y",
        "orange_rect_dx_px", "orange_rect_dy_px", "orange_rect_dist_px",
        "orange_homography_p1_cm_x", "orange_homography_p1_cm_y",
        "orange_homography_p2_cm_x", "orange_homography_p2_cm_y",
        "orange_homography_mid_cm_x", "orange_homography_mid_cm_y",
        "orange_homography_dx_cm", "orange_homography_dy_cm", "orange_homography_dist_cm",
        "camera_dist_z0_cm", "camera_dist_zh_cm", "effective_plane_z_cm", "height_plane_enabled",
    ]
    return keep_columns(summary_df_full, keep_cols)



def _series_to_bool(series: pd.Series) -> pd.Series:
    """Convert mixed bool/string/number columns to bool safely."""
    return series.fillna(False).astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y"])


def get_table_hit_speed_df_for_analysis(summary_df_full: pd.DataFrame, speed_col: str = "net_zone_max_speed_kmh") -> pd.DataFrame:
    """Return only table-hit rows for final mean/max speed analysis.

    This does not change/export/filter the original summary_df_full.
    Use this only at the final statistics step, so stroke_zone, landing_detail,
    zone_detail, and speed_detail still keep all strokes.
    """
    if summary_df_full is None or summary_df_full.empty:
        return pd.DataFrame()

    if "in_table" not in summary_df_full.columns or speed_col not in summary_df_full.columns:
        return pd.DataFrame()

    out = summary_df_full[_series_to_bool(summary_df_full["in_table"])].copy()
    out[speed_col] = pd.to_numeric(out[speed_col], errors="coerce")
    out = out.dropna(subset=[speed_col])
    return out


def compute_table_hit_speed_mean_max(summary_df_full: pd.DataFrame, speed_col: str = "net_zone_max_speed_kmh") -> Dict:
    """Compute final mean/max speed using only balls that landed on the table."""
    speed_df = get_table_hit_speed_df_for_analysis(summary_df_full, speed_col=speed_col)
    if speed_df.empty:
        return {
            "table_hit_speed_count": 0,
            "mean_speed_kmh": None,
            "max_speed_kmh": None,
            "max_speed_stroke_id": None,
        }

    max_idx = speed_df[speed_col].idxmax()
    return {
        "table_hit_speed_count": int(len(speed_df)),
        "mean_speed_kmh": float(speed_df[speed_col].mean()),
        "max_speed_kmh": float(speed_df.loc[max_idx, speed_col]),
        "max_speed_stroke_id": int(speed_df.loc[max_idx, "stroke_id"]) if "stroke_id" in speed_df.columns else None,
    }

def keep_columns(df: pd.DataFrame, keep_cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for col in keep_cols:
        if col not in out.columns:
            out[col] = ""
    return out[keep_cols]



LANDING_MERGE_COLS = [
    "bounce_frame",
    "ball_px",
    "ball_py",
    "x_cm",
    "y_cm",
    "zone_col",
    "zone_row",
    "zone_label",
    "in_table",
    "in_table_strict",
    "in_table_relaxed",
    "edge_bounce",
    "right_half_landing",
    "bounce_type",
    "ball_py_smooth",
    "drop_before_px",
    "rise_after_px",
    "pre_slope_px_per_frame",
    "post_slope_px_per_frame",
    "piecewise_rmse_px",
    "single_rmse_px",
    "model_improvement_px",
    "bounce_score",
]


def merge_landing_results(summary_df_full: pd.DataFrame, df_land: pd.DataFrame) -> pd.DataFrame:
    """Merge piecewise landing results back to the stroke summary."""
    out = summary_df_full.copy()

    if df_land is None or df_land.empty:
        for col in LANDING_MERGE_COLS:
            if col not in out.columns:
                out[col] = ""
        return out

    merge_cols = ["stroke_id"] + [c for c in LANDING_MERGE_COLS if c in df_land.columns]
    out = out.merge(df_land[merge_cols], on="stroke_id", how="left", suffixes=("", "_landing"))

    if "bounce_frame_landing" in out.columns:
        out["bounce_frame"] = out["bounce_frame_landing"].fillna(out["bounce_frame"]).fillna(0).astype(int)
        out.drop(columns=["bounce_frame_landing"], inplace=True)

    for col in LANDING_MERGE_COLS:
        landing_col = f"{col}_landing"
        if landing_col in out.columns:
            if col in out.columns:
                out[col] = out[landing_col].combine_first(out[col])
            else:
                out[col] = out[landing_col]
            out.drop(columns=[landing_col], inplace=True)
        elif col not in out.columns:
            out[col] = ""

    return out


def sync_bounce_frames_to_strokes(strokes: List[Dict], summary_df_full: pd.DataFrame) -> None:
    """Update strokes list in-place so visual video can draw updated fields."""
    if summary_df_full is None or summary_df_full.empty:
        return

    row_map = {}
    for _, row in summary_df_full.iterrows():
        try:
            row_map[int(row["stroke_id"])] = row
        except Exception:
            continue

    for stroke in strokes:
        sid = int(stroke.get("stroke_id", -1))
        if sid not in row_map:
            continue

        row = row_map[sid]

        if "bounce_frame" in row:
            try:
                stroke["bounce_frame"] = int(row.get("bounce_frame", 0) or 0)
            except Exception:
                pass

        if "valid" in row:
            try:
                stroke["valid"] = int(row.get("valid", stroke.get("valid", 1)) or 0)
            except Exception:
                pass

        if "note" in row:
            try:
                note_value = row.get("note", "")
                stroke["note"] = "" if pd.isna(note_value) else str(note_value)
            except Exception:
                pass

def process_single_video(
    video_file,
    ball_csv,
    save_dir,
    min_left_segments=5,
    min_candidate_frames=50,
    min_no_hit_candidate_frames=20,
    max_step_th=300.0,
    max_abs_dy_th=45.0,
    left_half_ratio=0.35,
    right_side_ratio=0.6,
    fps=120.0,
    frame_w=1920,
    frame_h=1080,
    save_video=False,
    video_codec="h264_nvenc",
    helper_table_json=None,
    near_dist=NEAR_NET_DIST,
    box_height=BOX_HEIGHT,
    use_height_plane_scale=False,
    plane_height_cm=DEFAULT_PLANE_HEIGHT_CM,
    camera_focal_scale=DEFAULT_CAMERA_FOCAL_SCALE,
    save_height_debug=False,
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

    if helper_table_json is None:
        raise ValueError("helper_table_json is required. Run helper_table.py first to create *_helper_table.json.")
    if not os.path.exists(helper_table_json):
        raise FileNotFoundError(f"helper_table_json not found: {helper_table_json}")

    corners_lf_rf_rb_lb = load_table_corners(helper_table_json)
    frame_shape = (frame_reader.height, frame_reader.width, 3)
    helper_zone_info = build_zone_info_from_helper_table(
        corners_lf_rf_rb_lb=corners_lf_rf_rb_lb,
        frame_shape=frame_shape,
        near_dist=near_dist,
        box_height=box_height,
    )
    print(f"[INFO] use helper_table geometry: {helper_table_json}")
    table_dbg, net_dbg = zone_info_to_arrays(helper_zone_info)
    print("[INFO] table order for cm mapping: p1=NL, p2=FL, p3=FR, p4=NR")
    print(f"[INFO] helper table_p1~p4:\n{table_dbg}")

    camera_model = estimate_camera_model_from_table(
        table_dbg,
        frame_reader.width,
        frame_reader.height,
        focal_scale=camera_focal_scale,
    )

    if use_height_plane_scale:
        if camera_model is None:
            print("[WARN] failed to estimate approximate camera model. Fallback to original y-based speed only.")
        else:
            print(f"[INFO] blue baseline + orange x-ratio + blue local sy correction enabled: plane_height_cm={plane_height_cm}, focal_scale={camera_focal_scale}")    

    try:
        strokes = detect_strokes_from_runs(
            df=df,
            frame_w=frame_reader.width,
            min_left_segments=min_left_segments,
            min_candidate_frames=min_candidate_frames,
            min_no_hit_candidate_frames=min_no_hit_candidate_frames,
            max_step_th=max_step_th,
            max_abs_dy_th=max_abs_dy_th,
            left_half_ratio=left_half_ratio,
            right_side_ratio=right_side_ratio,
        )

        summary_df_full = build_stroke_summary_csv(
            df=df,
            strokes=strokes,
            fps=frame_reader.fps,
            helper_zone_info=helper_zone_info,
            camera_model=camera_model,
            plane_height_cm=plane_height_cm,
            use_height_plane_scale=bool(use_height_plane_scale and camera_model is not None),
        )

        base = os.path.splitext(os.path.basename(video_file))[0] if has_video else strip_csv_suffix(ball_csv)

        # Landing module detects bounce_frame with the current piecewise trajectory method.
        # It writes <base>_landing_detail.csv, <base>_landing_heatmap.png,
        # <base>_landing_zones.png, and <base>_zone_stats.csv.
        landing_input_df = summary_df_full.copy()
        if "note" in landing_input_df.columns:
            landing_input_df = landing_input_df[
                ~landing_input_df["note"].fillna("").astype(str).str.split(";").apply(lambda parts: "no_hit" in parts)
            ].copy()

        df_land = landing.compute_landings_with_bounce(
            landing_input_df,
            df,
            save_dir=save_dir,
            base_name=base,
        )
        summary_df_full = merge_landing_results(summary_df_full, df_land)
        sync_bounce_frames_to_strokes(strokes, summary_df_full)

        # Keep all rows in every normal output.
        # If a later final analysis needs mean/max speed, use
        # compute_table_hit_speed_mean_max(summary_df_full) so only in_table=True
        # rows are included at calculation time.

        csv_path = os.path.join(save_dir, f"{base}_stroke_zone.csv")
        zone_detail_csv_path = os.path.join(save_dir, f"{base}_zone_detail.csv")
        speed_detail_csv_path = os.path.join(save_dir, f"{base}_net_zone_speed_detail.csv")
        video_path = os.path.join(save_dir, f"{base}_stroke_zone_visualize.mp4")

        build_export_stroke_csv(summary_df_full).to_csv(csv_path, index=False, encoding="utf-8-sig")
        build_export_zone_detail_csv(summary_df_full).to_csv(zone_detail_csv_path, index=False, encoding="utf-8-sig")
        build_export_speed_detail_csv(summary_df_full).to_csv(speed_detail_csv_path, index=False, encoding="utf-8-sig")

        if save_height_debug and has_video and use_height_plane_scale and camera_model is not None:
            save_height_debug_artifacts(video_file, df, summary_df_full, save_dir, base, camera_model, plane_height_cm)

        if save_video and has_video:
            draw_visual_video(video_file, df, strokes, summary_df_full, video_path, video_codec=video_codec)
        elif save_video and not has_video:
            print("[WARN] --save_video was set, but no mp4 was found. Skip visual video output.")
    finally:
        frame_reader.release()

    table_hit_speed_stats = compute_table_hit_speed_mean_max(summary_df_full)
    print(f"table-hit speed count : {table_hit_speed_stats['table_hit_speed_count']}")
    if table_hit_speed_stats["mean_speed_kmh"] is None:
        print("table-hit mean speed  : N/A")
        print("table-hit max speed   : N/A")
        print("table-hit max stroke  : N/A")
    else:
        print(f"table-hit mean speed  : {table_hit_speed_stats['mean_speed_kmh']:.2f} km/h")
        print(f"table-hit max speed   : {table_hit_speed_stats['max_speed_kmh']:.2f} km/h")
        print(f"table-hit max stroke  : {table_hit_speed_stats['max_speed_stroke_id']}")

    print(f"saved csv   : {csv_path}")
    print(f"saved zone  : {zone_detail_csv_path}")
    print(f"saved speed : {speed_detail_csv_path}")
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
    min_no_hit_candidate_frames=20,
    max_step_th=300.0,
    max_abs_dy_th=45.0,
    left_half_ratio=0.35,
    right_side_ratio=0.5,
    fps=120.0,
    frame_w=1920,
    frame_h=1080,
    save_video=False,
    video_codec="h264_nvenc",
    near_dist=NEAR_NET_DIST,
    box_height=BOX_HEIGHT,
    use_height_plane_scale=False,
    plane_height_cm=DEFAULT_PLANE_HEIGHT_CM,
    camera_focal_scale=DEFAULT_CAMERA_FOCAL_SCALE,
    save_height_debug=False,
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

        base_stem = strip_csv_suffix(ball_csv, csv_suffixes)
        helper_table_json = os.path.join(os.path.dirname(ball_csv), f"{base_stem}_helper_table.json")
        print(f"[BATCH] helper  : {helper_table_json}")

        try:
            if not os.path.exists(helper_table_json):
                raise FileNotFoundError(
                    f"helper_table_json not found: {helper_table_json}. "
                    "Run helper_table.py first to create this json."
                )
            process_single_video(
                video_file=video_file,
                ball_csv=ball_csv,
                save_dir=save_dir,
                min_left_segments=min_left_segments,
                min_candidate_frames=min_candidate_frames,
                min_no_hit_candidate_frames=min_no_hit_candidate_frames,
                max_step_th=max_step_th,
                max_abs_dy_th=max_abs_dy_th,
                left_half_ratio=left_half_ratio,
                right_side_ratio=right_side_ratio,
                fps=fps,
                frame_w=frame_w,
                frame_h=frame_h,
                save_video=save_video,
                video_codec=video_codec,
                helper_table_json=helper_table_json,
                near_dist=near_dist,
                box_height=box_height,
                use_height_plane_scale=use_height_plane_scale,
                plane_height_cm=plane_height_cm,
                camera_focal_scale=camera_focal_scale,
                save_height_debug=save_height_debug,
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
    parser.add_argument("--video_codec", type=str, default="h264_nvenc", choices=["h264_nvenc", "libx264"])
    parser.add_argument("--csv_suffixes", type=str, nargs="+", default=["_ball.csv", "_bass.csv"])
    parser.add_argument("--min_left_segments", type=int, default=5)
    parser.add_argument("--min_candidate_frames", type=int, default=50)
    parser.add_argument("--min_no_hit_candidate_frames", type=int, default=20)
    parser.add_argument("--max_step_th", type=float, default=300.0)
    parser.add_argument("--max_abs_dy_th", type=float, default=45.0)
    parser.add_argument("--left_half_ratio", type=float, default=0.35)
    parser.add_argument("--right_side_ratio", type=float, default=0.5)
    parser.add_argument("--helper_table_json", type=str, default=None)
    parser.add_argument("--near_dist", type=float, default=NEAR_NET_DIST)
    parser.add_argument("--box_height", type=float, default=BOX_HEIGHT)
    parser.add_argument("--fps", type=float, default=120.0)
    parser.add_argument("--frame_w", type=int, default=1920)
    parser.add_argument("--frame_h", type=int, default=1080)
    parser.add_argument("--use_height_plane_scale", action="store_true", help="Use blue baseline scale, orange/blue width ratio for sx, and rectified/original table-height ratio for sy.")
    parser.add_argument("--plane_height_cm", type=float, default=DEFAULT_PLANE_HEIGHT_CM)
    parser.add_argument("--camera_focal_scale", type=float, default=DEFAULT_CAMERA_FOCAL_SCALE)
    parser.add_argument("--save_height_debug", action="store_true", help="Save visual debug PNGs for the raised-plane correction.")
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
            min_no_hit_candidate_frames=args.min_no_hit_candidate_frames,
            max_step_th=args.max_step_th,
            max_abs_dy_th=args.max_abs_dy_th,
            left_half_ratio=args.left_half_ratio,
            right_side_ratio=args.right_side_ratio,
            fps=args.fps,
            frame_w=args.frame_w,
            frame_h=args.frame_h,
            save_video=args.save_video,
            video_codec=args.video_codec,
            near_dist=args.near_dist,
            box_height=args.box_height,
            use_height_plane_scale=args.use_height_plane_scale,
            plane_height_cm=args.plane_height_cm,
            camera_focal_scale=args.camera_focal_scale,
            save_height_debug=args.save_height_debug,
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
            min_no_hit_candidate_frames=args.min_no_hit_candidate_frames,
            max_step_th=args.max_step_th,
            max_abs_dy_th=args.max_abs_dy_th,
            left_half_ratio=args.left_half_ratio,
            right_side_ratio=args.right_side_ratio,
            fps=args.fps,
            frame_w=args.frame_w,
            frame_h=args.frame_h,
            save_video=args.save_video,
            video_codec=args.video_codec,
            helper_table_json=args.helper_table_json,
            near_dist=args.near_dist,
            box_height=args.box_height,
            use_height_plane_scale=args.use_height_plane_scale,
            plane_height_cm=args.plane_height_cm,
            camera_focal_scale=args.camera_focal_scale,
            save_height_debug=args.save_height_debug,
        )
        return

    raise ValueError("Please provide either --video_root or --ball_csv")


if __name__ == "__main__":
    main()
