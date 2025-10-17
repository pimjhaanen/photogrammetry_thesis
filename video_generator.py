#!/usr/bin/env python3
"""
Render multi-view videos (front/side/bottom/3D) from the COMPLETE (synced) dataset only.

Changes in this version:
- ALL layout (2×4):
  TOP:   [3D] [Bottom(y,x)] [Front(z,x)] [Side(x,z)]
  BOTTOM:[Video(180°, spans 2 cells)] [Hemisphere] [Metrics]
- Front view uses axes (z,x); Bottom view uses (y,x).
- AOA shown on Front; Span (UWB) on Side; Sideslip on Bottom.
- Video overlay shows 'UTC: …'.
- Span formatted to ≥3 decimals.
- Tether force displayed in kg; zero-phase telemetry filtering retained.
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from typing import Optional, Tuple, List

G0 = 9.80665  # N per kg

# ============================ Fixed colors ============================
COLORS = {
    "LE": "#FFD400",  # yellow
    0: "#1f77b4",
    1: "#ff7f0e",
    2: "#2ca02c",
    3: "#d62728",
    4: "#9467bd",
    5: "#8c564b",
    6: "#e377c2",
    7: "#7f7f7f",
}

# ============================ Small helpers ============================

def _ensure_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def _draw_cv_text_box(img, text, org, font_scale=0.7, thickness=2, inv=False):
    if not text:
        return img
    font = cv2.FONT_HERSHEY_SIMPLEX
    (w, h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = org
    pad = 5
    if inv:
        cv2.rectangle(img, (x, y - h - pad), (x + w + 2*pad, y + baseline + pad), (0,0,0), -1)
        cv2.putText(img, text, (x + pad, y), font, font_scale, (255,255,255), thickness, cv2.LINE_AA)
    else:
        cv2.rectangle(img, (x, y - h - pad), (x + w + 2*pad, y + baseline + pad), (255,255,255), -1)
        cv2.putText(img, text, (x + pad, y), font, font_scale, (0,0,0), thickness, cv2.LINE_AA)
    return img

def _matfig_to_bgr(fig, output_size=None, dpi=200):
    fig.tight_layout()
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if output_size is not None and (bgr.shape[1], bgr.shape[0]) != output_size:
        interp = cv2.INTER_AREA if (output_size[0] < bgr.shape[1]) else cv2.INTER_LANCZOS4
        bgr = cv2.resize(bgr, output_size, interpolation=interp)
    return bgr

def _format_metrics_line(m):
    if m is None:
        return ""
    va  = m.get("kite_measured_va_filt", m.get("kite_measured_va"))
    dep = m.get("kite_actual_depower_filt", m.get("kite_actual_depower"))
    ste = m.get("kite_actual_steering_filt", m.get("kite_actual_steering"))
    tfN = m.get("ground_tether_force_filt", m.get("ground_tether_force"))
    tfkg = (tfN / G0) if pd.notna(tfN) else np.nan
    parts = []
    if pd.notna(va):   parts.append(f"Va: {va:.2f} m/s")
    if pd.notna(dep):  parts.append(f"Depower: {dep:.2f} (-)")
    if pd.notna(ste):  parts.append(f"Steering: {ste:.2f} (-)")
    if pd.notna(tfkg): parts.append(f"Tether: {tfkg:.1f} kg")
    return "   |   ".join(parts)

def _get_video_frame(cap, t_sec, target_size=None, utc_text=None, rotate_180=False):
    cap.set(cv2.CAP_PROP_POS_MSEC, t_sec * 1000.0)
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    if rotate_180:
        frame = cv2.rotate(frame, cv2.ROTATE_180)
    if target_size is not None:
        tw, th = target_size
        frame = cv2.resize(frame, (tw, th), interpolation=cv2.INTER_AREA)
    if utc_text:
        h = frame.shape[0]
        _draw_cv_text_box(frame, f"UTC: {utc_text}", org=(10, h - 10), font_scale=0.8, thickness=2, inv=False)
    return frame

# ---------- simple smoothing by matching (for points) ----------

def smooth_points_by_matching_local(points: np.ndarray,
                                    prev_points: Optional[np.ndarray],
                                    alpha: float = 0.9,
                                    max_dist: float = 0.35) -> Tuple[np.ndarray, np.ndarray]:
    if points is None or len(points) == 0:
        return points, np.array([], dtype=int)
    pts = points.copy()
    if prev_points is None or len(prev_points) == 0:
        return pts, -np.ones(len(pts), dtype=int)

    match_idx = -np.ones(len(pts), dtype=int)
    prev_used = np.zeros(len(prev_points), dtype=bool)
    for i, p in enumerate(pts):
        d = np.linalg.norm(prev_points - p, axis=1)
        j = int(np.argmin(d))
        if d[j] <= max_dist and not prev_used[j]:
            pts[i] = alpha * prev_points[j] + (1.0 - alpha) * p
            match_idx[i] = j
            prev_used[j] = True
    return pts, match_idx

# ---------- basic line fit & LE path ----------

def fit_line_3d_local(points: np.ndarray, num_samples: int = 60) -> np.ndarray:
    if points.shape[0] < 2:
        return points
    P = points - points.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(P, full_matrices=False)
    dir_vec = Vt[0]
    t = P @ dir_vec
    ts = np.linspace(t.min(), t.max(), num_samples)
    seg = points.mean(axis=0) + np.outer(ts, dir_vec)
    return seg

def le_shortest_path_3d_local(points: np.ndarray) -> np.ndarray:
    if points.shape[0] < 2:
        return points
    pts = points.copy()
    n = len(pts)
    used = np.zeros(n, dtype=bool)
    start = int(np.argmin(pts[:, 0]))
    path = [start]
    used[start] = True
    curr = start
    for _ in range(n - 1):
        d = np.linalg.norm(pts - pts[curr], axis=1)
        d[used] = np.inf
        j = int(np.argmin(d))
        if not np.isfinite(d[j]):
            break
        path.append(j)
        used[j] = True
        curr = j
    return pts[path]

# ---------- zero-phase EMA for telemetry (no lag) ----------

def _ema_pass(x: np.ndarray, alpha: float) -> np.ndarray:
    y = np.empty_like(x, dtype=float); y[:] = np.nan
    idx = np.where(np.isfinite(x))[0]
    if len(idx) == 0: return y
    i0 = idx[0]; y[i0] = x[i0]
    for i in range(i0 + 1, len(x)):
        xi = x[i]
        y[i] = (alpha * y[i-1] + (1.0 - alpha) * xi) if np.isfinite(xi) else y[i-1]
    return y

def zero_phase_ema(series: pd.Series, alpha: float) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    if x.size == 0: return series.astype(float)
    fwd = _ema_pass(x, alpha)
    bwd = _ema_pass(fwd[::-1], alpha)[::-1]
    return pd.Series(bwd, index=series.index, dtype=float)

def apply_zero_phase_to_per_frame(per_frame: pd.DataFrame, columns: List[str],
                                  alpha: float, suffix: str = "_filt") -> pd.DataFrame:
    for col in columns:
        if col in per_frame.columns:
            per_frame[col + suffix] = zero_phase_ema(per_frame[col], alpha)
    return per_frame

# ---------- panel renders ----------

def _render_points_panel(points, segments_map, view, xlim, ylim, zlim, show_wireframe=True,
                         annotate=None, output_size=None, dpi=200, base_fontsize=14, lw=2.2):
    """
    view:
      - '3d'
      - 'front'     -> (y, z)
      - 'side'      -> (x, z)
      - 'bottom'    -> (x, y)
      - 'side_yz'   -> (y, z)     # requested side view (y horizontal, z vertical)
      - 'front_xz'  -> (x, z)     # alias so we can set titles/flip separately
    annotate: dict with optional keys:
      - 'text':     overlay text (top-right)
      - 'title':    title above axes
      - 'invert_x': bool to invert x-axis (mirror left/right)
    """
    rcParams.update({
        "font.size": base_fontsize,
        "axes.labelsize": base_fontsize,
        "axes.titlesize": base_fontsize,
        "lines.linewidth": lw,
        "xtick.labelsize": base_fontsize-2,
        "ytick.labelsize": base_fontsize-2,
    })

    ann = annotate or {}
    ann_text   = ann.get("text", "")
    ann_title  = ann.get("title", None)
    ann_flip_x = bool(ann.get("invert_x", False))

    if view == "3d":
        fig = plt.figure(figsize=(6,6), dpi=dpi)
        ax = fig.add_subplot(1,1,1, projection='3d')
        if show_wireframe:
            for k in range(8):
                seg = segments_map.get(k, None)
                if seg is not None and len(seg) > 0:
                    ax.plot(seg[:,0], seg[:,1], seg[:,2], color=COLORS[k])
            le = segments_map.get("LE", None)
            if le is not None and len(le) > 0:
                ax.plot(le[:,0], le[:,1], le[:,2], color=COLORS["LE"])
        if points is not None and len(points) > 0:
            ax.scatter(points[:,0], points[:,1], points[:,2], c='red', s=20)
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
        ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_zlim(*zlim)
        ax.view_init(elev=30, azim=-45)
        ax.grid(True, alpha=0.4)
        if ann_title: ax.set_title(ann_title)
        return _matfig_to_bgr(fig, output_size=output_size, dpi=dpi)

    # ---------- 2D orthographic ----------
    fig = plt.figure(figsize=(5,5), dpi=dpi)
    ax = fig.add_subplot(1,1,1)

    if view == "front":         # (y, z)
        xi, yi = 1, 2; xlab, ylab = "y (m)", "z (m)"; xlim_, ylim_ = ylim, zlim
    elif view == "side":        # (x, z)
        xi, yi = 0, 2; xlab, ylab = "x (m)", "z (m)"; xlim_, ylim_ = xlim, zlim
    elif view == "bottom":      # (x, y)
        xi, yi = 0, 1; xlab, ylab = "x (m)", "y (m)"; xlim_, ylim_ = xlim, ylim
    elif view == "side_yz":     # (y, z)  ← requested mapping
        xi, yi = 1, 2; xlab, ylab = "y (m)", "z (m)"; xlim_, ylim_ = ylim, zlim
    elif view == "front_xz":    # (x, z)  front with possible x-flip
        xi, yi = 0, 2; xlab, ylab = "x (m)", "z (m)"; xlim_, ylim_ = xlim, zlim
    else:
        raise ValueError("Unknown view")

    if show_wireframe:
        for k in range(8):
            seg = segments_map.get(k, None)
            if seg is not None and len(seg) > 0:
                ax.plot(seg[:, xi], seg[:, yi], color=COLORS[k])
        le = segments_map.get("LE", None)
        if le is not None and len(le) > 0:
            ax.plot(le[:, xi], le[:, yi], color=COLORS["LE"])
    if points is not None and len(points) > 0:
        ax.scatter(points[:, xi], points[:, yi], c='red', s=20)

    ax.set_xlabel(xlab); ax.set_ylabel(ylab)
    ax.set_xlim(*xlim_); ax.set_ylim(*ylim_)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.4)

    if ann_flip_x:
        ax.invert_xaxis()
    if ann_title:
        ax.set_title(ann_title)
    if ann_text:
        ax.text(0.98, 0.98, ann_text, transform=ax.transAxes,
                ha="right", va="top", fontsize=base_fontsize-2,
                bbox=dict(facecolor="white", edgecolor="black", linewidth=0.5, alpha=0.7))

    return _matfig_to_bgr(fig, output_size=output_size, dpi=dpi)



def _render_hemisphere_panel(lat_deg, lon_deg, radius, output_size, dpi=200, base_fontsize=12):
    lat = float(lat_deg) if pd.notna(lat_deg) else 0.0
    lon = float(lon_deg) if pd.notna(lon_deg) else 0.0
    r   = float(radius) if pd.notna(radius) else 1.0
    r_sphere = max(r, 1.0)

    rcParams.update({"font.size": base_fontsize})
    fig = plt.figure(figsize=(5,5), dpi=dpi)
    ax = fig.add_subplot(1,1,1, projection='3d')

    u = np.linspace(0, 2*np.pi, 40)
    v = np.linspace(0, np.pi/2, 20)
    xs = r_sphere * np.outer(np.cos(u), np.sin(v))
    ys = r_sphere * np.outer(np.sin(u), np.sin(v))
    zs = r_sphere * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(xs, ys, zs, rstride=1, cstride=1, color="#e6f2ff", edgecolor="none", alpha=0.9)

    lat_r = np.deg2rad(lat)
    lon_r = np.deg2rad(lon)
    px = r * np.cos(lat_r) * np.cos(lon_r)
    py = r * np.cos(lat_r) * np.sin(lon_r)
    pz = r * np.sin(lat_r)
    ax.scatter([px], [py], [pz], c="red", s=40)

    max_extent = r_sphere
    ax.set_xlim(-max_extent, max_extent)
    ax.set_ylim(-max_extent, max_extent)
    ax.set_zlim(0, max_extent)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.view_init(elev=25, azim=-45)
    ax.grid(True, alpha=0.2)

    return _matfig_to_bgr(fig, output_size=output_size, dpi=dpi)

def _compose_all_frame_top4(img_3d, img_bottom, img_front, img_side,
                            video_img, hemi_img, metrics_block,
                            out_size):
    """
    Compose 2 rows × 4 columns grid.
    Row1: [3D] [Bottom] [Front] [Side]
    Row2: [Video (colspan=2)] [Hemisphere] [Metrics]
    """
    W, H = out_size
    cw, ch = W // 4, H // 2
    canvas = np.full((H, W, 3), 255, np.uint8)

    def place(img, row, col, colspan=1, rowspan=1):
        x0 = col * cw
        y0 = row * ch
        w = cw * colspan
        h = ch * rowspan
        if img is None:
            return
        ih, iw = img.shape[:2]
        scale = min(w/iw, h/ih)
        new_w, new_h = int(iw*scale), int(ih*scale)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA if scale<1 else cv2.INTER_LANCZOS4)
        x = x0 + (w - new_w)//2
        y = y0 + (h - new_h)//2
        canvas[y:y+new_h, x:x+new_w] = resized

    # Row 1
    place(img_3d,     0, 0)
    place(img_bottom, 0, 1)
    place(img_front,  0, 2)
    place(img_side,   0, 3)

    # Row 2  ← video spans two blocks now
    place(video_img,     1, 0, colspan=2)
    place(hemi_img,      1, 2)
    place(metrics_block, 1, 3)

    return canvas


def _make_metrics_block(m, size=(480, 540)):
    w, h = size
    img = np.full((h, w, 3), 255, np.uint8)
    def fmt(v, unit="", prec=2):
        return f"{v:.{prec}f} {unit}".strip() if pd.notna(v) else "—"
    va  = m.get('kite_measured_va_filt', m.get('kite_measured_va'))
    dep = m.get('kite_actual_depower_filt', m.get('kite_actual_depower'))
    ste = m.get('kite_actual_steering_filt', m.get('kite_actual_steering'))
    tfN = m.get('ground_tether_force_filt', m.get('ground_tether_force'))
    tfkg = (tfN / G0) if pd.notna(tfN) else np.nan
    ro  = m.get('ground_tether_reelout_speed_filt', m.get('ground_tether_reelout_speed'))

    lines = [
        f"Va (m/s): {fmt(va, prec=2)}",
        f"Depower (-): {fmt(dep, prec=2)}",
        f"Steering (-): {fmt(ste, prec=2)}",
        f"Tether force (kg): {fmt(tfkg, prec=1)}",
        f"Reelout speed (m/s): {fmt(ro, prec=2)}",
    ]
    y = 60
    for s in lines:
        _draw_cv_text_box(img, s, org=(20, y), font_scale=0.8, thickness=2, inv=False)
        y += 70
    return img

# ============================ Main render routine ============================

def render_video(
    dataset_csv,
    output_video="output/out.mp4",
    mode="2d",                # "2d" | "3d" | "all"
    fps=30,
    # smoothing for points (frame-to-frame)
    smooth_points=True,
    point_alpha=0.9,
    match_radius=0.35,
    show_wireframe=True,
    # telemetry smoothing (zero-phase EMA)
    telemetry_zero_phase_alpha=0.95,
    # Axes limits
    x_lim=(-5,5), y_lim=(-3,3), z_lim=(3,8),
    # Video (ALL mode)
    input_video_path=None,
    input_video_start_s=0.0,
    # Sizes
    size_2d=(2400,800),
    size_3d=(1200,1200),
    size_all=(1920,1080),
):
    """
    Render from a single complete dataset CSV that includes points and metrics.
    Required columns: frame_idx, marker_id, x, y, z, timestamp, utc.
    """
    df = pd.read_csv(dataset_csv)

    needed_pts = {"frame_idx", "marker_id", "x", "y", "z"}
    if not needed_pts.issubset(df.columns):
        raise ValueError(f"dataset_csv missing point columns: {needed_pts - set(df.columns)}")
    if "timestamp" not in df.columns:
        raise ValueError("dataset_csv missing 'timestamp' column")
    if "utc" not in df.columns:
        df["utc"] = ""

    telemetry_cols = [
        "kite_measured_va", "kite_actual_depower", "kite_actual_steering",
        "ground_tether_force", "ground_tether_reelout_speed",
        "airspeed_angle_of_attack", "airspeed_sideslip_angle",
        "kite_0_latitude", "kite_0_longitude", "ground_tether_length",
        "uwb_distance_m"
    ]
    for c in telemetry_cols:
        if c not in df.columns:
            df[c] = np.nan

    unique_frames = np.sort(df["frame_idx"].unique().astype(int))
    per_frame = df.sort_values("frame_idx").groupby("frame_idx", as_index=True).first()

    # zero-phase (acausal) smoothing
    TELEMETRY_SMOOTH_COLS = [
        "kite_measured_va",
        "ground_tether_force",
        "ground_tether_reelout_speed",
        "kite_actual_depower",
        "kite_actual_steering",
        "airspeed_angle_of_attack",
        "airspeed_sideslip_angle",
        "uwb_distance_m",
    ]
    per_frame = apply_zero_phase_to_per_frame(
        per_frame, TELEMETRY_SMOOTH_COLS, alpha=telemetry_zero_phase_alpha, suffix="_filt"
    )

    cap = None
    first_ts = None
    if mode.lower() == "all":
        if input_video_path is None:
            raise ValueError("ALL mode requires input_video_path.")
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open input video: {input_video_path}")
        first_ts = per_frame["timestamp"].dropna().iloc[0] if not per_frame["timestamp"].dropna().empty else 0.0

    _ensure_dir(output_video)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_size = size_2d if mode.lower()=="2d" else (size_3d if mode.lower()=="3d" else size_all)
    writer = cv2.VideoWriter(output_video, fourcc, fps, out_size)

    prev_points_smoothed = None
    frame_counter = 0

    for frame_idx in unique_frames:
        fr = df[df["frame_idx"] == frame_idx]
        if fr.empty:
            continue

        pts_world = fr[["x","y","z"]].values.astype(float)

        # remove isolated points
        if pts_world.shape[0] >= 2:
            diffs = pts_world[:,None,:] - pts_world[None,:,:]
            D = np.linalg.norm(diffs, axis=2)
            np.fill_diagonal(D, np.inf)
            keep = D.min(axis=1) <= 0.8
            pts = pts_world[keep]
            fr = fr.loc[keep].copy()
        else:
            pts = pts_world
            fr = fr.copy()

        # optional smoothing of points
        if smooth_points and pts.size > 0:
            pts_smooth, _ = smooth_points_by_matching_local(
                pts, prev_points_smoothed, alpha=point_alpha, max_dist=match_radius
            )
            pts = pts_smooth
        prev_points_smoothed = pts.copy() if pts.size > 0 else None

        # wireframe
        segments = {}
        fr["marker_str"] = fr["marker_id"].astype(str)
        for k in range(8):
            grp = fr[fr["marker_str"] == str(k)]
            if grp.shape[0] >= 2:
                segments[k] = fit_line_3d_local(grp[["x","y","z"]].values)
            else:
                segments[k] = None
        le_grp = fr[fr["marker_str"].str.upper() == "LE"]
        segments["LE"] = le_shortest_path_3d_local(le_grp[["x","y","z"]].values) if le_grp.shape[0] >= 2 else None

        # metrics row
        mrow = per_frame.loc[frame_idx] if frame_idx in per_frame.index else None
        metrics = mrow.to_dict() if isinstance(mrow, pd.Series) else {}

        mode_l = mode.lower()
        if mode_l in ("2d", "3d"):
            if mode_l == "3d":
                pane = _render_points_panel(pts, segments, "3d",
                                            x_lim, y_lim, z_lim, show_wireframe=show_wireframe,
                                            annotate=None, output_size=out_size)
            else:
                cw = out_size[0]//3
                ch = out_size[1]
                # bottom: sideslip
                ss = metrics.get("airspeed_sideslip_angle_filt", metrics.get("airspeed_sideslip_angle"))
                bottom_txt = f"Sideslip: {ss:.1f}°" if pd.notna(ss) else ""
                img_bottom = _render_points_panel(pts, segments, "bottom", x_lim, y_lim, z_lim,
                                                  show_wireframe=show_wireframe,
                                                  annotate={"text": bottom_txt} if bottom_txt else None,
                                                  output_size=(cw, ch))
                # front (z,x): AOA
                aoa = metrics.get("airspeed_angle_of_attack_filt", metrics.get("airspeed_angle_of_attack"))
                front_txt = f"AOA: {aoa:.1f}°" if pd.notna(aoa) else ""
                img_front = _render_points_panel(pts, segments, "front", x_lim, y_lim, z_lim,
                                                 show_wireframe=show_wireframe,
                                                 annotate={"text": front_txt} if front_txt else None,
                                                 output_size=(cw, ch))
                # side (x,z): Span (UWB) with ≥3 decimals
                uwb = metrics.get("uwb_distance_m_filt", metrics.get("uwb_distance_m"))
                side_txt = f"Span (UWB): {uwb:.3f} m" if pd.notna(uwb) else ""
                img_side  = _render_points_panel(pts, segments, "side",  x_lim, y_lim, z_lim,
                                                 show_wireframe=show_wireframe,
                                                 annotate={"text": side_txt} if side_txt else None,
                                                 output_size=(cw, ch))
                pane = np.hstack([img_bottom, img_front, img_side])

            txt = _format_metrics_line(metrics)
            _draw_cv_text_box(pane, txt, org=(10, pane.shape[0]-10), font_scale=0.8, thickness=2, inv=False)
            writer.write(pane)
            frame_counter += 1
            continue

        # ======= "ALL" mode =======
        cw, ch = (size_all[0] // 4, size_all[1] // 2)

        # annotations (Span with 3 decimals)
        uwb = metrics.get("uwb_distance_m_filt", metrics.get("uwb_distance_m"))
        span_txt = f"Span (UWB): {uwb:.3f} m" if pd.notna(uwb) else ""
        aoa = metrics.get("airspeed_angle_of_attack_filt", metrics.get("airspeed_angle_of_attack"))
        aoa_txt = f"AOA: {aoa:.1f}°" if pd.notna(aoa) else ""
        ss = metrics.get("airspeed_sideslip_angle_filt", metrics.get("airspeed_sideslip_angle"))
        sideslip_txt = f"Sideslip: {ss:.1f}°" if pd.notna(ss) else ""

        # Row 1 (left→right): 3D | Bottom(x,y, flip x) | Side(y,z) | Front(x,z, flip x)
        img_3d = _render_points_panel(
            pts, segments, "3d",
            x_lim, y_lim, z_lim, show_wireframe=show_wireframe,
            annotate={"title": None},
            output_size=(cw, ch)
        )

        img_bottom = _render_points_panel(
            pts, segments, "bottom",
            x_lim, y_lim, z_lim, show_wireframe=show_wireframe,
            annotate={"title": "Bottom view", "invert_x": True, "text": sideslip_txt},
            output_size=(cw, ch)
        )

        # SIDE view now uses (y,z) with title "Side view" and shows SPAN here
        img_side = _render_points_panel(
            pts, segments, "side_yz",
            x_lim, y_lim, z_lim, show_wireframe=show_wireframe,
            annotate={"title": "Side view", "text": span_txt},
            output_size=(cw, ch)
        )

        # FRONT view (x,z) with x flip & AOA
        img_front = _render_points_panel(
            pts, segments, "front_xz",
            x_lim, y_lim, z_lim, show_wireframe=show_wireframe,
            annotate={"title": "Front view", "invert_x": True, "text": aoa_txt},
            output_size=(cw, ch)
        )

        # Video timing & frame (make it sized for two cells: cw*2 × ch)
        utc_overlay = metrics.get("utc", "")
        t_curr = metrics.get("timestamp", np.nan)
        if pd.isna(t_curr) or first_ts is None:
            t_video = input_video_start_s + frame_counter * (1.0 / fps)
        else:
            t_video = input_video_start_s + float(t_curr - first_ts)
        video_img = _get_video_frame(cap, t_video, target_size=(cw * 2, ch),
                                     utc_text=utc_overlay, rotate_180=True)

        # Hemisphere and metrics
        lat = metrics.get("kite_0_latitude", np.nan)
        lon = metrics.get("kite_0_longitude", np.nan)
        rlen = metrics.get("ground_tether_length", np.nan)
        hemi_img = _render_hemisphere_panel(lat, lon, rlen, output_size=(cw, ch))
        metrics_block = _make_metrics_block(metrics, size=(cw, ch))

        composite = _compose_all_frame_top4(
            img_3d, img_bottom, img_front, img_side,
            video_img, hemi_img, metrics_block, out_size=size_all
        )

        writer.write(composite)

        frame_counter += 1
        if frame_counter % 30 == 0:
            seconds_written = frame_counter / 30
            print(f"Written {seconds_written} seconds to video")

    writer.release()
    if cap is not None:
        cap.release()
    print(f"[✓] Total frames written: {frame_counter} -> {output_video}")


# ============================ Script main ============================
if __name__ == "__main__":
    # === USER INPUTS ===
    DATASET_CSV      = r"output/09_10_downloop_218_complete_dataset.csv"

    OUTPUT_VIDEO     = r"output/09_10_downloop_218_all.mp4"
    MODE             = "all"     # "2d" | "3d" | "all"
    FPS              = 30

    SMOOTH_POINTS    = True
    POINT_ALPHA      = 0.95
    MATCH_RADIUS     = 0.35

    SHOW_WIREFRAME   = True
    TELEMETRY_ALPHA  = 0.95

    X_LIM = (-5, 5)
    Y_LIM = (-3, 3)
    Z_LIM = (3, 8)

    INPUT_VIDEO_PATH = r"Photogrammetry/input/left_videos/09_10_merged.MP4"
    INPUT_VIDEO_START_S = 218

    SIZE_2D = (2400, 800)
    SIZE_3D = (1200, 1200)
    SIZE_ALL= (1920, 1080)

    render_video(
        dataset_csv=DATASET_CSV,
        output_video=OUTPUT_VIDEO,
        mode=MODE,
        fps=FPS,
        smooth_points=SMOOTH_POINTS,
        point_alpha=POINT_ALPHA,
        match_radius=MATCH_RADIUS,
        show_wireframe=SHOW_WIREFRAME,
        telemetry_zero_phase_alpha=TELEMETRY_ALPHA,
        x_lim=X_LIM, y_lim=Y_LIM, z_lim=Z_LIM,
        input_video_path=INPUT_VIDEO_PATH,
        input_video_start_s=INPUT_VIDEO_START_S,
        size_2d=SIZE_2D, size_3d=SIZE_3D, size_all=SIZE_ALL,
    )
