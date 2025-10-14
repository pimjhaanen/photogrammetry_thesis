#!/usr/bin/env python3
"""
Render multi-view videos (front/side/top/3D) from the COMPLETE (synced) dataset only.

Features:
- Uses one CSV that already contains: frame_idx, marker_id, x, y, z, timestamp, utc, and telemetry.
- Fixed wireframe colors (LE yellow; struts 0..7 constant colors)
- Wireframe on/off toggle
- 2D/3D modes show Va/Depower/Steering/Tether Force below the plot
- "ALL" mode: 2×4 grid:
  Row1: [3D] [Top(+Sideslip)] [Video spans two cells]
  Row2: [Front] [Side(+AOA)] [Hemisphere(lat/long/r=tether)] [Metrics block]
- Video overlay shows UTC string from the dataset; video time is aligned by (timestamp - first_timestamp) + start offset.

Inputs set under __main__.
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
from typing import Optional, Tuple

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
    """Draw black-on-white (inv=False) or white-on-black (inv=True) boxed text."""
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
    va  = m.get("kite_measured_va")
    dep = m.get("kite_actual_depower")
    ste = m.get("kite_actual_steering")
    tf  = m.get("ground_tether_force")
    parts = []
    if pd.notna(va):  parts.append(f"Va: {va:.2f} m/s")
    if pd.notna(dep): parts.append(f"Depower: {dep:.2f} (-)")
    if pd.notna(ste): parts.append(f"Steering: {ste:.2f} (-)")
    if pd.notna(tf):  parts.append(f"Tether F: {tf:.0f} N")
    return "   |   ".join(parts)

def _get_video_frame(cap, t_sec, target_size=None, utc_text=None):
    """Grab frame at given time (sec) using POS_MSEC seek; overlay UTC if provided."""
    cap.set(cv2.CAP_PROP_POS_MSEC, t_sec * 1000.0)
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    if target_size is not None:
        tw, th = target_size
        frame = cv2.resize(frame, (tw, th), interpolation=cv2.INTER_AREA)
    # UTC overlay (white box bottom-left)
    if utc_text:
        h = frame.shape[0]
        _draw_cv_text_box(frame, utc_text, org=(10, h - 10), font_scale=0.8, thickness=2, inv=False)
    return frame

# ---------- simple smoothing by matching (no ArUco deps) ----------

def smooth_points_by_matching_local(points: np.ndarray,
                                    prev_points: Optional[np.ndarray],
                                    alpha: float = 0.9,
                                    max_dist: float = 0.35) -> Tuple[np.ndarray, np.ndarray]:
    """
    EMA smoothing by nearest-neighbor matching to previous points.
    y = alpha * prev + (1-alpha) * current, if matched within max_dist; else use current.
    Returns (smoothed_points, match_indices_of_prev or -1).
    """
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

# ---------- basic line fit & LE path (no external utils) ----------

def fit_line_3d_local(points: np.ndarray, num_samples: int = 60) -> np.ndarray:
    """
    Fit a 3D line via PCA first component and return a sampled segment along min/max projections.
    """
    if points.shape[0] < 2:
        return points
    P = points - points.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(P, full_matrices=False)
    dir_vec = Vt[0]  # principal direction
    t = P @ dir_vec
    tmin, tmax = t.min(), t.max()
    ts = np.linspace(tmin, tmax, num_samples)
    seg = points.mean(axis=0) + np.outer(ts, dir_vec)
    return seg

def le_shortest_path_3d_local(points: np.ndarray) -> np.ndarray:
    """
    Greedy nearest-neighbor path through the set of points.
    Start at point with smallest x then connect nearest unused.
    """
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

# ---------- panel renders ----------

def _render_points_panel(points, segments_map, view, xlim, ylim, zlim, show_wireframe=True,
                         annotate=None, output_size=None, dpi=200, base_fontsize=14, lw=2.2):
    """
    view: '3d', 'front', 'side', 'top'
    annotate: optional dict {'text': 'Sideslip: 3.2°'} -> placed top-right
    """
    rcParams.update({
        "font.size": base_fontsize,
        "axes.labelsize": base_fontsize,
        "axes.titlesize": base_fontsize,
        "lines.linewidth": lw,
        "xtick.labelsize": base_fontsize-2,
        "ytick.labelsize": base_fontsize-2,
    })

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
        return _matfig_to_bgr(fig, output_size=output_size, dpi=dpi)

    # 2D orthographic
    fig = plt.figure(figsize=(5,5), dpi=dpi)
    ax = fig.add_subplot(1,1,1)
    if view == "front":   # (y,z)
        xi, yi = 1, 2; xlab, ylab = "y (m)", "z (m)"; xlim_, ylim_ = ylim, zlim
    elif view == "side":  # (x,z)
        xi, yi = 0, 2; xlab, ylab = "x (m)", "z (m)"; xlim_, ylim_ = xlim, zlim
    elif view == "top":   # (x,y)
        xi, yi = 0, 1; xlab, ylab = "x (m)", "y (m)"; xlim_, ylim_ = xlim, ylim
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

    if annotate and "text" in annotate and annotate["text"]:
        ax.text(0.98, 0.98, annotate["text"], transform=ax.transAxes,
                ha="right", va="top", fontsize=base_fontsize-2,
                bbox=dict(facecolor="white", edgecolor="black", linewidth=0.5, alpha=0.7))

    return _matfig_to_bgr(fig, output_size=output_size, dpi=dpi)

def _render_hemisphere_panel(lat_deg, lon_deg, radius, output_size, dpi=200, base_fontsize=12):
    """
    Draw an upper hemisphere with a point at (lat, lon), at distance = radius (tether length).
    """
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

def _compose_all_frame(img_3d, img_top, img_front, img_side,
                       video_img, hemi_img, metrics_block,
                       out_size):
    """
    Compose 2 rows × 4 columns grid.
    Layout:
      Row1: [3D] [Top] [Video spans two cells]
      Row2: [Front] [Side] [Hemisphere] [Metrics]
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
    place(img_3d,   0, 0)
    place(img_top,  0, 1)
    place(video_img,0, 2, colspan=2)

    # Row 2
    place(img_front, 1, 0)
    place(img_side,  1, 1)
    place(hemi_img,  1, 2)
    place(metrics_block, 1, 3)

    return canvas

def _make_metrics_block(m, size=(480, 540)):
    """Render a simple stacked values panel (white background)."""
    w, h = size
    img = np.full((h, w, 3), 255, np.uint8)
    def fmt(v, unit="", prec=2):
        return f"{v:.{prec}f} {unit}".strip() if pd.notna(v) else "—"
    lines = []
    lines.append(f"UTC: {m.get('utc', '')}")
    lines.append(f"Va (m/s): {fmt(m.get('kite_measured_va'), prec=2)}")
    lines.append(f"Depower (-): {fmt(m.get('kite_actual_depower'), prec=2)}")
    lines.append(f"Steering (-): {fmt(m.get('kite_actual_steering'), prec=2)}")
    lines.append(f"Tether force (N): {fmt(m.get('ground_tether_force'), prec=0)}")
    lines.append(f"Reelout speed (m/s): {fmt(m.get('ground_tether_reelout_speed'), prec=2)}")
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
    # smoothing (no ArUco)
    smooth_points=True,
    point_alpha=0.9,
    match_radius=0.35,
    show_wireframe=True,
    # Axes limits
    x_lim=(-5,5), y_lim=(-3,3), z_lim=(3,8),
    # Video inputs for ALL mode
    input_video_path=None,
    input_video_start_s=0.0,
    # Sizes
    size_2d=(2400,800),  # wide so each 2D subplot is square
    size_3d=(1200,1200),
    size_all=(1920,1080),  # 2 rows × 4 cols grid
):
    """
    Render from a single complete dataset CSV that includes points and metrics.
    Required columns: frame_idx, marker_id, x, y, z, timestamp, utc.
    """
    df = pd.read_csv(dataset_csv)

    # Ensure required columns
    needed_pts = {"frame_idx", "marker_id", "x", "y", "z"}
    if not needed_pts.issubset(df.columns):
        raise ValueError(f"dataset_csv missing point columns: {needed_pts - set(df.columns)}")
    if "timestamp" not in df.columns:
        raise ValueError("dataset_csv missing 'timestamp' column")
    if "utc" not in df.columns:
        df["utc"] = ""  # not fatal

    # Make sure telemetry fields exist (fill NaN if absent)
    telemetry_cols = [
        "kite_measured_va", "kite_actual_depower", "kite_actual_steering",
        "ground_tether_force", "ground_tether_reelout_speed",
        "airspeed_angle_of_attack", "airspeed_sideslip_angle",
        "kite_0_latitude", "kite_0_longitude", "ground_tether_length"
    ]
    for c in telemetry_cols:
        if c not in df.columns:
            df[c] = np.nan

    # Unique frames
    unique_frames = np.sort(df["frame_idx"].unique().astype(int))

    # Per-frame metrics (take first occurrence)
    per_frame = df.sort_values("frame_idx").groupby("frame_idx", as_index=True).first()

    # Video capture for ALL mode
    cap = None
    first_ts = None
    if mode.lower() == "all":
        if input_video_path is None:
            raise ValueError("ALL mode requires input_video_path.")
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open input video: {input_video_path}")
        # establish first timestamp for time-zero alignment
        first_ts = per_frame["timestamp"].dropna().iloc[0] if not per_frame["timestamp"].dropna().empty else 0.0

    # Writer
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

        # Remove isolated points (nearest neighbor <= 0.8 m)
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

        # Optional smoothing
        if smooth_points and pts.size > 0:
            pts_smooth, _ = smooth_points_by_matching_local(
                pts, prev_points_smoothed, alpha=point_alpha, max_dist=match_radius
            )
            pts = pts_smooth
        prev_points_smoothed = pts.copy() if pts.size > 0 else None

        # Build wireframe segments dict (fixed colors)
        segments = {}
        fr["marker_str"] = fr["marker_id"].astype(str)
        for k in range(8):
            grp = fr[fr["marker_str"] == str(k)]
            if grp.shape[0] >= 2:
                segments[k] = fit_line_3d_local(grp[["x","y","z"]].values)
            else:
                segments[k] = None
        le_grp = fr[fr["marker_str"].str.upper() == "LE"]
        if le_grp.shape[0] >= 2:
            segments["LE"] = le_shortest_path_3d_local(le_grp[["x","y","z"]].values)
        else:
            segments["LE"] = None

        # Metrics row
        mrow = per_frame.loc[frame_idx] if frame_idx in per_frame.index else None
        metrics = mrow.to_dict() if isinstance(mrow, pd.Series) else {}

        mode_l = mode.lower()
        if mode_l in ("2d", "3d"):
            if mode_l == "3d":
                pane = _render_points_panel(pts, segments, "3d",
                                            x_lim, y_lim, z_lim, show_wireframe=show_wireframe,
                                            annotate=None, output_size=out_size)
            else:
                # 3 horizontal 2D panels (front, side, top)
                cw = out_size[0]//3
                ch = out_size[1]
                img_front = _render_points_panel(pts, segments, "front", x_lim, y_lim, z_lim,
                                                 show_wireframe=show_wireframe, annotate=None, output_size=(cw, ch))
                img_side  = _render_points_panel(pts, segments, "side",  x_lim, y_lim, z_lim,
                                                 show_wireframe=show_wireframe, annotate=None, output_size=(cw, ch))
                img_top   = _render_points_panel(pts, segments, "top",   x_lim, y_lim, z_lim,
                                                 show_wireframe=show_wireframe, annotate=None, output_size=(cw, ch))
                pane = np.hstack([img_front, img_side, img_top])

            # Add metrics line under plot
            txt = _format_metrics_line(metrics)
            _draw_cv_text_box(pane, txt, org=(10, pane.shape[0]-10), font_scale=0.8, thickness=2, inv=False)
            writer.write(pane)
            frame_counter += 1
            continue

        # ======= "ALL" mode =======
        cw, ch = (size_all[0]//4, size_all[1]//2)
        # annotations
        sideslip_text = ""
        aoa_text = ""
        if metrics:
            ss = metrics.get("airspeed_sideslip_angle")
            if pd.notna(ss): sideslip_text = f"Sideslip: {ss:.1f}°"
            aoa = metrics.get("airspeed_angle_of_attack")
            if pd.notna(aoa): aoa_text = f"AOA: {aoa:.1f}°"

        img_3d = _render_points_panel(pts, segments, "3d",
                                      x_lim, y_lim, z_lim, show_wireframe=show_wireframe,
                                      annotate=None, output_size=(cw, ch))
        img_top = _render_points_panel(pts, segments, "top",
                                       x_lim, y_lim, z_lim, show_wireframe=show_wireframe,
                                       annotate={"text": sideslip_text} if sideslip_text else None,
                                       output_size=(cw, ch))
        img_front = _render_points_panel(pts, segments, "front",
                                         x_lim, y_lim, z_lim, show_wireframe=show_wireframe,
                                         annotate=None, output_size=(cw, ch))
        img_side  = _render_points_panel(pts, segments, "side",
                                         x_lim, y_lim, z_lim, show_wireframe=show_wireframe,
                                         annotate={"text": aoa_text} if aoa_text else None,
                                         output_size=(cw, ch))

        # Video time from timestamps
        utc_overlay = metrics.get("utc", "")
        t_curr = metrics.get("timestamp", np.nan)
        if pd.isna(t_curr) or first_ts is None:
            t_video = input_video_start_s + frame_counter*(1.0/fps)
        else:
            t_video = input_video_start_s + float(t_curr - first_ts)
        video_img = _get_video_frame(cap, t_video, target_size=(cw*2, ch), utc_text=utc_overlay)

        # Hemisphere panel
        lat  = metrics.get("kite_0_latitude", np.nan)
        lon  = metrics.get("kite_0_longitude", np.nan)
        rlen = metrics.get("ground_tether_length", np.nan)
        hemi_img = _render_hemisphere_panel(lat, lon, rlen, output_size=(cw, ch))

        # Metrics block
        metrics_block = _make_metrics_block(metrics, size=(cw, ch))

        composite = _compose_all_frame(img_3d, img_top, img_front, img_side,
                                       video_img, hemi_img, metrics_block, out_size=size_all)
        writer.write(composite)
        frame_counter += 1

    writer.release()
    if cap is not None:
        cap.release()
    print(f"[✓] Total frames written: {frame_counter} -> {output_video}")


# ============================ Script main ============================
if __name__ == "__main__":
    # === USER INPUTS ===
    # Complete/synced dataset (must include points + metrics)
    DATASET_CSV      = r"output/09_10_downloop_218_complete_dataset.csv"

    # Output
    OUTPUT_VIDEO     = r"output/09_10_downloop_218_all_no_wf.mp4"
    MODE             = "all"     # "2d" | "3d" | "all"
    FPS              = 30

    # Smoothing (no ArUco)
    SMOOTH_POINTS    = True
    POINT_ALPHA      = 0.95
    MATCH_RADIUS     = 0.35

    # Wireframe toggle + fixed colors
    SHOW_WIREFRAME   = False

    # Axes limits
    X_LIM = (-5, 5)
    Y_LIM = (-3, 3)
    Z_LIM = (3, 8)

    # Video (ALL mode)
    INPUT_VIDEO_PATH = r"Photogrammetry/input/left_videos/09_10_merged.MP4"
    INPUT_VIDEO_START_S = 218  # where to start the input video (sec)

    # Sizes
    SIZE_2D = (2400, 800)   # 3 panels in one row
    SIZE_3D = (1200, 1200)
    SIZE_ALL= (1920, 1080)  # 2×4 grid

    render_video(
        dataset_csv=DATASET_CSV,
        output_video=OUTPUT_VIDEO,
        mode=MODE,
        fps=FPS,
        smooth_points=SMOOTH_POINTS,
        point_alpha=POINT_ALPHA,
        match_radius=MATCH_RADIUS,
        show_wireframe=SHOW_WIREFRAME,
        x_lim=X_LIM, y_lim=Y_LIM, z_lim=Z_LIM,
        input_video_path=INPUT_VIDEO_PATH,
        input_video_start_s=INPUT_VIDEO_START_S,
        size_2d=SIZE_2D, size_3d=SIZE_3D, size_all=SIZE_ALL,
    )
