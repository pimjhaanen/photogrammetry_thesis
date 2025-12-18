#!/usr/bin/env python3
"""
Render multi-view videos (front/side/bottom/3D) from the COMPLETE (synced) dataset only.

Updates in this version:
- LE reconstruction improved (lowest-Z endpoints; greedy+2opt), imported from utils.
- Reference frame computed from center struts (3 & 4); optional transform_to_local.
- Hemisphere panel REMOVED. New 2D panel shows LAT vs LON history (colored by tether length).
- Front view uses (z,x); Bottom uses (y,x). AOA on Front, Span(UWB) on Side, Sideslip on Bottom.
- Video overlay shows 'UTC: …'. Span formatted to ≥3 decimals. Tether force displayed in kg.
- Zero-phase telemetry filtering retained.

Panels in ALL mode (2×4):
  TOP:    [3D] [Bottom(y,x)] [Front(z,x)] [Side(x,z)]
  BOTTOM: [Video (spans 2 cells)] [Lat/Lon history] [Metrics]
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

# === import our utils (no ArUco) ===
from Photogrammetry.kite_shape_reconstruction_utils import (
    fit_line_3d, le_shortest_path_3d,
    compute_ref_frame_from_center_struts,
    smooth_points_by_matching,
)

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
    bg = (0,0,0) if inv else (255,255,255)
    fg = (255,255,255) if inv else (0,0,0)
    cv2.rectangle(img, (x, y - h - pad), (x + w + 2*pad, y + baseline + pad), bg, -1)
    cv2.putText(img, text, (x + pad, y), font, font_scale, fg, thickness, cv2.LINE_AA)
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


# ---------- zero-phase EMA (forward + backward, no lag) ----------

def _ema_pass(x: np.ndarray, a: float) -> np.ndarray:
    """One-direction EMA with NaN carry; starts at first finite sample."""
    y = np.empty_like(x, dtype=float)
    y[:] = np.nan
    idx = np.where(np.isfinite(x))[0]
    if len(idx) == 0:
        return y
    i0 = idx[0]
    y[i0] = x[i0]
    for i in range(i0 + 1, len(x)):
        xi = x[i]
        y[i] = (a * y[i - 1] + (1.0 - a) * xi) if np.isfinite(xi) else y[i - 1]
    return y


def _zero_phase_ema(x: np.ndarray, a: float) -> np.ndarray:
    """Zero-phase (forward + backward) EMA smoothing."""
    if x.size == 0:
        return x
    fwd = _ema_pass(x, a)
    bwd = _ema_pass(fwd[::-1], a)[::-1]
    return bwd


def zero_phase_ema(series: pd.Series, alpha: float) -> pd.Series:
    """Apply zero-phase EMA to a pandas Series."""
    x = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    if x.size == 0:
        return series.astype(float)
    filt = _zero_phase_ema(x, alpha)
    return pd.Series(filt, index=series.index, dtype=float)


def apply_zero_phase_to_per_frame(per_frame: pd.DataFrame, columns: List[str],
                                  alpha: float, suffix: str = "_filt") -> pd.DataFrame:
    """Apply zero-phase EMA to selected telemetry columns (in-place)."""
    for col in columns:
        if col in per_frame.columns:
            per_frame[col + suffix] = zero_phase_ema(per_frame[col], alpha)
    return per_frame


# ---------- panels ----------

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

    # ---------- 2D ----------
    fig = plt.figure(figsize=(5,5), dpi=dpi)
    ax = fig.add_subplot(1,1,1)

    if view == "front":         # (y, z)
        xi, yi = 1, 2; xlab, ylab = "y (m)", "z (m)"; xlim_, ylim_ = ylim, zlim
    elif view == "side":        # (x, z)
        xi, yi = 0, 2; xlab, ylab = "x (m)", "z (m)"; xlim_, ylim_ = xlim, zlim
    elif view == "bottom":      # (x, y)
        xi, yi = 0, 1; xlab, ylab = "x (m)", "y (m)"; xlim_, ylim_ = xlim, ylim
    elif view == "side_yz":     # (y, z)
        xi, yi = 1, 2; xlab, ylab = "y (m)", "z (m)"; xlim_, ylim_ = ylim, zlim
    elif view == "front_xz":    # (x, z)
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


def _render_position_history_panel(lon_hist, lat_hist, len_hist,
                                   output_size, dpi=200, base_fontsize=12):
    """2D LAT vs LON history; color by tether length (meters). Latest point highlighted."""
    rcParams.update({"font.size": base_fontsize})
    fig = plt.figure(figsize=(5,5), dpi=dpi)
    ax = fig.add_subplot(1,1,1)

    lon = np.array(lon_hist, dtype=float)
    lat = np.array(lat_hist, dtype=float)
    L   = np.array(len_hist, dtype=float)

    # mask NaNs
    mask = np.isfinite(lon) & np.isfinite(lat)
    lon, lat, L = lon[mask], lat[mask], L[mask]

    if lon.size >= 2:
        # draw trail
        ax.plot(lon, lat, linewidth=1.5, alpha=0.8)
    if lon.size >= 1:
        # color by tether length (normalize)
        if np.isfinite(L).any():
            c = (L - np.nanmin(L)) / (np.nanmax(L) - np.nanmin(L) + 1e-12)
        else:
            c = np.zeros_like(L)
        sc = ax.scatter(lon, lat, c=c, s=18, cmap="viridis", alpha=0.9)
        # latest point
        ax.scatter(lon[-1], lat[-1], s=60, edgecolor="black", facecolor="red", zorder=5)
        # colorbar
        cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("Tether length [m]")

    ax.set_xlabel("Longitude [deg]")
    ax.set_ylabel("Latitude [deg]")
    ax.grid(True, alpha=0.4)
    ax.set_title("Lat/Lon history (latest in red)")
    return _matfig_to_bgr(fig, output_size=output_size, dpi=dpi)


def _compose_all_frame_top4(img_3d, img_bottom, img_front, img_side,
                            video_img, poshist_img, metrics_block,
                            out_size):
    """
    Compose 2 rows × 4 columns grid.
    Row1: [3D] [Bottom] [Front] [Side]
    Row2: [Video (colspan=2)] [Lat/Lon history] [Metrics]
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

    # Row 2
    place(video_img,     1, 0, colspan=2)
    place(poshist_img,   1, 2)
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
    tfkg = m.get('ground_tether_force_filt', m.get('ground_tether_force'))
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


def spherical_to_cartesian(lat_deg, lon_deg, radius_m):
    """Convert (lat, lon, radius) to Cartesian (x,y,z)."""
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    x = radius_m * np.cos(lat) * np.cos(lon)
    y = radius_m * np.cos(lat) * np.sin(lon)
    z = radius_m * np.sin(lat)
    return np.column_stack((x, y, z))


def _render_spherical_panel(lat, lon, length, output_size=(360,360), dpi=120):
    """Render current spherical kite position (no history) in a compact form."""
    pts = spherical_to_cartesian(lat, lon, length)
    if len(pts) == 0:
        return np.full((output_size[1], output_size[0], 3), 255, np.uint8)

    # Smaller figure, lower dpi → more compact panel
    fig = plt.figure(figsize=(3.5,3.5), dpi=dpi)
    ax = fig.add_subplot(1,1,1, projection="3d")

    # Smaller marker size
    ax.scatter(pts[:,0], pts[:,1], pts[:,2], c="r", s=20)

    # Keep axis labels small
    ax.set_xlabel("X [m]", fontsize=8)
    ax.set_ylabel("Y [m]", fontsize=8)
    ax.set_zlabel("Z [m]", fontsize=8)

    # Same spatial limits
    ax.set_xlim(0,150)
    ax.set_ylim(-50,0)
    ax.set_zlim(0,300)

    # Smaller title
    ax.set_title("Kite position", fontsize=10)

    # Return resized BGR image
    return _matfig_to_bgr(fig, output_size=output_size, dpi=dpi)



# ============================ Main render routine ============================

def render_video(
    dataset_csv,
    output_video="output/out.mp4",
    mode="2d",                # "2d" | "3d" | "all"
    fps=30,
    # smoothing for points (frame-to-frame)
    smooth_points=True,
    point_alpha=0.95,
    match_radius=0.25,
    show_wireframe=True,
    # telemetry smoothing (zero-phase EMA)
    telemetry_zero_phase_alpha=0.95,
    # Axes limits (in whatever frame you plot: world or local)
    x_lim=(-5,5), y_lim=(-3,3), z_lim=(3,8),
    # Video (ALL mode)
    input_video_path=None,
    input_video_start_s=0.0,
    # Sizes
    size_2d=(2400,800),
    size_3d=(1200,1200),
    size_all=(1920,1080),
    # New: coordinate frame options
    transform_to_local=True,
    force_flip_x=False,
):
    """
    Render from a single complete dataset CSV that includes points and metrics.
    Required columns: frame_idx, marker_id, x, y, z, timestamp, utc.
    Optional for panels: kite_0_latitude, kite_0_longitude, ground_tether_length, uwb_distance_m.
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
        "ground_tether_length",
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
            pts_smooth, _ = smooth_points_by_matching(
                pts, prev_points_smoothed, alpha=point_alpha, max_dist=match_radius
            )
            pts = pts_smooth
        prev_points_smoothed = pts.copy() if pts.size > 0 else None

        # group points per marker for struts and LE
        fr["marker_str"] = fr["marker_id"].astype(str)
        strut_pts = {}
        for k in range(8):
            grp = fr[fr["marker_str"] == str(k)][["x","y","z"]].values
            strut_pts[k] = grp if grp.shape[0] >= 1 else None
        le_grp = fr[fr["marker_str"].str.upper() == "LE"][["x","y","z"]].values
        le_path_world = le_shortest_path_3d(le_grp) if le_grp.shape[0] >= 2 else None

        # build reference frame from center struts (3 & 4)
        R_axes, origin = compute_ref_frame_from_center_struts(
            le_path_world, strut_pts.get(3), strut_pts.get(4),
            global_up=np.array([0.0, 0.0, 1.0]), force_flip_x=force_flip_x
        )

        # transform points and segments to local, if requested and frame available
        def _to_frame(P):
            if P is None or P.size == 0:
                return P
            if transform_to_local and (R_axes is not None) and (origin is not None):
                return (P - origin) @ R_axes
            return P

        pts_plot = _to_frame(pts)
        segments = {}
        for k in range(8):
            Pk = _to_frame(strut_pts.get(k))
            if Pk is not None and Pk.shape[0] >= 2:
                seg = fit_line_3d(Pk)
                segments[k] = seg
            else:
                segments[k] = None
        segments["LE"] = _to_frame(le_path_world) if le_path_world is not None else None

        # metrics row
        mrow = per_frame.loc[frame_idx] if frame_idx in per_frame.index else None
        metrics = mrow.to_dict() if isinstance(mrow, pd.Series) else {}

        mode_l = mode.lower()
        if mode_l in ("2d", "3d"):
            if mode_l == "3d":
                pane = _render_points_panel(pts_plot, segments, "3d",
                                            x_lim, y_lim, z_lim, show_wireframe=show_wireframe,
                                            annotate=None, output_size=out_size)
            else:
                cw = out_size[0]//3
                ch = out_size[1]
                # bottom: sideslip
                ss = metrics.get("airspeed_sideslip_angle_filt", metrics.get("airspeed_sideslip_angle"))
                bottom_txt = f"Sideslip: {ss:.1f}°" if pd.notna(ss) else ""
                img_bottom = _render_points_panel(pts_plot, segments, "bottom", x_lim, y_lim, z_lim,
                                                  show_wireframe=show_wireframe,
                                                  annotate={"text": bottom_txt} if bottom_txt else None,
                                                  output_size=(cw, ch))
                # front (z,x): AOA
                aoa = metrics.get("airspeed_angle_of_attack_filt", metrics.get("airspeed_angle_of_attack"))
                front_txt = rf"$\alpha_{{fl}}$: {aoa:.1f}°" if pd.notna(aoa) else ""
                img_front = _render_points_panel(pts_plot, segments, "front", x_lim, y_lim, z_lim,
                                                 show_wireframe=show_wireframe,
                                                 annotate={"text": front_txt} if front_txt else None,
                                                 output_size=(cw, ch))
                # side (x,z): Span (UWB) with ≥3 decimals
                uwb = metrics.get("uwb_distance_m_filt", metrics.get("uwb_distance_m"))
                side_txt = f"Span (UWB): {uwb:.3f} m" if pd.notna(uwb) else ""
                img_side  = _render_points_panel(pts_plot, segments, "side",  x_lim, y_lim, z_lim,
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
        aoa_txt = fr"$\alpha_{{fl}}$: {aoa:.1f}°" if pd.notna(aoa) else ""
        ss = metrics.get("airspeed_sideslip_angle_filt", metrics.get("airspeed_sideslip_angle"))
        sideslip_txt = f"Sideslip: {ss:.1f}°" if pd.notna(ss) else ""

        # Row 1 (left→right): 3D | Bottom(x,y, flip x) | Side(y,z) | Front(x,z, flip x)
        img_3d = _render_points_panel(
            pts_plot, segments, "3d",
            x_lim, y_lim, z_lim, show_wireframe=show_wireframe,
            annotate={"title": None},
            output_size=(cw, ch)
        )

        img_bottom = _render_points_panel(
            pts_plot, segments, "bottom",
            x_lim, y_lim, z_lim, show_wireframe=show_wireframe,
            annotate={"title": "Bottom view", "invert_x": True, "text": sideslip_txt},
            output_size=(cw, ch)
        )

        # SIDE view (y,z) – show AOA text here per your mapping note
        img_side = _render_points_panel(
            pts_plot, segments, "side_yz",
            x_lim, y_lim, z_lim, show_wireframe=show_wireframe,
            annotate={"title": "Side view", "text": aoa_txt},
            output_size=(cw, ch)
        )

        # FRONT view (x,z) with flip & SPAN text
        img_front = _render_points_panel(
            pts_plot, segments, "front_xz",
            x_lim, y_lim, z_lim, show_wireframe=show_wireframe,
            annotate={"title": "Front view", "invert_x": True, "text": span_txt},
            output_size=(cw, ch)
        )

        # Video timing & frame (size two cells)
        utc_overlay = metrics.get("utc", "")
        t_curr = metrics.get("timestamp", np.nan)
        if pd.isna(t_curr) or first_ts is None:
            t_video = input_video_start_s + frame_counter * (1.0 / fps)
        else:
            t_video = input_video_start_s + float(t_curr - first_ts)
        video_img = _get_video_frame(cap, t_video, target_size=(cw * 2, ch),
                                     utc_text=utc_overlay, rotate_180=True)

        poshist_img = _render_spherical_panel(
            [metrics.get("kite_0_latitude", np.nan)],
            [metrics.get("kite_0_longitude", np.nan)],
            [metrics.get("ground_tether_length_filt", metrics.get("ground_tether_length", np.nan))],
            output_size=(cw, ch)
        )

        metrics_block = _make_metrics_block(metrics, size=(cw, ch))

        composite = _compose_all_frame_top4(
            img_3d, img_bottom, img_front, img_side,
            video_img, poshist_img, metrics_block, out_size=size_all
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
    DATASET_CSV      = r"output/left_turn_frame_7182_complete_dataset.csv"

    OUTPUT_VIDEO     = r"output/left_turn_frame_7182_all.mp4"
    MODE             = "all"     # "2d" | "3d" | "all"
    FPS              = 30

    SMOOTH_POINTS    = True
    POINT_ALPHA      = 0.95
    MATCH_RADIUS     = 0.2

    SHOW_WIREFRAME   = True
    TELEMETRY_ALPHA  = 0.95

    X_LIM = (-5, 5)
    Y_LIM = (-3, 3)
    Z_LIM = (3, 8)

    INPUT_VIDEO_PATH = r"Photogrammetry/input/left_videos/09_10_merged.MP4"
    INPUT_VIDEO_START_S = 7182/30

    SIZE_2D = (2400, 800)
    SIZE_3D = (1200, 1200)
    SIZE_ALL= (1920, 1080)

    # NEW: plot in local kite frame built from struts 3 & 4
    TRANSFORM_TO_LOCAL = False
    FORCE_FLIP_X       = False

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
        transform_to_local=TRANSFORM_TO_LOCAL,
        force_flip_x=FORCE_FLIP_X,
    )
