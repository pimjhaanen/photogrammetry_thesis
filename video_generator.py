"""This code can be used to render multi-view videos (front/side/top and/or 3D) from reconstructed
kite markers. It reads cross and ArUco 3D coordinates from CSV, optionally transforms points into
a dynamic ArUco reference frame, applies configurable smoothing, fits straight segments (markers 0–7)
and a shortest-path LE curve, and writes an MP4.

You can adapt input/output paths, FPS, canvas size, and smoothing toggles/parameters.
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # headless rendering
import matplotlib.pyplot as plt
from matplotlib import rcParams
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (needed for 3D projection)

# --- IMPORT ONLY THE GEOMETRY/UTILITY FUNCTIONS (keep limits + render_frame locally!) ---
from Photogrammetry.kite_shape_reconstruction_utils import (
    transform_points_to_aruco,
    smooth_points_by_matching,
    fit_line_3d,
    le_shortest_path_3d,
    build_aruco_reference_table
)

# ============================ CONFIG ============================
INPUT_CSV        = "Photogrammetry/output/3d_coordinates_25_06_test2_merged.csv"
ARUCO_COORDS_CSV = "Photogrammetry/output/aruco_coordinates_25_06_test2_merged.csv"  # not directly used here, kept for parity
OUTPUT_VIDEO     = "Photogrammetry/output/test_02_camera_frame_LPF_2d.mp4"
FPS              = 30

# ------------------- Smoothing Options (set once) -------------------
LOWPASS         = True   # master toggle enabling smoothing behaviors
ALPHA           = 0.9    # previous-frame weight; higher = smoother (pose) – used by your utils if applicable
USE_ARUCO_FRAME = True   # transform to ArUco frame

# Choose *where* to apply smoothing (combine as you wish):
LPF_POSE      = True   # smooth pose (R,t) with SLERP for rotation and EMA for origin (in utils)
SMOOTH_POINTS = True   # smooth transformed points by matching to previous frame

# Post-transform point smoothing params (used iff SMOOTH_POINTS)
POINT_ALPHA  = 0.9      # previous-frame weight for point EMA
MATCH_RADIUS = 0.35     # meters: max distance to match a point to previous frame

# Convenience alias for transform_points_to_aruco
LOWPASS_AXES = (LOWPASS and LPF_POSE)

# ------------------- Plot styling -------------------
POINT_SIZE   = 20
LINE_SAMPLES = 60  # for PCA lines (ids 0..7), not used for LE

# ============================ LOAD DATA ============================
cross_df = pd.read_csv(INPUT_CSV)

required_cross_cols = {"frame_idx", "marker_id", "x", "y", "z"}
if not required_cross_cols.issubset(cross_df.columns):
    raise ValueError(f"INPUT_CSV missing columns: {required_cross_cols - set(cross_df.columns)}")

unique_frames = np.sort(cross_df["frame_idx"].unique())

import Photogrammetry.kite_shape_reconstruction_utils as kres

#Make the aruco coordinates into a reference table:
aruco_coords_df = pd.read_csv(ARUCO_COORDS_CSV)
# If your utils' builder supports a lowpass flag, pass it; if not, omit the arg.
aruco_coords_df = pd.read_csv(ARUCO_COORDS_CSV)

# ensure frames are plain ints
frames_cover = np.asarray(unique_frames, dtype=int)

try:
    kres.aruco_ref_table = kres.build_aruco_reference_table(
        aruco_coords_df,
        frames_to_cover=frames_cover,
        lowpass_tracks=bool(LOWPASS)  # if your builder supports it
    )
except TypeError:
    kres.aruco_ref_table = kres.build_aruco_reference_table(
        aruco_coords_df,
        frames_to_cover=frames_cover
    )

# (optional) make sure smoothing globals exist on the module
if not hasattr(kres, "filtered_axes"):
    kres.filtered_axes = {}
kres.ALPHA = ALPHA


# ============================ RENDERING LIMITS (KEEP LOCAL) ============================
X_LIM = (-5, 2)
Y_LIM = (-6, 6)
Z_LIM = (-5, 2)

# ============================ RENDERING ============================
def render_frame(points, line_segments, mode, output_size=None,
                 panel_letters=False,           # add (a)(b)(c) under panels (video use)
                 pdf_path=None,                  # save a clean vector PDF of this frame (no titles)
                 dpi=200,                        # overall raster density
                 base_fontsize=14,               # larger fonts per feedback
                 tick_fontsize=12,
                 label_fontsize=14,
                 lw=2.2):
    """RELEVANT FUNCTION INPUTS:
    - points: (N,3) float array of 3D points (already in desired world/frame)
    - line_segments: list of optional (M_k,3) arrays for fitted/ordered lines to overlay
    - mode: "3d" | "2d" | "all" — choose single 3D view, 3× 2D orthographic panels, or both
    - output_size: (width_px, height_px) raster size; if None, figure size determines resolution (prefered)
    - panel_letters: if True, annotate panels with (a)(b)(c)(d) beneath axes
    - pdf_path: if provided, also saves a clean vector PDF of this frame (no titles)
    - dpi: dots per inch for rasterization when output_size is None
    - base_fontsize / tick_fontsize / label_fontsize: font sizing controls
    - lw: line width for plotted segments

    Returns:
    - img_bgr: H×W×3 uint8 image (BGR) ready for cv2.VideoWriter.
    """
    # --- A. Visual style (bigger and clean) ---
    rcParams.update({
        "font.size": base_fontsize,
        "axes.labelsize": label_fontsize,
        "axes.titlesize": base_fontsize,  # we won't use titles in 2D per feedback
        "xtick.labelsize": tick_fontsize,
        "ytick.labelsize": tick_fontsize,
        "lines.linewidth": lw,
    })

    def plot_segments_scatter(ax, segs, pts, xi, yi, xlab, ylab, xlim, ylim):
        for seg in segs:
            if seg is not None and len(seg) > 0:
                ax.plot(seg[:, xi], seg[:, yi])
        ax.scatter(pts[:, xi], pts[:, yi], c='red', s=POINT_SIZE)
        ax.set_xlabel(xlab); ax.set_ylabel(ylab)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.4)
        try:
            ax.set_box_aspect(1)
        except Exception:
            pass
        ax.set_anchor('C')

    # Figure size from desired pixel target (if given)
    if output_size is not None:
        Wpx, Hpx = output_size
        fig_w, fig_h = Wpx / dpi, Hpx / dpi
    else:
        if mode.lower() == "2d":
            fig_w, fig_h = 15, 5
        elif mode.lower() == "3d":
            fig_w, fig_h = 6, 6
        else:  # "all"
            fig_w, fig_h = 8, 8

    # --- Build figure ---
    m = mode.lower()
    if m == "3d":
        fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
        ax3d = fig.add_subplot(1, 1, 1, projection='3d')
        for seg in line_segments:
            if seg is not None and len(seg) > 0:
                ax3d.plot(seg[:, 0], seg[:, 1], seg[:, 2])
        ax3d.scatter(points[:, 0], points[:, 1], points[:, 2], c='red', s=POINT_SIZE)
        ax3d.set_xlabel(r"$x_\mathrm{w}\,(\mathrm{m})$")
        ax3d.set_ylabel(r"$y_\mathrm{w}\,(\mathrm{m})$")
        ax3d.set_zlabel(r"$z_\mathrm{w}\,(\mathrm{m})$")
        ax3d.set_xlim(*X_LIM); ax3d.set_ylim(*Y_LIM); ax3d.set_zlim(*Z_LIM)
        ax3d.view_init(elev=30, azim=-45)
        ax3d.grid(True, alpha=0.4)

    elif m == "2d":
        fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
        ax_front = fig.add_subplot(1, 3, 1)
        ax_side  = fig.add_subplot(1, 3, 2)
        ax_top   = fig.add_subplot(1, 3, 3)

        plot_segments_scatter(ax_front, line_segments, points, 1, 2,
                              r"$y_\mathrm{w}\,(\mathrm{m})$", r"$z_\mathrm{w}\,(\mathrm{m})$", Y_LIM, Z_LIM)
        plot_segments_scatter(ax_side,  line_segments, points, 0, 2,
                              r"$x_\mathrm{w}\,(\mathrm{m})$", r"$z_\mathrm{w}\,(\mathrm{m})$", X_LIM, Z_LIM)
        plot_segments_scatter(ax_top,   line_segments, points, 0, 1,
                              r"$x_\mathrm{w}\,(\mathrm{m})$", r"$y_\mathrm{w}\,(\mathrm{m})$", X_LIM, Y_LIM)

        if panel_letters:
            letters = ["(a)", "(b)", "(c)"]
            for ax, lab in zip([ax_front, ax_side, ax_top], letters):
                pos = ax.get_position()
                cx  = 0.5*(pos.x0 + pos.x1)
                y   = pos.y0 - 0.02
                fig.text(cx, y, lab, ha="center", va="top", fontsize=label_fontsize)

    else:  # "all"
        fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
        ax3d    = fig.add_subplot(2, 2, 1, projection='3d')
        ax_front= fig.add_subplot(2, 2, 2)
        ax_side = fig.add_subplot(2, 2, 3)
        ax_top  = fig.add_subplot(2, 2, 4)

        for seg in line_segments:
            if seg is not None and len(seg) > 0:
                ax3d.plot(seg[:, 0], seg[:, 1], seg[:, 2])
        ax3d.scatter(points[:, 0], points[:, 1], points[:, 2], c='red', s=POINT_SIZE)
        ax3d.set_xlabel(r"$x_\mathrm{w}\,(\mathrm{m})$")
        ax3d.set_ylabel(r"$y_\mathrm{w}\,(\mathrm{m})$")
        ax3d.set_zlabel(r"$z_\mathrm{w}\,(\mathrm{m})$")
        ax3d.set_xlim(*X_LIM); ax3d.set_ylim(*Y_LIM); ax3d.set_zlim(*Z_LIM)
        ax3d.view_init(elev=30, azim=-45); ax3d.grid(True, alpha=0.4)

        plot_segments_scatter(ax_front, line_segments, points, 1, 2,
                              r"$y_\mathrm{w}\,(\mathrm{m})$", r"$z_\mathrm{w}\,(\mathrm{m})$", Y_LIM, Z_LIM)
        plot_segments_scatter(ax_side,  line_segments, points, 0, 2,
                              r"$x_\mathrm{w}\,(\mathrm{m})$", r"$z_\mathrm{w}\,(\mathrm{m})$", X_LIM, Z_LIM)
        plot_segments_scatter(ax_top,   line_segments, points, 0, 1,
                              r"$x_\mathrm{w}\,(\mathrm{m})$", r"$y_\mathrm{w}\,(\mathrm{m})$", X_LIM, Y_LIM)

        if panel_letters:
            letters = ["(a)", "(b)", "(c)", "(d)"]
            for ax, lab in zip([ax3d, ax_front, ax_side, ax_top], letters):
                pos = ax.get_position()
                cx  = 0.5*(pos.x0 + pos.x1)
                y   = pos.y0 - 0.02
                fig.text(cx, y, lab, ha="center", va="top", fontsize=label_fontsize)

    # Optional PDF
    if pdf_path is not None:
        fig.savefig(pdf_path, format="pdf", bbox_inches="tight")

    # Rasterize for video pipeline
    fig.tight_layout()
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Optional resize
    if output_size is not None and (img.shape[1], img.shape[0]) != output_size:
        interp = cv2.INTER_AREA if (output_size[0] < img.shape[1]) else cv2.INTER_LANCZOS4
        img = cv2.resize(img, output_size, interpolation=interp)

    return img


# ============================ MAIN (2D triptych) ============================
def main():
    MODE = "2d"                   # keep other modes unchanged elsewhere
    OUTPUT_SIZE_2D = (2400, 800)  # wide so each 2D subplot is square

    out = None
    frame_count = 0
    prev_points_smoothed = None
    for frame_idx in unique_frames:
        frame_df = cross_df[cross_df["frame_idx"] == frame_idx]
        if frame_df.shape[0] == 0:
            continue
        pts_world = frame_df[["x", "y", "z"]].values.astype(float)
        if USE_ARUCO_FRAME:
            pts = transform_points_to_aruco(pts_world, int(frame_idx), lowpass=LOWPASS_AXES)
            if pts is None:
                print("points is none")
                continue
        else:
            pts = pts_world

        # Use transformed coords for grouping/fitting
        frame_df = frame_df.copy()
        frame_df["marker_str"] = frame_df["marker_id"].astype(str)
        frame_df[["x", "y", "z"]] = pts

        # Drop isolated points (no neighbor within 2 m, Euclidean)
        xyz = frame_df[["x", "y", "z"]].to_numpy()
        if xyz.shape[0] >= 2:
            diffs = xyz[:, None, :] - xyz[None, :, :]
            D = np.linalg.norm(diffs, axis=2)
            np.fill_diagonal(D, np.inf)
            keep = D.min(axis=1) <= 2.0
            frame_df = frame_df.loc[keep].copy()
        else:
            frame_df = frame_df.iloc[0:0].copy()

        # Optional smoothing in ArUco frame by matching to previous frame
        points_to_plot = frame_df[["x", "y", "z"]].to_numpy()
        if SMOOTH_POINTS and LOWPASS and points_to_plot.size > 0:
            points_smoothed, _ = smooth_points_by_matching(
                points_to_plot, prev_points_smoothed,
                alpha=POINT_ALPHA, max_dist=MATCH_RADIUS
            )
            frame_df[["x", "y", "z"]] = points_smoothed
            points_to_plot = points_smoothed

        # Build line segments
        line_segments = []
        for k in range(8):  # 0..7 straight lines
            grp = frame_df[frame_df["marker_str"] == str(k)]
            if grp.shape[0] >= 2:
                line_segments.append(fit_line_3d(grp[["x", "y", "z"]].values))
            else:
                line_segments.append(None)

        # LE: shortest 3D path
        le_grp = frame_df[frame_df["marker_str"].str.upper() == "LE"]
        if le_grp.shape[0] >= 2:
            le_points = le_grp[["x", "y", "z"]].values
            le_path = le_shortest_path_3d(le_points, exact_limit=12)
            line_segments.append(le_path)
        else:
            line_segments.append(None)

        # Render wide 1×3
        img = render_frame(points_to_plot, line_segments, mode=MODE, output_size=OUTPUT_SIZE_2D)

        # Lazy-init writer to the image size
        if out is None:
            h, w = img.shape[:2]
            os.makedirs(os.path.dirname(OUTPUT_VIDEO) or ".", exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, FPS, (w, h))

        out.write(img)
        frame_count += 1
        prev_points_smoothed = points_to_plot.copy()

    if out is not None:
        out.release()
    print(f"[✓] Total frames written: {frame_count}")


if __name__ == "__main__":
    main()
