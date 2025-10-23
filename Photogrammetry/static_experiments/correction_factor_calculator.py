"""
Coarse→Medium→Fine grid search over yaw/pitch/roll using ONE manual click session (on raw images),
with a combined-best pick: span within ±0.05 m (5 cm) prioritized, then minimal epipolar error.

Stages:
  1) COARSE  step = 0.5°
  2) MEDIUM  step = 0.05°  (around best from stage 1)
  3) FINE    step = 0.005° (around best from stage 2)

For each (dyaw, dpitch, droll):
  - R_adj = dR @ R  (dR in LEFT/world frame; yaw +Y, pitch +X, roll +Z)
  - stereoRectify -> R1,R2,P1,P2,Q
  - Map RAW clicks -> rectified pixels (cv2.undistortPoints with R_rect, P_rect[:3,:3])
  - Compute mean |dy| and triangulate (span from first two points)

Reports:
  - best-by-epipolar
  - best-by-span (closest to UWB)
  - best-by-combined (±0.05 m span tolerance, then epipolar; fallback=min violation)

Includes:
  - 3D translucent scatter of span metric (or |Δspan| if UWB given)
  - 2D heatmaps (pitch×roll) at the best yaw
"""

import itertools
import pickle
from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.colors import Normalize

# --------------------------- Math helpers ---------------------------

def rodrigues_axis_angle(axis: str, angle_deg: float) -> np.ndarray:
    axis = axis.lower()
    if axis not in ("x", "y", "z"):
        raise ValueError("axis must be 'x','y','z'")
    v = {"x": np.array([1.0, 0.0, 0.0]),
         "y": np.array([0.0, 1.0, 0.0]),
         "z": np.array([0.0, 0.0, 1.0])}[axis]
    theta = np.deg2rad(float(angle_deg))
    R_delta, _ = cv2.Rodrigues(v * theta)   # axis*angle -> 3x3
    return R_delta


def adjust_R_yaw_pitch_roll_leftframe(R: np.ndarray, yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0) -> np.ndarray:
    """
    Apply dR in the LEFT (cam1/world) frame: R' = dR @ R
    yaw about +Y, pitch about +X, roll about +Z. Order: yaw->pitch->roll.
    """
    Rz = rodrigues_axis_angle("z", roll_deg)
    Rx = rodrigues_axis_angle("x", pitch_deg)
    Ry = rodrigues_axis_angle("y", yaw_deg)
    dR = Ry @ Rx @ Rz
    return dR @ R


def triangulate_cv(Lxy, Rxy, P1, P2) -> np.ndarray:
    """Triangulate with OpenCV; Lxy/Rxy are Nx2 rectified pixel coords; returns (N,3)."""
    if len(Lxy) == 0 or len(Rxy) == 0:
        return np.empty((0, 3), float)
    n = min(len(Lxy), len(Rxy))
    L = np.asarray(Lxy[:n], float).T  # 2xN
    R = np.asarray(Rxy[:n], float).T  # 2xN
    Xh = cv2.triangulatePoints(P1, P2, L, R)  # 4xN
    X = (Xh[:3] / Xh[3]).T  # Nx3
    return X


# --------------------------- I/O helpers ---------------------------

def _load_calibration(calib_pkl_path: str) -> Dict:
    with open(calib_pkl_path, "rb") as f:
        calib = pickle.load(f)
    required = ["camera_matrix_1", "dist_coeffs_1", "camera_matrix_2", "dist_coeffs_2", "R", "T"]
    miss = [k for k in required if k not in calib]
    if miss:
        raise KeyError(f"Calibration file is missing keys: {miss}")
    return calib


def _imread_color(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Could not read image: {path}")
    return img


# ---------------------------- Click UI on RAW images ----------------------------

class ZoomClickSession:
    """
    Zoomable/pannable click UI on the provided image.
    Keys: left-click add, wheel zoom, right-drag pan, 'u' undo, Enter accept, Esc cancel.
    """
    def __init__(self, window_name: str, image_bgr: np.ndarray, *, display_scale: float = 0.6):
        self.win = window_name
        self.base = image_bgr
        H, W = self.base.shape[:2]
        self.init_w = max(640, int(W * display_scale))
        self.init_h = max(360, int(H * display_scale))
        self.zoom = 1.0
        self.cx, self.cy = W / 2.0, H / 2.0
        self.dragging = False
        self.last_mouse = (0, 0)
        self.points: List[Tuple[float, float]] = []
        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.win, self.init_w, self.init_h)
        cv2.setMouseCallback(self.win, self._on_mouse)
        self._redraw()

    def _win_size(self) -> Tuple[int, int]:
        try:
            _, _, ww, hh = cv2.getWindowImageRect(self.win)
            return int(max(1, ww)), int(max(1, hh))
        except Exception:
            return getattr(self, "_last_drawn_size", (self.init_w, self.init_h))

    def _roi_dims(self):
        H, W = self.base.shape[:2]
        vw = max(16, int(round(W / self.zoom)))
        vh = max(16, int(round(H / self.zoom)))
        x0 = int(round(self.cx - vw / 2))
        y0 = int(round(self.cy - vh / 2))
        x0 = max(0, min(x0, W - vw))
        y0 = max(0, min(y0, H - vh))
        return x0, y0, vw, vh

    def _disp_to_image(self, x_win: int, y_win: int) -> Tuple[float, float]:
        x0, y0, vw, vh = self._roi_dims()
        ww, hh = self._win_size()
        u = np.clip(x_win / max(1, ww), 0.0, 1.0)
        v = np.clip(y_win / max(1, hh), 0.0, 1.0)
        ix = x0 + u * vw
        iy = y0 + v * vh
        H, W = self.base.shape[:2]
        return float(np.clip(ix, 0, W - 1)), float(np.clip(iy, 0, H - 1))

    def _image_to_disp(self, ix: float, iy: float) -> Tuple[int, int]:
        x0, y0, vw, vh = self._roi_dims()
        ww, hh = self._win_size()
        u = (ix - x0) / max(1e-9, vw)
        v = (iy - y0) / max(1e-9, vh)
        return int(round(u * ww)), int(round(v * hh))

    def _zoom_at(self, x_win: int, y_win: int, factor: float):
        old_zoom = self.zoom
        new_zoom = float(np.clip(old_zoom * factor, 0.2, 40.0))
        if abs(new_zoom - old_zoom) < 1e-6:
            return
        ix, iy = self._disp_to_image(x_win, y_win)
        self.zoom = new_zoom
        ww, hh = self._win_size()
        u, v = x_win / max(1, ww), y_win / max(1, hh)
        vw, vh = int(round(self.base.shape[1] / self.zoom)), int(round(self.base.shape[0] / self.zoom))
        x0 = ix - u * vw
        y0 = iy - v * vh
        self.cx = x0 + vw / 2.0
        self.cy = y0 + vh / 2.0
        self._redraw()

    def _on_mouse(self, event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            ix, iy = self._disp_to_image(x, y)
            self.points.append((ix, iy))
            self._redraw()
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.dragging = True
            self.last_mouse = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            dx = x - self.last_mouse[0]
            dy = y - self.last_mouse[1]
            _, _, vw, vh = self._roi_dims()
            ww, hh = self._win_size()
            self.cx -= dx * (vw / max(1, ww))
            self.cy -= dy * (vh / max(1, hh))
            self.last_mouse = (x, y)
            self._redraw()
        elif event == cv2.EVENT_RBUTTONUP:
            self.dragging = False
        elif event == cv2.EVENT_MOUSEWHEEL:
            self._zoom_at(x, y, 1.25 if flags > 0 else 1 / 1.25)

    def _redraw(self):
        x0, y0, vw, vh = self._roi_dims()
        roi = self.base[y0:y0 + vh, x0:x0 + vw]
        ww, hh = self._win_size()
        disp = cv2.resize(roi, (ww, hh), interpolation=cv2.INTER_AREA)
        for i, (px, py) in enumerate(self.points):
            dx, dy = self._image_to_disp(px, py)
            if 0 <= dx < ww and 0 <= dy < hh:
                cv2.drawMarker(disp, (dx, dy), (0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=18, thickness=2)
                cv2.putText(disp, str(i), (dx + 8, dy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(disp, str(i), (dx + 8, dy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        status = f"zoom={self.zoom:.2f}  pts={len(self.points)}   (u=undo, Enter=accept)"
        cv2.putText(disp, status, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(disp, status, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(self.win, disp)
        self._last_drawn_size = (ww, hh)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('+'), ord('=')):
            self._zoom_at(ww // 2, hh // 2, 1.25)
        elif key in (ord('-'), ord('_')):
            self._zoom_at(ww // 2, hh // 2, 1 / 1.25)
        elif key in (ord('u'), ord('U')):
            if self.points:
                self.points.pop()
                self._redraw()

    def collect(self, n_expected: Optional[int], group_label: str) -> List[Tuple[float, float]]:
        while True:
            cv2.setWindowTitle(self.win, f"{self.win} — {group_label} | clicked {len(self.points)}"
                                         + (f"/{n_expected}" if n_expected else ""))  # noqa: E501
            key = cv2.waitKey(20) & 0xFF
            if key in (13, 10):  # Enter
                break
            elif key == 27:  # Esc
                raise KeyboardInterrupt("User cancelled.")
            if n_expected is not None and len(self.points) >= n_expected:
                break
        return list(self.points)

    def close(self):
        try:
            cv2.destroyWindow(self.win)
        except cv2.error:
            pass


# --------------------------- Core mapping/metrics ---------------------------

def stereo_rectify_params(K1, D1, K2, D2, size, R, T, alpha=0.0):
    """Compute (R1,R2,P1,P2,Q) — no image remap needed for the search."""
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K1, D1, K2, D2, size, R, T,
        flags=cv2.CALIB_FIX_INTRINSIC, alpha=alpha
    )
    return R1, R2, P1, P2, Q


def undistort_rectify_points(px_list: List[Tuple[float, float]], K, D, R_rect, P_rectK) -> np.ndarray:
    """
    Map raw pixel coords -> rectified pixel coords using undistortPoints with R,P.
    P_rectK must be 3x3 (use P[:3,:3] from stereoRectify).
    Returns (N,2) rectified pixel coordinates.
    """
    pts = np.asarray(px_list, dtype=float).reshape(-1, 1, 2)
    pr = cv2.undistortPoints(pts, K, D, R=R_rect, P=P_rectK)  # Nx1x2 pixels in rectified frame
    return pr.reshape(-1, 2)


def mean_abs_dy(Lr_xy: np.ndarray, Rr_xy: np.ndarray) -> float:
    n = min(Lr_xy.shape[0], Rr_xy.shape[0])
    if n == 0:
        return float('inf')
    dy = np.abs(Lr_xy[:n, 1] - Rr_xy[:n, 1])
    return float(np.mean(dy))


# ---------- Combined pick (span tolerance then epipolar) ----------

def pick_best_combined(results: List[dict], uwb_span_m: Optional[float], span_tol_m: float = 0.05) -> Optional[dict]:
    """
    Choose best with |span_err| <= span_tol_m and minimal epipolar; fallback:
    minimize violation amount, then epipolar.
    """
    if uwb_span_m is None:
        return None
    valid = [r for r in results if r.get("span_err") is not None]
    if not valid:
        return None

    within = [r for r in valid if r["span_err"] is not None and r["span_err"] <= span_tol_m]
    if within:
        best = min(within, key=lambda r: r["mean_abs_dy"])
        best["_combined_reason"] = f"within {span_tol_m:.3f} m, min epipolar"
        return best

    # No candidate meets tolerance: minimize violation amount, then epipolar
    best = min(valid, key=lambda r: (max(r["span_err"] - span_tol_m, 0.0), r["mean_abs_dy"]))
    best["_combined_reason"] = f"min violation of {span_tol_m:.3f} m, then epipolar"
    return best


# ---------- 3D + 2D plotting helpers ----------

def _results_to_arrays_basic(results: List[Dict]):
    yaw   = np.array([r["dyaw"]   for r in results], dtype=float)  # ψ
    pitch = np.array([r["dpitch"] for r in results], dtype=float)  # θ
    roll  = np.array([r["droll"]  for r in results], dtype=float)  # φ
    epi   = np.array([r["mean_abs_dy"] for r in results], dtype=float)
    span  = np.array([r["span_m"] for r in results], dtype=float)
    span_err = np.array([
        (r.get("span_err") if r.get("span_err") is not None else np.nan)
        for r in results
    ], dtype=float)
    return pitch, roll, yaw, epi, span, span_err


def _build_slice_grid(results: List[Dict], yaw_target: float):
    yaws = np.array(sorted(set([r["dyaw"] for r in results])), dtype=float)
    idx = int(np.argmin(np.abs(yaws - yaw_target)))
    yaw_chosen = float(yaws[idx])
    slice_results = [r for r in results if abs(r["dyaw"] - yaw_chosen) < 1e-12]
    pitches = np.array(sorted(set([r["dpitch"] for r in slice_results])), dtype=float)
    rolls   = np.array(sorted(set([r["droll"]  for r in slice_results])), dtype=float)
    cell = {(float(r["dpitch"]), float(r["droll"])): r for r in slice_results}
    return pitches, rolls, cell, yaw_chosen


def _grid_from_cells(pitches, rolls, cell_dict, key: str) -> np.ndarray:
    Z = np.full((len(rolls), len(pitches)), np.nan, dtype=float)
    for i, p in enumerate(pitches):
        for j, r in enumerate(rolls):
            d = cell_dict.get((float(p), float(r)))
            if d is not None and (key in d) and (d[key] is not None):
                Z[j, i] = float(d[key])
    return Z


# --------------------------- Evaluation + Search ---------------------------

def _eval_grid_given_clicks(
    K1, D1, K2, D2, R, T, size,
    L_pts_raw: List[Tuple[float, float]],
    R_pts_raw: List[Tuple[float, float]],
    yaw_list_deg: List[float],
    pitch_list_deg: List[float],
    roll_list_deg: List[float],
    rectify_alpha: float,
    uwb_span_m: Optional[float]
):
    results = []
    combos = list(itertools.product(yaw_list_deg, pitch_list_deg, roll_list_deg))
    print(f"[SEARCH] Evaluating {len(combos)} combinations...")
    for (dyaw, dpitch, droll) in combos:
        R_adj = adjust_R_yaw_pitch_roll_leftframe(R, yaw_deg=dyaw, pitch_deg=dpitch, roll_deg=droll)
        R1, R2, P1, P2, Q = stereo_rectify_params(K1, D1, K2, D2, size, R_adj, T, alpha=rectify_alpha)
        L_rect_xy = undistort_rectify_points(L_pts_raw, K1, D1, R1, P1[:3, :3])
        R_rect_xy = undistort_rectify_points(R_pts_raw, K2, D2, R2, P2[:3, :3])

        mdy = mean_abs_dy(L_rect_xy, R_rect_xy)
        X = triangulate_cv(L_rect_xy, R_rect_xy, P1, P2)
        span_m = float('nan')
        if X.shape[0] >= 2:
            span_m = float(np.linalg.norm(X[1] - X[0]))

        span_err = None
        if uwb_span_m is not None and np.isfinite(span_m):
            span_err = abs(span_m - uwb_span_m)

        results.append({
            "dyaw": dyaw, "dpitch": dpitch, "droll": droll,
            "mean_abs_dy": mdy,
            "span_m": span_m,
            "span_err": span_err,
            "P1": P1, "P2": P2, "R1": R1, "R2": R2, "Q": Q
        })

    # Bests
    by_epi = sorted(results, key=lambda r: r["mean_abs_dy"])
    best_epi = by_epi[0]
    best_span = None
    if uwb_span_m is not None:
        valid_span = [r for r in results if r["span_err"] is not None]
        if valid_span:
            best_span = min(valid_span, key=lambda r: r["span_err"])

    return results, best_epi, best_span


def _mk_list(center: float, half_window: float, step: float) -> List[float]:
    lo = center - half_window
    hi = center + half_window
    nudge = step * 0.5 * 1e-6
    return list(np.arange(lo, hi + nudge, step))

# ==== Four plots from the COARSE grid ====
# - 3D epipolar + 3D |Δspan| (or span) side-by-side
# - 2D smoothed slice (pitch×roll) at the "right" yaw from the coarse grid, for both metrics
#
# Usage (after your coarse stage):
#   coarse_results = out["stage1"]["results"]
#   UWB = UWB_SPAN_M  # or None
#   plot_coarse_3d_pair(coarse_results, uwb_span_m=UWB, alpha=0.55)
#   plot_coarse_slice_smoothed(coarse_results, yaw_target=None, uwb_span_m=UWB, levels=24)
#
# You may also pass an explicit yaw_target (deg). If None, it picks the yaw of the
# best-|Δspan| (if UWB given) else best-epipolar within the COARSE results.
#
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

def _extract_arrays(results):
    pitch = np.array([r["dpitch"] for r in results], dtype=float)
    roll  = np.array([r["droll"]  for r in results], dtype=float)
    yaw   = np.array([r["dyaw"]   for r in results], dtype=float)
    epi   = np.array([r["mean_abs_dy"] for r in results], dtype=float)
    span  = np.array([r["span_m"] for r in results], dtype=float)
    span_err = np.array([r["span_err"] if r.get("span_err") is not None else np.nan
                         for r in results], dtype=float)
    return pitch, roll, yaw, epi, span, span_err

def _best_by_epipolar(results):
    return min(results, key=lambda r: r["mean_abs_dy"])

def _best_by_span(results):
    cand = [r for r in results if r.get("span_err") is not None]
    return (min(cand, key=lambda r: r["span_err"]) if cand else None)

def _pad(lo, hi, frac=0.05):
    d = (hi - lo) if hi > lo else 1.0
    return lo - frac*d, hi + frac*d

# --------- 1) & 2) COARSE 3D PLOTS (side-by-side) ----------
def plot_coarse_3d_pair(coarse_results, *, uwb_span_m=None, alpha=0.9, s=6):
    """
    Left: epipolar mean|dy| (px)  —  Right: |Δspan|(m) if UWB is given, else span(m).
    Clean 3D "point cloud" look (tiny pixel markers, no edges). No best-point markers.
    """
    pitch, roll, yaw, epi, span, span_err = _extract_arrays(coarse_results)

    # Choose right metric
    if uwb_span_m is not None and np.isfinite(span_err).any():
        right_vals = span_err
        right_label = "|Δspan| (m)"
        right_title = f"|span − UWB|  (UWB={uwb_span_m:.3f} m)"
    else:
        right_vals = span
        right_label = "span (m)"
        right_title = "Estimated span (m)"

    # Shared axes limits
    xlim = _pad(float(np.nanmin(pitch)), float(np.nanmax(pitch)))
    ylim = _pad(float(np.nanmin(roll)),  float(np.nanmax(roll)))
    zlim = _pad(float(np.nanmin(yaw)),   float(np.nanmax(yaw)))

    fig = plt.figure(figsize=(12, 5))
    fig.suptitle("COARSE grid — 3D fields (θ=pitch, φ=roll, ψ=yaw)", fontsize=12)

    # Left: epipolar
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    sc1 = ax1.scatter(pitch, roll, yaw, c=epi, s=s, alpha=alpha, marker=',', linewidths=0)
    cb1 = fig.colorbar(sc1, ax=ax1, shrink=0.82, pad=0.1); cb1.set_label("mean |dy| (px)")
    ax1.set_xlabel("θ (pitch) [deg]"); ax1.set_ylabel("φ (roll) [deg]"); ax1.set_zlabel("ψ (yaw) [deg]")
    ax1.set_title("Epipolar mean |dy| (px)")
    ax1.set_xlim(xlim); ax1.set_ylim(ylim); ax1.set_zlim(zlim)

    # Right: |Δspan| or span
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    sc2 = ax2.scatter(pitch, roll, yaw, c=right_vals, s=s, alpha=alpha, marker=',', linewidths=0)
    cb2 = fig.colorbar(sc2, ax=ax2, shrink=0.82, pad=0.1); cb2.set_label(right_label)
    ax2.set_xlabel("θ (pitch) [deg]"); ax2.set_ylabel("φ (roll) [deg]"); ax2.set_zlabel("ψ (yaw) [deg]")
    ax2.set_title(right_title)
    ax2.set_xlim(xlim); ax2.set_ylim(ylim); ax2.set_zlim(zlim)

    plt.tight_layout()
    plt.show()

# --------- 3) & 4) COARSE 2D SMOOTHED SLICE (side-by-side) ----------
def plot_coarse_slice_smoothed(
    coarse_results,
    *,
    yaw_target=None,
    uwb_span_m=None,
    levels=64,
    final_best=None  # dict from Stage-3 (fine) with dpitch/droll/dyaw; only this point is shown
):
    """
    Two smoothed (tricontourf) fields on pitch×roll at a single yaw slice:
      Left: epipolar mean|dy| (px)
      Right: |Δspan| (m) if UWB given, else span (m)
    No sample dots are drawn. Only the final Stage-3 best point is marked (if provided),
    using its true (fine) dpitch/droll (not snapped to coarse).
    """
    # Pick yaw: default to best |Δspan| (if UWB), else best epipolar — within COARSE set.
    if yaw_target is None:
        if uwb_span_m is not None and _best_by_span(coarse_results) is not None:
            yaw_pick = _best_by_span(coarse_results)["dyaw"]
        else:
            yaw_pick = _best_by_epipolar(coarse_results)["dyaw"]
    else:
        yaw_pick = float(yaw_target)

    # Use nearest coarse yaw to populate the slice field (data availability)
    yaws = np.array(sorted(set([r["dyaw"] for r in coarse_results])), dtype=float)
    yaw_chosen = float(yaws[np.argmin(np.abs(yaws - yaw_pick))])

    slice_res = [r for r in coarse_results if abs(r["dyaw"] - yaw_chosen) < 1e-12]
    if not slice_res:
        print("[plot_coarse_slice_smoothed] No coarse results at that yaw.")
        return

    pitch, roll, _, epi, span, span_err = _extract_arrays(slice_res)

    # Right metric
    if uwb_span_m is not None and np.isfinite(span_err).any():
        right_vals = span_err
        right_label = "|Δspan| (m)"
        right_title = f"|span − UWB| at ψ={yaw_chosen:.3f}° (UWB={uwb_span_m:.3f} m)"
    else:
        right_vals = span
        right_label = "span (m)"
        right_title = f"span (m) at ψ={yaw_chosen:.3f}°"

    # Valid masks
    m_epi = np.isfinite(epi)
    m_right = np.isfinite(right_vals)

    # Limits padded to also include the *true* fine best marker (if given)
    xmin, xmax = float(np.nanmin(pitch)), float(np.nanmax(pitch))
    ymin, ymax = float(np.nanmin(roll)),  float(np.nanmax(roll))
    if final_best is not None:
        xmin = min(xmin, float(final_best["dpitch"]))
        xmax = max(xmax, float(final_best["dpitch"]))
        ymin = min(ymin, float(final_best["droll"]))
        ymax = max(ymax, float(final_best["droll"]))
    xlim = _pad(xmin, xmax); ylim = _pad(ymin, ymax)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    fig.suptitle(f"COARSE slice at ψ={yaw_chosen:.3f}° (smoothed)", fontsize=12)

    # Left: epipolar (smooth gradient)
    cn1 = axes[0].tricontourf(pitch[m_epi], roll[m_epi], epi[m_epi], levels=levels)
    cb1 = fig.colorbar(cn1, ax=axes[0], shrink=0.9, pad=0.02); cb1.set_label("mean |dy| (px)")
    axes[0].set_xlabel("θ (pitch) [deg]"); axes[0].set_ylabel("φ (roll) [deg]")
    axes[0].set_title("Epipolar mean |dy| (px)")
    axes[0].set_xlim(xlim); axes[0].set_ylim(ylim)

    # Right: |Δspan| / span (smooth gradient)
    cn2 = axes[1].tricontourf(pitch[m_right], roll[m_right], right_vals[m_right], levels=levels)
    cb2 = fig.colorbar(cn2, ax=axes[1], shrink=0.9, pad=0.02); cb2.set_label(right_label)
    axes[1].set_xlabel("θ (pitch) [deg]"); axes[1].set_ylabel("φ (roll) [deg]")
    axes[1].set_title(right_title)
    axes[1].set_xlim(xlim); axes[1].set_ylim(ylim)

    # Only mark the final (Stage-3) best point, at its true dpitch/droll
    if final_best is not None:
        px, rx = float(final_best["dpitch"]), float(final_best["droll"])
        for ax in axes:
            ax.plot([px], [rx], marker='x', markersize=9, mew=2, color='k')  # single crisp marker

    plt.tight_layout()
    plt.show()



def coarse_to_fine_search_ypr(
    calib_file: str,
    left_path: str,
    right_path: str,
    *,
    n_clicks: int = 6,
    rectify_alpha: float = 0.0,
    uwb_span_m: Optional[float] = None,
    span_tol_m: float = 0.05,          # ±5 cm
    display_scale: float = 0.6,
    # Stage 1 (coarse)
    yaw_coarse=( -2.0,  +2.0,  0.5),
    pitch_coarse=(-1.0,  +1.0,  0.5),
    roll_coarse=( -1.0,  +1.0,  0.5),
    # Stage 2 (medium) window around stage-1 best
    yaw_win2=0.5,  pitch_win2=0.5,  roll_win2=0.5,   step2=0.05,
    # Stage 3 (fine) window around stage-2 best
    yaw_win3=0.05, pitch_win3=0.05, roll_win3=0.05,  step3=0.005
):
    # --- Load calib + images ---
    calib = _load_calibration(calib_file)
    K1, D1 = calib["camera_matrix_1"], calib["dist_coeffs_1"]
    K2, D2 = calib["camera_matrix_2"], calib["dist_coeffs_2"]
    R,  T  = calib["R"], calib["T"]

    Lraw = _imread_color(left_path)
    Rraw = _imread_color(right_path)
    if Lraw.shape[:2] != Rraw.shape[:2]:
        raise ValueError("Left/right image sizes must match.")
    h, w = Lraw.shape[:2]
    size = (w, h)

    # --- Click once on RAW ---
    uiL = ZoomClickSession("RAW LEFT", Lraw, display_scale=display_scale)
    uiR = ZoomClickSession("RAW RIGHT", Rraw, display_scale=display_scale)
    print(f"[CLICK] Click {n_clicks} points on LEFT, press ENTER.")
    L_pts_raw = uiL.collect(n_expected=n_clicks, group_label="Click N points (LEFT)")
    print(f"[CLICK] Click the SAME {len(L_pts_raw)} points on RIGHT in the same order, press ENTER.")
    R_pts_raw = uiR.collect(n_expected=len(L_pts_raw), group_label="Click N points (RIGHT)")
    uiL.close(); uiR.close()

    if len(L_pts_raw) < 2 or len(R_pts_raw) < 2:
        raise RuntimeError("Need at least 2 matched points.")

    # --------------- STAGE 1: COARSE ---------------
    yaw_list_1   = list(np.arange(yaw_coarse[0],   yaw_coarse[1]   + 0.5e-6, yaw_coarse[2]))
    pitch_list_1 = list(np.arange(pitch_coarse[0], pitch_coarse[1] + 0.5e-6, pitch_coarse[2]))
    roll_list_1  = list(np.arange(roll_coarse[0],  roll_coarse[1]  + 0.5e-6, roll_coarse[2]))

    print("\n===== STAGE 1: COARSE =====")
    res1, best_epi_1, best_span_1 = _eval_grid_given_clicks(
        K1, D1, K2, D2, R, T, size, L_pts_raw, R_pts_raw,
        yaw_list_1, pitch_list_1, roll_list_1, rectify_alpha, uwb_span_m
    )
    best_combined_1 = pick_best_combined(res1, uwb_span_m=uwb_span_m, span_tol_m=span_tol_m)

    pivot_1 = best_combined_1 or best_span_1 or best_epi_1
    print(f"[STAGE 1] pivot @ yaw={pivot_1['dyaw']:+.3f}, pitch={pivot_1['dpitch']:+.3f}, roll={pivot_1['droll']:+.3f}")

    # --------------- STAGE 2: MEDIUM ---------------
    print("\n===== STAGE 2: MEDIUM =====")
    yaw_list_2   = _mk_list(pivot_1["dyaw"],   yaw_win2,   step2)
    pitch_list_2 = _mk_list(pivot_1["dpitch"], pitch_win2, step2)
    roll_list_2  = _mk_list(pivot_1["droll"],  roll_win2,  step2)

    res2, best_epi_2, best_span_2 = _eval_grid_given_clicks(
        K1, D1, K2, D2, R, T, size, L_pts_raw, R_pts_raw,
        yaw_list_2, pitch_list_2, roll_list_2, rectify_alpha, uwb_span_m
    )
    best_combined_2 = pick_best_combined(res2, uwb_span_m=uwb_span_m, span_tol_m=span_tol_m)

    pivot_2 = best_combined_2 or best_span_2 or best_epi_2
    print(f"[STAGE 2] pivot @ yaw={pivot_2['dyaw']:+.3f}, pitch={pivot_2['dpitch']:+.3f}, roll={pivot_2['droll']:+.3f}")

    # --------------- STAGE 3: FINE ---------------
    print("\n===== STAGE 3: FINE =====")
    yaw_list_3   = _mk_list(pivot_2["dyaw"],   yaw_win3,   step3)
    pitch_list_3 = _mk_list(pivot_2["dpitch"], pitch_win3, step3)
    roll_list_3  = _mk_list(pivot_2["droll"],  roll_win3,  step3)

    res3, best_epi_3, best_span_3 = _eval_grid_given_clicks(
        K1, D1, K2, D2, R, T, size, L_pts_raw, R_pts_raw,
        yaw_list_3, pitch_list_3, roll_list_3, rectify_alpha, uwb_span_m
    )
    best_combined_3 = pick_best_combined(res3, uwb_span_m=uwb_span_m, span_tol_m=span_tol_m)

    # ----------- Reporting final stage -----------
    print("\n=== Stage 3 — Best by epipolar ===")
    print(f"  yaw={best_epi_3['dyaw']:+.4f}°, pitch={best_epi_3['dpitch']:+.4f}°, roll={best_epi_3['droll']:+.4f}°")
    print(f"  mean|dy| = {best_epi_3['mean_abs_dy']:.6f} px, span = {best_epi_3['span_m']:.6f} m")

    if best_span_3 is not None:
        print("\n=== Stage 3 — Best by span ===")
        print(f"  yaw={best_span_3['dyaw']:+.4f}°, pitch={best_span_3['dpitch']:+.4f}°, roll={best_span_3['droll']:+.4f}°")
        print(f"  span = {best_span_3['span_m']:.6f} m  (UWB={uwb_span_m:.6f} m  ->  |Δ|={best_span_3['span_err']:.6f} m)")

    if best_combined_3 is not None:
        print("\n=== Stage 3 — Best COMBINED (±{:.0f} mm) ===".format(span_tol_m*1000))
        print(f"  reason: {best_combined_3.get('_combined_reason','')}")
        print(f"  yaw={best_combined_3['dyaw']:+.4f}°, pitch={best_combined_3['dpitch']:+.4f}°, roll={best_combined_3['droll']:+.4f}°")
        print(f"  mean|dy| = {best_combined_3['mean_abs_dy']:.6f} px | span = {best_combined_3['span_m']:.6f} m"
              f" | |Δspan|={best_combined_3.get('span_err', float('nan')):.6f} m")

    return {
        "stage1": {"results": res1, "best_epi": best_epi_1, "best_span": best_span_1, "best_combined": best_combined_1},
        "stage2": {"results": res2, "best_epi": best_epi_2, "best_span": best_span_2, "best_combined": best_combined_2},
        "stage3": {"results": res3, "best_epi": best_epi_3, "best_span": best_span_3, "best_combined": best_combined_3}
    }


# ------------------------------- CLI ----------------------------------

if __name__ == "__main__":
    # --- Inputs you’ll change ---
    calib_file = "../Calibration/stereoscopic_calibration/stereo_calibration_output/final_stereo_calibration_V3.pkl"
    left_path  = "left_input_static/L_P1_TL1.jpg"
    right_path = "right_input_static/R_P1_TL1.jpg"

    N_CLICKS = 6
    UWB_SPAN_M = 7.843     # set None to ignore span constraints
    SPAN_TOL_M = 0.05      # ±5 cm

    out = coarse_to_fine_search_ypr(
        calib_file=calib_file,
        left_path=left_path,
        right_path=right_path,
        n_clicks=N_CLICKS,
        rectify_alpha=0.0,
        uwb_span_m=UWB_SPAN_M,
        span_tol_m=SPAN_TOL_M,
        display_scale=0.6,
        # Stage 1 coarse ranges (min, max, step = 0.5°)
        yaw_coarse=(0, 3.0, 0.25),
        pitch_coarse=(-1.50, 1.50, 0.25),
        roll_coarse=(-1.5, 1.5, 0.25),
        # Stage 2 window (±0.5°) with 0.05° steps
        yaw_win2=0.2, pitch_win2=0.2, roll_win2=0.2, step2=0.02,
        # Stage 3 window (±0.05°) with 0.005° steps
        yaw_win3=0.02, pitch_win3=0.02, roll_win3=0.02, step3=0.002
    )

    # Visualize final (stage 3)
    s3 = out["stage3"]
    results3 = s3["results"]
    best_combined = s3["best_combined"]
    best_span = s3["best_span"]

    # Coarse plots (no dots/markers besides colors)
    coarse_results = out["stage1"]["results"]
    plot_coarse_3d_pair(coarse_results, uwb_span_m=UWB_SPAN_M, alpha=0.9, s=6)

    # Use the yaw slice from the coarse set, but only mark the *true* final best (from Stage 3)
    final_best = out["stage3"]["best_combined"]  # or whichever you consider "eventual best"
    plot_coarse_slice_smoothed(
        coarse_results,
        yaw_target=final_best["dyaw"],  # slice near the fine yaw, but using COARSE data
        uwb_span_m=UWB_SPAN_M,
        levels=64,
        final_best=final_best
    )

    # 2) 2D slice at the best-by-combined yaw (or best-by-span/epipolar fallback)
    yaw_star = None
    if best_combined is not None:
        yaw_star = best_combined["dyaw"]

    elif best_span is not None:
        yaw_star = best_span["dyaw"]

    else:
        yaw_star = s3["best_epi"]["dyaw"]


    # Quick summaries
    be = s3["best_epi"]
    print("\n[SUMMARY] Stage-3 Best-by-epipolar:",
          f"yaw={be['dyaw']:+.3f}, pitch={be['dpitch']:+.3f}, roll={be['droll']:+.3f}",
          f"| mean|dy|={be['mean_abs_dy']:.4f} px | span={be['span_m']:.4f} m")

    if s3["best_span"] is not None:
        bs = s3["best_span"]
        print("[SUMMARY] Stage-3 Best-by-span:",
              f"yaw={bs['dyaw']:+.3f}, pitch={bs['dpitch']:+.3f}, roll={bs['droll']:+.3f}",
              f"| span={bs['span_m']:.4f} m (target={UWB_SPAN_M:.4f} m, |Δ|={bs['span_err']:.4f} m)",
              f"| mean|dy|={bs['mean_abs_dy']:.4f} px")

    if s3["best_combined"] is not None:
        bc = s3["best_combined"]
        print("[SUMMARY] Stage-3 Best-by-COMBINED:",
              f"yaw={bc['dyaw']:+.3f}, pitch={bc['dpitch']:+.3f}, roll={bc['droll']:+.3f}",
              f"| mean|dy|={bc['mean_abs_dy']:.4f} px | span={bc['span_m']:.4f} m",
              f"| |Δspan|={bc.get('span_err', float('nan')):.4f} m")
