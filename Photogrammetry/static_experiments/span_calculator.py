# span_probe.py
"""
Manual two-point span measurement on a rectified stereo pair.
- Click two points on LEFT in order (A, B), press Enter.
- Click the same two points on RIGHT in the same order, press Enter.
Prints per-point epipolar y-error (|y_L - y_R|), disparities, baseline/focal,
in-frame pixel spans in each view, and the 3D Euclidean span between A and B.

IMPORTANT: Rectification is IDENTICAL to your gridwise accuracy test:
  cv2.stereoRectify(flags=CALIB_FIX_INTRINSIC, alpha=<arg>)
  + initUndistortRectifyMap + remap

Depends on:
- triangulate_points(...) from your stereo utils (only triangulation is imported)
- subpixel_cross_center(...) if you enable refine_subpix
"""

import pickle
from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np

# --- import only what we still need from your utils ---
from Photogrammetry.stereo_photogrammetry_utils import triangulate_points
from Photogrammetry.marker_detection.marker_detection_utils import subpixel_cross_center

def rodrigues_axis_angle(axis: str, angle_deg: float) -> np.ndarray:
    """Return ΔR for a rotation of angle_deg about x/y/z axis in the LEFT (cam1) frame."""
    axis = axis.lower()
    if axis not in ("x", "y", "z"):
        raise ValueError("axis must be 'x', 'y', or 'z'")
    v = {"x": np.array([1.0, 0.0, 0.0]),
         "y": np.array([0.0, 1.0, 0.0]),
         "z": np.array([0.0, 0.0, 1.0])}[axis]
    theta = np.deg2rad(float(angle_deg))
    R_delta, _ = cv2.Rodrigues(v * theta)   # axis*angle -> 3x3
    return R_delta

def adjust_R_yaw_pitch_roll(R: np.ndarray, yaw_deg: float=0.0, pitch_deg: float=0.0, roll_deg: float=0.0) -> np.ndarray:
    """
    Apply ΔR (yaw about +Y, pitch about +X, roll about +Z) in the LEFT camera frame:
    R' = ΔR * R.
    """
    Rz = rodrigues_axis_angle("z", roll_deg)
    Rx = rodrigues_axis_angle("x", pitch_deg)
    Ry = rodrigues_axis_angle("y", yaw_deg)
    # Choose an order; yaw->pitch->roll is fine for small angles:
    dR = Ry @ Rx @ Rz
    return dR @ R

def axis_angle_from_R(R: np.ndarray) -> tuple[float, np.ndarray]:
    """Return (angle_deg, unit_axis[3]) from a rotation matrix (for quick diagnostics)."""
    rvec, _ = cv2.Rodrigues(R)
    angle = np.linalg.norm(rvec)
    axis = (rvec.flatten() / angle) if angle > 1e-12 else np.array([0.0, 0.0, 1.0])
    return np.rad2deg(angle), axis

# --------------------------- I/O helpers ---------------------------

def _load_calibration(calib_pkl_path: str) -> Dict:
    with open(calib_pkl_path, "rb") as f:
        calib = pickle.load(f)
    required = ["camera_matrix_1","dist_coeffs_1","camera_matrix_2","dist_coeffs_2","R","T"]
    miss = [k for k in required if k not in calib]
    if miss:
        raise KeyError(f"Calibration file is missing keys: {miss}")
    return calib

def _imread_color(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Could not read image: {path}")
    return img


# --------------------------- rectification (IDENTICAL to gridwise) ---------------------------

def rectify_stereo_pair(
    left: np.ndarray, right: np.ndarray,
    mtx1: np.ndarray, dist1: np.ndarray,
    mtx2: np.ndarray, dist2: np.ndarray,
    R: np.ndarray, T: np.ndarray,
    *,
    alpha: float = 0.0,              # same free-scaling param you use in gridwise
    contrast_alpha: float = 2.0,     # same visualization tweak used there
    brightness_beta: float = 0.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns left_rect, right_rect, P1, P2, R1, R2, Q using the exact same pipeline as your gridwise code."""
    h, w = left.shape[:2]
    image_size = (w, h)

    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        mtx1, dist1, mtx2, dist2, image_size, R, T,
        flags=cv2.CALIB_FIX_INTRINSIC, alpha=alpha
    )
    map1x, map1y = cv2.initUndistortRectifyMap(mtx1, dist1, R1, P1, image_size, cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(mtx2, dist2, R2, P2, image_size, cv2.CV_32FC1)

    left_rect  = cv2.remap(left,  map1x, map1y, cv2.INTER_LINEAR)
    right_rect = cv2.remap(right, map2x, map2y, cv2.INTER_LINEAR)

    # same contrast/brightness tweak used in your gridwise script
    left_rect  = cv2.convertScaleAbs(left_rect,  alpha=contrast_alpha, beta=brightness_beta)
    right_rect = cv2.convertScaleAbs(right_rect, alpha=contrast_alpha, beta=brightness_beta)

    return left_rect, right_rect, P1, P2, R1, R2, Q


def check_rectified_cameras(P1: np.ndarray, P2: np.ndarray, tag="[RECT]"):
    fx1 = float(P1[0, 0]); fy1 = float(P1[1, 1]); cx1 = float(P1[0, 2]); cy1 = float(P1[1, 2])
    fx2 = float(P2[0, 0]); fy2 = float(P2[1, 1]); cx2 = float(P2[0, 2]); cy2 = float(P2[1, 2])
    baseline_est = -float(P2[0, 3]) / fx2 if fx2 != 0 else np.nan  # baseline = -Tx/fx for rectified pair

    print(f"{tag} P1 fx,fy={fx1:.2f},{fy1:.2f}  cx,cy={cx1:.2f},{cy1:.2f}")
    print(f"{tag} P2 fx,fy={fx2:.2f},{fy2:.2f}  cx,cy={cx2:.2f},{cy2:.2f}")
    print(f"{tag} Estimated baseline from P2: {baseline_est:.6f} m")
    if abs(fx1 - fx2) > 1e-3 or abs(fy1 - fy2) > 1e-3:
        print(f"{tag} WARN: fx/fy differ between P1 and P2 (should be equal after stereoRectify).")
    if abs(cy1 - cy2) > 0.5:
        print(f"{tag} WARN: cy differs between views by {abs(cy1 - cy2):.3f} px (should be ~0 after rectification).")


# ---------------------------- Zoomable click UI ----------------------------

class ZoomClickSession:
    """
    Zoomable/pannable click UI with rectified pixel coordinates.
    Keys: left-click add, wheel zoom, right-drag pan, 'u' undo, Enter accept, Esc cancel.
    Optional subpixel refinement via subpixel_cross_center.
    """
    def __init__(
        self,
        window_name: str,
        image_bgr: np.ndarray,
        *,
        display_scale: float = 0.6,
        refine_subpix: bool = False,
        refine_patch: int = 25,
        refine_max_shift_px: float = 4.0,
    ):
        self.win = window_name
        self.base = image_bgr
        self.gray = cv2.cvtColor(self.base, cv2.COLOR_BGR2GRAY)

        H, W = self.base.shape[:2]
        self.init_w = max(640, int(W * display_scale))
        self.init_h = max(360, int(H * display_scale))

        self.zoom = 1.0
        self.cx, self.cy = W / 2.0, H / 2.0
        self.dragging = False
        self.last_mouse = (0, 0)

        self.points: List[Tuple[float, float]] = []

        self.refine = bool(refine_subpix)
        self.refine_patch = int(refine_patch)
        self.refine_max_shift_px = float(refine_max_shift_px)

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
            px, py = ix, iy
            if self.refine:
                sp = subpixel_cross_center(self.gray, (int(round(ix)), int(round(iy))), patch=self.refine_patch)
                if sp is not None:
                    rx, ry = float(sp[0]), float(sp[1])
                    if abs(rx - ix) <= self.refine_max_shift_px and abs(ry - iy) <= self.refine_max_shift_px:
                        px, py = rx, ry
            self.points.append((px, py))
            self._redraw()
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.dragging = True; self.last_mouse = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            dx = x - self.last_mouse[0]; dy = y - self.last_mouse[1]
            _, _, vw, vh = self._roi_dims(); ww, hh = self._win_size()
            self.cx -= dx * (vw / max(1, ww)); self.cy -= dy * (vh / max(1, hh))
            self.last_mouse = (x, y); self._redraw()
        elif event == cv2.EVENT_RBUTTONUP:
            self.dragging = False
        elif event == cv2.EVENT_MOUSEWHEEL:
            self._zoom_at(x, y, 1.25 if flags > 0 else 1/1.25)

    def _redraw(self):
        x0, y0, vw, vh = self._roi_dims()
        roi = self.base[y0:y0+vh, x0:x0+vw]
        ww, hh = self._win_size()
        disp = cv2.resize(roi, (ww, hh), interpolation=cv2.INTER_AREA)

        for i, (px, py) in enumerate(self.points):
            dx, dy = self._image_to_disp(px, py)
            if 0 <= dx < ww and 0 <= dy < hh:
                cv2.drawMarker(disp, (dx, dy), (0,255,0), markerType=cv2.MARKER_CROSS, markerSize=18, thickness=2)
                cv2.putText(disp, str(i), (dx+8, dy-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA)
                cv2.putText(disp, str(i), (dx+8, dy-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)

        status = f"zoom={self.zoom:.2f}  pts={len(self.points)}"
        cv2.putText(disp, status, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA)
        cv2.putText(disp, status, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)

        cv2.imshow(self.win, disp)
        self._last_drawn_size = (ww, hh)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('+'), ord('=')): self._zoom_at(ww//2, hh//2, 1.25)
        elif key in (ord('-'), ord('_')): self._zoom_at(ww//2, hh//2, 1/1.25)

    def collect(self, n_expected: Optional[int], group_label: str, reset: bool = True) -> List[Tuple[float, float]]:
        if reset:
            self.points = []
            self._redraw()
        while True:
            title = f"{self.win} — {group_label} | clicked {len(self.points)}"
            if n_expected is not None: title += f"/{n_expected}"
            cv2.setWindowTitle(self.win, title)
            key = cv2.waitKey(20) & 0xFF
            if key in (ord('u'), ord('U')):
                if self.points: self.points.pop(); self._redraw()
            elif key in (13, 10):  # Enter
                break
            elif key == 27:  # Esc
                raise KeyboardInterrupt("User cancelled.")
            if n_expected is not None and len(self.points) >= n_expected:
                break
        return list(self.points)

    def close(self):
        try: cv2.destroyWindow(self.win)
        except cv2.error: pass


# --------------------------- main span function ---------------------------

def measure_span_two_points(
    calib_file: str,
    left_image_path: str,
    right_image_path: str,
    *,
    rectify_alpha: float = 0.0,
    display_scale: float = 0.6,
    refine_subpix: bool = False,
    contrast_alpha: float = 1.0,
    brightness_beta: float = 0.0,
    # >>> NEW: small adjustment knobs (degrees) <<<
    delta_yaw_deg: float = 0.0,    # about +Y (changes x-disparity)
    delta_pitch_deg: float = 0.0,  # about +X
    delta_roll_deg: float = 0.0    # about +Z
) -> float:
    ...
    # Load calibration
    calib = _load_calibration(calib_file)
    K1, D1 = calib["camera_matrix_1"], calib["dist_coeffs_1"]
    K2, D2 = calib["camera_matrix_2"], calib["dist_coeffs_2"]
    R,  T  = calib["R"], calib["T"]

    # >>> NEW: apply small ΔR in the LEFT-camera frame
    R_adj = adjust_R_yaw_pitch_roll(R, yaw_deg=delta_yaw_deg,
                                       pitch_deg=delta_pitch_deg,
                                       roll_deg=delta_roll_deg)

    # Tiny sanity prints so you know what happened
    ang0, ax0 = axis_angle_from_R(R)
    ang1, ax1 = axis_angle_from_R(R_adj)
    print(f"[ΔR] Original R:  angle={ang0:.4f}° axis={ax0}")
    print(f"[ΔR] Adjusted R:  angle={ang1:.4f}° axis={ax1}")
    print(f"[ΔR] Applied (yaw, pitch, roll) = ({delta_yaw_deg:.4f}°, {delta_pitch_deg:.4f}°, {delta_roll_deg:.4f}°) in LEFT frame")

    # Read images
    Lraw = _imread_color(left_image_path)
    Rraw = _imread_color(right_image_path)

    # Rectify with R_adj (NOT the original R)
    Lr, Rr, P1, P2, R1, R2, Q = rectify_stereo_pair(
        Lraw, Rraw, K1, D1, K2, D2, R_adj, T,
        alpha=rectify_alpha,
        contrast_alpha=contrast_alpha,
        brightness_beta=brightness_beta
    )
    check_rectified_cameras(P1, P2, tag="[RECT]")

    # Effective baseline/focal (sanity)
    f_px = float(P1[0, 0])
    baseline_rect_m = float(-P2[0, 3] / P2[0, 0])  # meters
    baseline_calib_m = float(np.linalg.norm(T))
    print(f"\n[Rectify] focal ≈ {f_px:.1f} px | baseline_rect ≈ {baseline_rect_m:.4f} m | "
          f"baseline_from_calib_T ≈ {baseline_calib_m:.4f} m")

    # Click 2 points left, then 2 points right
    left_ui  = ZoomClickSession("Rectified LEFT",  Lr, display_scale=display_scale, refine_subpix=refine_subpix)
    right_ui = ZoomClickSession("Rectified RIGHT", Rr, display_scale=display_scale, refine_subpix=refine_subpix)

    print("[LEFT] Click TWO points (A, then B), press ENTER.")
    Lpts = left_ui.collect(n_expected=2, group_label="Two points (LEFT)", reset=True)
    print("[RIGHT] Click the SAME TWO points in the SAME ORDER (A, then B), press ENTER.")
    Rpts = right_ui.collect(n_expected=2, group_label="Two points (RIGHT)", reset=True)

    left_ui.close(); right_ui.close()

    if len(Lpts) != 2 or len(Rpts) != 2:
        raise RuntimeError("Need exactly 2 points on LEFT and 2 points on RIGHT.")

    # ---- Diagnostics: coordinates and spans in pixels ----
    (AxL, AyL), (BxL, ByL) = Lpts[0], Lpts[1]
    (AxR, AyR), (BxR, ByR) = Rpts[0], Rpts[1]
    span_L_px = float(np.hypot(BxL - AxL, ByL - AyL))
    span_R_px = float(np.hypot(BxR - AxR, ByR - AyR))

    print("\n[PIXELS] Clicked coordinates:")
    print(f"  LEFT  A=({AxL:.2f}, {AyL:.2f})  B=({BxL:.2f}, {ByL:.2f})  |AB|={span_L_px:.3f} px")
    print(f"  RIGHT A=({AxR:.2f}, {AyR:.2f})  B=({BxR:.2f}, {ByR:.2f})  |AB|={span_R_px:.3f} px")

    # Per-point epipolar y-errors & disparities
    for i, (Ll, Rr_) in enumerate(zip(Lpts, Rpts)):
        lx, ly = Ll; rx, ry = Rr_
        y_err = (ly - ry)
        disp  = (lx - rx)  # same convention used elsewhere
        print(f"[pair {i}] y-epipolar yL-yR = {y_err:.3f} px   disparity (xL-xR) = {disp:.3f} px")

    # ---- Triangulate two 3D points (using your triangulator) ----
    L_labeled = [(Lpts[0][0], Lpts[0][1], "A"), (Lpts[1][0], Lpts[1][1], "B")]
    tri = triangulate_points(L_labeled, Rpts, P1, P2)
    P3D = np.array([p[0] if isinstance(p, tuple) else p for p in tri], dtype=float)  # (2,3)

    span_m = float(np.linalg.norm(P3D[1] - P3D[0]))
    print(f"\n[SPAN] Euclidean distance |A-B| = {span_m:.6f} m")
    return span_m


# ------------------------------- CLI ----------------------------------
if __name__ == "__main__":
    delta_yaw = 1.285
    delta_roll = 0.45
    delta_pitch = 0.45
    span = measure_span_two_points(
        calib_file="../Calibration/stereoscopic_calibration/stereo_calibration_output/final_stereo_calibration_V3.pkl",
        left_image_path="left_input_static/L_P1_TL1.jpg",
        right_image_path="right_input_static/R_P1_TL1.jpg",
        rectify_alpha=0,      # keep identical to gridwise run
        delta_yaw_deg=delta_yaw,
        delta_roll_deg=delta_roll,
        delta_pitch_deg=delta_pitch
    )
    print(f"\nDone. Measured span = {span:.6f} m")
    uwb_span = 5.042
    print(f"There is a {(span-uwb_span)/uwb_span * 100} percent difference in span")
    if (span-uwb_span)/uwb_span * 100 > 0:
        print("Increase yaw angle")
    else:
        print("Decrease yaw angle")