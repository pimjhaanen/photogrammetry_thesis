# span_probe_fisheye.py
"""
Manual two-point span measurement on a rectified fisheye stereo pair.

- Click two points on LEFT in order (A, B), press Enter.
- Click the same two points on RIGHT in the same order, press Enter.
- Prints per-point epipolar y-error (|yL - yR|), disparities, focal/baseline,
  and the 3D Euclidean span |A - B| in meters.

Requirements:
- Fisheye stereo calibration pickle with keys:
  ["camera_matrix_1","dist_coeffs_1","camera_matrix_2","dist_coeffs_2","R","T"]
- OpenCV 4.5.x (Python bindings vary; rectifier below handles overloads)
"""

import os
import pickle
from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np

# --- Try to use your project triangulator; fall back if missing ---
try:
    from Photogrammetry.stereo_photogrammetry_utils import triangulate_points as _triangulate_points_project
    def triangulate_points(pts1, pts2, P1, P2):
        return _triangulate_points_project(pts1, pts2, P1, P2)
except Exception:
    def triangulate_points(pts1, pts2, P1, P2):
        if len(pts1) != len(pts2):
            raise ValueError(f"pts1 and pts2 must have same length; got {len(pts1)} vs {len(pts2)}")
        L = np.array([p[:2] for p in pts1], dtype=np.float32).T
        R = np.array([p[:2] for p in pts2], dtype=np.float32).T
        X4 = cv2.triangulatePoints(P1, P2, L, R)
        X3 = (X4[:3, :] / X4[3, :]).T
        if any(len(p) == 3 for p in pts1):
            labels = [p[2] if len(p) == 3 else None for p in pts1]
            return [(X3[i], labels[i]) for i in range(len(pts1))]
        return [X3[i] for i in range(len(pts1))]

# Optional subpixel refiner from your project (if present)
try:
    from Photogrammetry.marker_detection.marker_detection_utils import subpixel_cross_center
except Exception:
    subpixel_cross_center = None


# ---------------- FISHEYE stereo rectification (works across 4.5.x) ----------------

def _fisheye_rectify_via_pinhole_maps(
    K1, D1, K2, D2, R, T, size, balance=0.0, fov_scale=1.0
):
    """
    Robust fisheye stereo rectification:
      1) Convert fisheye intrinsics -> 'pinhole-like' newK1/newK2.
      2) Run cv2.stereoRectify (pinhole) to get R1,R2,P1,P2,Q.
      3) Build maps with cv2.fisheye.initUndistortRectifyMap using original fisheye K,D and R1,P1 / R2,P2.
    Returns: (R1,R2,P1,P2,Q), (map1x,map1y,map2x,map2y)
    """
    import numpy as np, cv2

    # sanitize
    K1 = np.asarray(K1, dtype=np.float64).reshape(3,3)
    K2 = np.asarray(K2, dtype=np.float64).reshape(3,3)
    D1 = np.asarray(D1, dtype=np.float64).reshape(4,1)
    D2 = np.asarray(D2, dtype=np.float64).reshape(4,1)
    R  = np.asarray(R,  dtype=np.float64).reshape(3,3)
    T  = np.asarray(T,  dtype=np.float64).reshape(3,1)
    size = (int(size[0]), int(size[1]))  # (w,h)

    # 1) fisheye -> new "pinhole-like" intrinsics
    newK1 = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K1, D1, size, np.eye(3), balance=float(balance), fov_scale=float(fov_scale)
    )
    newK2 = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K2, D2, size, np.eye(3), balance=float(balance), fov_scale=float(fov_scale)
    )

    # 2) pinhole stereoRectify (dist set to None/zeros)
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        newK1, None, newK2, None, size, R, T,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0, newImageSize=size
    )

    # 3) fisheye maps using original K,D and rect transforms; pass a 3x3 P (intrinsics)
    K1rect = P1[:,:3].astype(np.float64)
    K2rect = P2[:,:3].astype(np.float64)

    map1x, map1y = cv2.fisheye.initUndistortRectifyMap(
        K1, D1, R1, K1rect, size, cv2.CV_32FC1
    )
    map2x, map2y = cv2.fisheye.initUndistortRectifyMap(
        K2, D2, R2, K2rect, size, cv2.CV_32FC1
    )
    return (R1, R2, P1, P2, Q), (map1x, map1y, map2x, map2y)



# ---------------------------- I/O helpers ----------------------------

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


# ---------------------------- Zoom UI ----------------------------

class ZoomClickSession:
    """
    Zoom/pan click UI on rectified frames.
    Keys: left-click add, wheel zoom, right-drag pan, 'u' undo, Enter accept, Esc cancel.
    Optional subpixel refinement if subpixel_cross_center is available.
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

        self.refine = bool(refine_subpix and (subpixel_cross_center is not None))
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
            if self.refine and subpixel_cross_center is not None:
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
            elif key in (13, 10):
                break
            elif key == 27:
                raise KeyboardInterrupt("User cancelled.")
            if n_expected is not None and len(self.points) >= n_expected:
                break
        return list(self.points)

    def close(self):
        try: cv2.destroyWindow(self.win)
        except cv2.error: pass


# --------------------------- main span function ---------------------------

def measure_span_two_points_fisheye(
    calib_file: str,
    left_image_path: str,
    right_image_path: str,
    *,
    rectify_balance: float = 0.0,
    rectify_fov_scale: float = 1.0,
    zero_disparity: bool = True,
    display_scale: float = 0.6,
    refine_subpix: bool = False,
) -> float:
    """
    Returns the 3D span (meters) between two clicked points on a
    fisheye-rectified stereo pair, and prints diagnostics.
    """
    calib = _load_calibration(calib_file)
    K1, D1 = calib["camera_matrix_1"], calib["dist_coeffs_1"]
    K2, D2 = calib["camera_matrix_2"], calib["dist_coeffs_2"]
    R, T   = calib["R"], calib["T"]

    L = _imread_color(left_image_path)
    Rimg = _imread_color(right_image_path)
    if L.shape[:2] != Rimg.shape[:2]:
        raise ValueError("Left/right images must have the same size.")
    h, w = L.shape[:2]
    size = (w, h)

    (_, _, P1, P2, _), maps = _fisheye_rectify_via_pinhole_maps(
        K1, D1, K2, D2, R, T, size,
        balance=rectify_balance, fov_scale=rectify_fov_scale
    )

    map1x, map1y, map2x, map2y = maps
    Lr = cv2.remap(L,    map1x, map1y, cv2.INTER_LINEAR)
    Rr = cv2.remap(Rimg, map2x, map2y, cv2.INTER_LINEAR)

    # Effective focal & baselines (sanity)
    f_px = float(P1[0, 0])
    baseline_rect_m = float(-P2[0, 3] / max(P2[0, 0], 1e-9))  # meters
    baseline_calib_m = float(np.linalg.norm(T))
    print(f"\n[Fisheye Rectify] focal ≈ {f_px:.1f} px | baseline_rect ≈ {baseline_rect_m:.6f} m | "
          f"baseline_from_T ≈ {baseline_calib_m:.6f} m")

    # Click points
    left_ui  = ZoomClickSession("Rectified LEFT (fisheye)",  Lr, display_scale=display_scale, refine_subpix=refine_subpix)
    right_ui = ZoomClickSession("Rectified RIGHT (fisheye)", Rr, display_scale=display_scale, refine_subpix=refine_subpix)

    print("[LEFT] Click TWO points (A, then B), press ENTER.")
    Lpts = left_ui.collect(n_expected=2, group_label="Two points (LEFT)", reset=True)
    print("[RIGHT] Click the SAME TWO points in the SAME ORDER (A, then B), press ENTER.")
    Rpts = right_ui.collect(n_expected=2, group_label="Two points (RIGHT)", reset=True)

    left_ui.close(); right_ui.close()

    if len(Lpts) != 2 or len(Rpts) != 2:
        raise RuntimeError("Need exactly 2 points on LEFT and 2 points on RIGHT.")

    # Epipolar y-errors & disparities
    for i in range(2):
        lx, ly = Lpts[i]
        rx, ry = Rpts[i]
        y_err = abs(ly - ry)
        disp  = (lx - rx)
        print(f"[pair {i}]  y-epipolar |yL-yR| = {y_err:.3f} px   disparity (xL-xR) = {disp:.3f} px")

    # Triangulate 3D points and compute span
    L_labeled = [(Lpts[0][0], Lpts[0][1], "A"), (Lpts[1][0], Lpts[1][1], "B")]
    tri = triangulate_points(L_labeled, Rpts, P1, P2)
    P3D = np.array([p[0] if isinstance(p, tuple) else p for p in tri], dtype=float)  # (2,3)
    span_m = float(np.linalg.norm(P3D[1] - P3D[0]))
    print(f"\n[SPAN]  Euclidean distance |A-B| = {span_m:.6f} m")
    return span_m


# ------------------------------- CLI example ----------------------------------
if __name__ == "__main__":
    span = measure_span_two_points_fisheye(
        calib_file="../Calibration/stereoscopic_calibration/stereo_calibration_output/stereo_calibration_ireland_fisheye.pkl",
        left_image_path="left_input_static/L_P1_PL1.jpg",
        right_image_path="right_input_static/R_P1_PL1.jpg",
        rectify_balance=0.0,
        rectify_fov_scale=1.0,
        zero_disparity=True,
        display_scale=0.6,
        refine_subpix=False,
    )
    print(f"\nDone. Measured span = {span:.6f} m")
