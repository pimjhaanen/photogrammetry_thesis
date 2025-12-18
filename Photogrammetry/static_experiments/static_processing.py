"""
Manual static stereo test with zoomable clicking + training patch export.
CSV-only output (no plotting). Each group starts fresh.

Adds:
- SKIP support:
    LEFT: short right-click = skip this slot (auto-advances)
    RIGHT: mirrored skips auto-applied; you can still skip extra occluded slots
    'u' undo (point or skip), 'r' toggle subpixel refine, right-drag pan, wheel zoom
- Small yaw/pitch/roll adjustment (degrees) applied to calibration R:
    R' = dR @ R   (dR in LEFT/world frame; yaw +Y, pitch +X, roll +Z)
- Reconciliation step to handle extra RIGHT-side skips without mismatches.
"""

import os
import json
import pickle
import time
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
import pandas as pd

from Photogrammetry.stereo_photogrammetry_utils import _stereo_rectify_maps, triangulate_points
from Photogrammetry.marker_detection.marker_detection_utils import subpixel_cross_center

# ------------------------------ Video helpers ---------------------------------
_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv"}

def _is_video_path(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext in _VIDEO_EXTS

def _frame_from_video(video_path: str, frame_idx: int = None, time_sec: float = None) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")
    try:
        if frame_idx is not None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_idx))
        elif time_sec is not None:
            cap.set(cv2.CAP_PROP_POS_MSEC, float(time_sec) * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            raise IOError(f"Failed to read frame from {video_path}")
        return frame
    finally:
        cap.release()

def _load_image_or_video_frame(path: str, frame_idx: int = None, time_sec: float = None) -> np.ndarray:
    if _is_video_path(path):
        return _frame_from_video(path, frame_idx=frame_idx, time_sec=time_sec)
    else:
        return _imread_color(path)


# ------------------------------ Helper I/O ------------------------------------

def _load_calibration(calib_pkl_path: str):
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

def _apply_brightness_contrast(img: np.ndarray, contrast: float, brightness: float) -> np.ndarray:
    """Apply brightness/contrast adjustments on-the-fly."""
    return cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)


# -------------------------- Small rotation adjustment -------------------------

def _rodrigues_axis_angle(axis: str, angle_deg: float) -> np.ndarray:
    axis = axis.lower()
    if axis not in ("x", "y", "z"):
        raise ValueError("axis must be 'x','y','z'")
    v = {"x": np.array([1.0, 0.0, 0.0]),
         "y": np.array([0.0, 1.0, 0.0]),
         "z": np.array([0.0, 0.0, 1.0])}[axis]
    theta = np.deg2rad(float(angle_deg))
    R_delta, _ = cv2.Rodrigues(v * theta)   # axis*angle -> 3x3
    return R_delta


def _adjust_R_yaw_pitch_roll_leftframe(R: np.ndarray, *, yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0) -> np.ndarray:
    """
    Apply a small delta rotation dR in the LEFT/world frame:
        R' = dR @ R
    with yaw about +Y, pitch about +X, roll about +Z. Order: yaw -> pitch -> roll.
    """
    Rz = _rodrigues_axis_angle("z", roll_deg)
    Rx = _rodrigues_axis_angle("x", pitch_deg)
    Ry = _rodrigues_axis_angle("y", yaw_deg)
    dR = Ry @ Rx @ Rz
    return dR @ R


# -------------------------- Training patch saving -----------------------------

def _save_patch(img, center_xy, size, out_dir, label, img_id, cam_tag, extra_suffix=""):
    Path(out_dir, label).mkdir(parents=True, exist_ok=True)
    cx, cy = float(center_xy[0]), float(center_xy[1])
    patch = cv2.getRectSubPix(img, (size, size), (cx, cy))
    fname = f"{img_id}_{cam_tag}_{label}{extra_suffix}_{int(round(cx))}x{int(round(cy))}.png"
    fpath = str(Path(out_dir, label, fname))
    cv2.imwrite(fpath, patch)
    return fpath


def _dump_training_from_clicks(Lr, Rr, clicks_dict, out_dir="training_patches", patch=20, img_id="frame"):
    meta = []
    for k, (ptsL, ptsR) in clicks_dict.items():
        label = "STRUT" if k.startswith("strut") else ("LE" if k == "LE" else "CAN")
        suffix = f"_{k}" if k.startswith("strut") else ""
        for p in ptsL:
            path = _save_patch(Lr, p, patch, out_dir, label, img_id, "L", extra_suffix=suffix)
            meta.append({"file": path, "label": label, "group": k, "camera": "L",
                         "cx": float(p[0]), "cy": float(p[1]), "patch": patch, "image_id": img_id})
        for p in ptsR:
            path = _save_patch(Rr, p, patch, out_dir, label, img_id, "R", extra_suffix=suffix)
            meta.append({"file": path, "label": label, "group": k, "camera": "R",
                         "cx": float(p[0]), "cy": float(p[1]), "patch": patch, "image_id": img_id})

    meta_path = Path(out_dir, f"{img_id}_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[✓] Saved {len(meta)} patches to {out_dir}")
    return str(meta_path)


# --------------------------- Reconciliation helper ----------------------------

def _reconcile_correspondences(
    left_points: List[Tuple[float, float]],
    left_skips: List[int],
    right_points: List[Tuple[float, float]],
    right_skips: List[int],
    n_expected: int,
):
    """
    Given ordered clicks for LEFT/RIGHT (each in increasing slot order) and their skip indices,
    compute the intersection of kept slots and return filtered (left_points, right_points, keep_slots).

    - Slots are 0..n_expected-1
    - Points arrays correspond to slots with skips removed, in increasing order.
    - We drop any slots that are skipped on either side (union of skips).
    """
    left_sk = set(int(i) for i in left_skips)
    right_sk = set(int(i) for i in right_skips)
    drop = left_sk | right_sk

    keep_slots = [s for s in range(n_expected) if s not in drop]

    # Build slot lists (complements of skip sets)
    left_point_slots = [s for s in range(n_expected) if s not in left_sk]
    right_point_slots = [s for s in range(n_expected) if s not in right_sk]

    # slot -> index within points arrays
    left_slot_to_idx = {slot: idx for idx, slot in enumerate(left_point_slots)}
    right_slot_to_idx = {slot: idx for idx, slot in enumerate(right_point_slots)}

    # Filter to intersection in slot order
    filt_left, filt_right, final_keep_slots = [], [], []
    for s in keep_slots:
        li = left_slot_to_idx.get(s, None)
        ri = right_slot_to_idx.get(s, None)
        if li is not None and ri is not None:
            filt_left.append(left_points[li])
            filt_right.append(right_points[ri])
            final_keep_slots.append(s)

    return filt_left, filt_right, final_keep_slots


# ---------------------------- Zoomable click UI -------------------------------

class ZoomClickSession:
    """
    Zoomable/pannable click UI with correct coordinate mapping + SKIP support.
    - Left click   = add point at cursor (optionally subpixel-refined)
    - Right drag   = pan
    - Right click (short tap, tiny movement) = SKIP current slot
    - 'u'          = undo last action (point or skip)
    - 'r'          = toggle refine
    - Enter        = accept group
    - Esc          = cancel

    Mirroring: pass mirror_skips=[...] to collect() so the same indices are auto-skipped.
    """
    def __init__(
        self,
        window_name: str,
        image_bgr: np.ndarray,
        *,
        display_scale: float = 0.5,
        refine_subpix: bool = False,
        refine_patch: int = 25,
        refine_max_shift_px: float = 4.0,
    ):
        assert display_scale > 0, "display_scale must be > 0"
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

        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.win, self.init_w, self.init_h)

        # --- New sliders for brightness & contrast ---
        self.brightness = 0
        self.contrast = 10  # represents 1.0× initially (mapped as contrast/10)
        cv2.createTrackbar("Brightness", self.win, 100, 200, lambda v: None)
        cv2.createTrackbar("Contrast", self.win, self.contrast, 30, lambda v: None)

        # Current group state
        self.points: List[Tuple[float, float]] = []
        self.skips: List[int] = []                  # slot indices skipped
        self.history: List[Tuple[str, int]] = []    # ("pt", slot) or ("skip", slot)
        self.slot = 0                                # 0..n_expected-1
        self._n_expected: Optional[int] = None
        self._mirror_skips_set: Optional[set] = None

        # Subpixel options
        self.refine = bool(refine_subpix)
        self.refine_patch = int(refine_patch)
        self.refine_max_shift_px = float(refine_max_shift_px)

        # Right-click gesture tracking for skip vs pan
        self._r_down_time: Optional[float] = None
        self._r_drag_dist: float = 0.0

        cv2.setMouseCallback(self.win, self._on_mouse)
        self._redraw()

    # ---- coords & zoom helpers ----
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

    # ---- register actions ----
    def _register_point(self, ix: float, iy: float):
        if (self._n_expected is not None) and (self.slot >= self._n_expected):
            return
        px, py = ix, iy
        if self.refine:
            sp = subpixel_cross_center(self.gray, (int(round(ix)), int(round(iy))), patch=self.refine_patch)
            if sp is not None:
                rx, ry = float(sp[0]), float(sp[1])
                if abs(rx - ix) <= self.refine_max_shift_px and abs(ry - iy) <= self.refine_max_shift_px:
                    px, py = rx, ry
        self.points.append((px, py))
        self.history.append(("pt", self.slot))
        self.slot += 1
        self._redraw()

    def _register_skip(self):
        if (self._n_expected is not None) and (self.slot >= self._n_expected):
            return
        self.skips.append(self.slot)
        self.history.append(("skip", self.slot))
        self.slot += 1
        print(f"[SKIP] slot {self.slot-1}")
        self._redraw()

    def _maybe_auto_skip_from_mirror(self):
        if self._mirror_skips_set is None or self._n_expected is None:
            return
        while (self.slot < self._n_expected) and (self.slot in self._mirror_skips_set) and (self.slot not in self.skips):
            self._register_skip()

    # ---- event handling ----
    def _on_mouse(self, event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            ix, iy = self._disp_to_image(x, y)
            self._register_point(ix, iy)

        elif event == cv2.EVENT_RBUTTONDOWN:
            self.dragging = True
            self.last_mouse = (x, y)
            self._r_drag_dist = 0.0
            self._r_down_time = time.time()

        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            dx = x - self.last_mouse[0]
            dy = y - self.last_mouse[1]
            self._r_drag_dist += (dx * dx + dy * dy) ** 0.5
            _, _, vw, vh = self._roi_dims()
            ww, hh = self._win_size()
            self.cx -= dx * (vw / max(1, ww))
            self.cy -= dy * (vh / max(1, hh))
            self.last_mouse = (x, y)
            self._redraw()

        elif event == cv2.EVENT_RBUTTONUP:
            # Short right-click with small movement => SKIP
            dt = (time.time() - (self._r_down_time or time.time()))
            if self._r_drag_dist < 5.0 and dt < 0.35:
                self._register_skip()
            self.dragging = False

        elif event == cv2.EVENT_MOUSEWHEEL:
            self._zoom_at(x, y, 1.25 if flags > 0 else 1 / 1.25)

    # ---- draw ----
    def _redraw(self):
        x0, y0, vw, vh = self._roi_dims()
        roi = self.base[y0:y0 + vh, x0:x0 + vw].copy()

        # --- Read trackbar values and apply live adjustment ---
        b = cv2.getTrackbarPos("Brightness", self.win) - 100  # -100..+100
        c = cv2.getTrackbarPos("Contrast", self.win)
        contrast = c / 10.0 if c > 0 else 0.1
        roi = _apply_brightness_contrast(roi, contrast=contrast, brightness=b)

        ww, hh = self._win_size()
        disp = cv2.resize(roi, (ww, hh), interpolation=cv2.INTER_AREA)

        # points
        for i, (px, py) in enumerate(self.points):
            dx, dy = self._image_to_disp(px, py)
            if 0 <= dx < ww and 0 <= dy < hh:
                cv2.drawMarker(disp, (dx, dy), (0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=18, thickness=2)
                cv2.putText(disp, str(i), (dx + 8, dy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(disp, str(i), (dx + 8, dy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        status = f"zoom={self.zoom:.2f}  pts={len(self.points)}  skips={len(self.skips)}  slot={self.slot}"
        if self._n_expected is not None:
            status += f"/{self._n_expected}"
        status += f"  refine={'ON' if self.refine else 'OFF'}"
        cv2.putText(disp, status, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(disp, status, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow(self.win, disp)
        self._last_drawn_size = (ww, hh)

    # ---- public API ----
    def collect(
        self,
        n_expected: Optional[int],
        group_label: str,
        reset: bool = True,
        mirror_skips: Optional[List[int]] = None
    ) -> Tuple[List[Tuple[float, float]], List[int]]:
        """
        Returns (points_list, skip_indices).

        If mirror_skips is provided, those slot indices are auto-skipped so the user
        only clicks the remaining slots.
        """
        if reset:
            self.points = []
            self.skips = []
            self.history = []
            self.slot = 0
            self._redraw()

        self._n_expected = n_expected
        self._mirror_skips_set = set(mirror_skips) if mirror_skips else None

        while True:
            # Auto-apply mirrored skips before waiting for input
            self._maybe_auto_skip_from_mirror()

            title = f"{self.win} — {group_label} | pts={len(self.points)} skips={len(self.skips)}"
            if n_expected is not None:
                title += f"  [{self.slot}/{n_expected}]"
            title += f"  [refine={'ON' if self.refine else 'OFF'}]"
            cv2.setWindowTitle(self.win, title)

            key = cv2.waitKey(20) & 0xFF

            if key in (ord('u'), ord('U')):
                if self.history:
                    last_type, last_slot = self.history.pop()
                    if last_type == "pt" and self.points:
                        self.points.pop()
                    elif last_type == "skip" and self.skips:
                        self.skips.pop()
                    self.slot = max(0, self.slot - 1)
                    self._redraw()

            elif key in (ord('r'), ord('R')):
                self.refine = not self.refine
                self._redraw()

            elif key in (13, 10):  # Enter
                break

            elif key == 27:  # Esc
                raise KeyboardInterrupt("User cancelled.")

            # Stop when we've consumed all slots (points + skips)
            if (n_expected is not None) and (self.slot >= n_expected):
                break

        return list(self.points), list(self.skips)

    def close(self):
        try:
            cv2.destroyWindow(self.win)
        except cv2.error:
            pass


# ------------------------------ Main routine ----------------------------------

def static_manual_triangulation_test(
    calib_file: str,
    left_image_path: str,
    right_image_path: str,
    *,
    n_le: int,
    n_per_strut: int = 3,
    n_per_strut_list: Optional[List[int]] = None,
    n_can: int = 9,
    rectify_alpha: float = 0.0,
    refine_subpix: bool = False,
    display_scale: float = 0.5,
    patch_size: int = 20,
    training_out_dir: str = "training_patches",
    output_csv_basename: str = "static_manual_3d",
    yaw_adj_deg: float = 0.0,
    pitch_adj_deg: float = 0.0,
    roll_adj_deg: float = 0.0,
    # --- NEW ---
    left_frame_idx: Optional[int] = None,
    right_frame_idx: Optional[int] = None,
    left_time_sec: Optional[float] = None,
    right_time_sec: Optional[float] = None,
) -> str:

    os.makedirs("static_test_output", exist_ok=True)

    calib = _load_calibration(calib_file)
    K1, D1 = calib["camera_matrix_1"], calib["dist_coeffs_1"]
    K2, D2 = calib["camera_matrix_2"], calib["dist_coeffs_2"]
    R_cal, T = calib["R"], calib["T"]

    # Apply YPR adjustment in LEFT/world frame: R' = dR @ R
    R_use = _adjust_R_yaw_pitch_roll_leftframe(
        R_cal, yaw_deg=yaw_adj_deg, pitch_deg=pitch_adj_deg, roll_deg=roll_adj_deg
    )
    print(f"[YPR ADJ] yaw={yaw_adj_deg:+.6f}°, pitch={pitch_adj_deg:+.6f}°, roll={roll_adj_deg:+.6f}°  ->  applied to R")

    L = _load_image_or_video_frame(left_image_path, frame_idx=left_frame_idx, time_sec=left_time_sec)
    Rimg = _load_image_or_video_frame(right_image_path, frame_idx=right_frame_idx, time_sec=right_time_sec)

    if L.shape[:2] != Rimg.shape[:2]:
        raise ValueError("Left/right images must have the same size.")
    h, w = L.shape[:2]
    size = (w, h)

    # Rectify (with adjusted R)
    (_, _, P1, P2, _), maps = _stereo_rectify_maps(K1, D1, K2, D2, R_use, T, size, rectify_alpha)
    map1x, map1y, map2x, map2y = maps
    Lr = cv2.remap(L, map1x, map1y, cv2.INTER_LINEAR)
    Rr = cv2.remap(Rimg, map2x, map2y, cv2.INTER_LINEAR)

    # Click order: all LEFT, then all RIGHT (RIGHT mirrors LEFT skips but can add extra)
    left_ui = ZoomClickSession("Rectified LEFT", Lr, display_scale=display_scale, refine_subpix=refine_subpix)
    right_ui = ZoomClickSession("Rectified RIGHT", Rr, display_scale=display_scale, refine_subpix=refine_subpix)

    print(f"[LE-LEFT] Click {n_le} points (right-click to SKIP), press ENTER.")
    le_L, le_skip = left_ui.collect(n_expected=n_le, group_label="LE (LEFT)", reset=True)

    counts = n_per_strut_list if (n_per_strut_list and len(n_per_strut_list) == 8) else [n_per_strut] * 8
    struts_L: List[List[Tuple[float, float]]] = []
    struts_skip_L: List[List[int]] = []
    for si in range(8):
        c = counts[si]
        print(f"[strut{si}-LEFT] Click {c} points (right-click to SKIP), press ENTER.")
        sL, sSkip = left_ui.collect(n_expected=c, group_label=f"strut{si} (LEFT)", reset=True)
        struts_L.append(sL)
        struts_skip_L.append(sSkip)

    print(f"[CAN-LEFT] Click {n_can} points (right-click to SKIP), press ENTER.")
    can_L, can_skip_L = left_ui.collect(n_expected=n_can, group_label="CAN (LEFT)", reset=True)

    print(f"[LE-RIGHT] Click remaining points; mirrored SKIPs are auto-applied. You can still skip extras. Press ENTER.")
    le_R, le_skip_R = right_ui.collect(n_expected=n_le, group_label="LE (RIGHT)", reset=True, mirror_skips=le_skip)

    struts_R: List[List[Tuple[float, float]]] = []
    struts_skip_R: List[List[int]] = []
    for si in range(8):
        c = counts[si]
        print(f"[strut{si}-RIGHT] Click remaining points; mirrored SKIPs are auto-applied. You can still skip extras. Press ENTER.")
        sR, sSkipR = right_ui.collect(n_expected=c, group_label=f"strut{si} (RIGHT)", reset=True,
                                      mirror_skips=struts_skip_L[si])
        struts_R.append(sR)
        struts_skip_R.append(sSkipR)

    print(f"[CAN-RIGHT] Click remaining points; mirrored SKIPs are auto-applied. You can still skip extras. Press ENTER.")
    can_R, can_skip_R = right_ui.collect(n_expected=n_can, group_label="CAN (RIGHT)", reset=True, mirror_skips=can_skip_L)

    left_ui.close()
    right_ui.close()

    # ------------------ Reconcile LEFT/RIGHT with extra RIGHT-side skips ------------------
    # LEFT has its own skips; RIGHT mirrored those and may add extra skips.
    # Keep only slots present on BOTH sides to avoid mismatches.

    # LE
    le_L_filt, le_R_filt, le_keep = _reconcile_correspondences(
        le_L, le_skip, le_R, le_skip_R, n_le
    )

    # STRUTS
    struts_L_filt, struts_R_filt = [], []
    for si in range(8):
        c = counts[si]
        sL_f, sR_f, _ = _reconcile_correspondences(
            struts_L[si], struts_skip_L[si], struts_R[si], struts_skip_R[si], c
        )
        struts_L_filt.append(sL_f)
        struts_R_filt.append(sR_f)

    # CAN
    can_L_filt, can_R_filt, _ = _reconcile_correspondences(
        can_L, can_skip_L, can_R, can_skip_R, n_can
    )

    # Triangulate (uses P1,P2 from adjusted R)
    def _tri(lpts, rpts, label: str):
        if len(lpts) == 0 or len(rpts) == 0:
            print(f"[WARN] Skipping {label}: no matching points after reconciliation.")
            return np.empty((0, 3), float)
        l_lbl = [(x, y, label) for (x, y) in lpts]
        tri = triangulate_points(l_lbl, rpts, P1, P2)
        xyz = np.array([t[0] if isinstance(t, tuple) else t for t in tri], dtype=float)
        return xyz

    points_3d = {"LE": _tri(le_L_filt, le_R_filt, "LE")}
    for si in range(8):
        points_3d[f"strut{si}"] = _tri(struts_L_filt[si], struts_R_filt[si], f"strut{si}")
    points_3d["CAN"] = _tri(can_L_filt, can_R_filt, "CAN")

    # Save CSV
    rows = []
    for g, P in points_3d.items():
        for k, p in enumerate(P):
            rows.append({"group": g, "idx_in_group": int(k), "x": float(p[0]), "y": float(p[1]), "z": float(p[2])})
    df = pd.DataFrame(rows, columns=["group", "idx_in_group", "x", "y", "z"])
    csv_path = os.path.join("static_test_output", f"{output_csv_basename}.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)

    # Save training patches (filtered/kept correspondences)
    clicks = {"LE": (le_L_filt, le_R_filt)}
    for si in range(8):
        clicks[f"strut{si}"] = (struts_L_filt[si], struts_R_filt[si])
    clicks["CAN"] = (can_L_filt, can_R_filt)
    img_id = Path(output_csv_basename).name
    _ = _dump_training_from_clicks(Lr, Rr, clicks, out_dir=training_out_dir, patch=patch_size, img_id=img_id)

    print(f"\n[✓] Triangulated {sum(len(v) for v in points_3d.values())} points.")
    print(f"[→] Saved CSV: {csv_path}")
    return csv_path


# ------------------------------ Script entry ----------------------------------

if __name__ == "__main__":
    left_frame = 7811

    CSV = static_manual_triangulation_test(
        calib_file="../Calibration/stereoscopic_calibration/stereo_calibration_output/final_stereo_calibration_V3.pkl",
        left_image_path="../input/left_videos/09_10_merged.MP4",
        right_image_path="../input/right_videos/09_10_merged.MP4",
        left_frame_idx=left_frame,
        right_frame_idx=left_frame - 15,    # e.g. small offset
        n_le=23,
        n_per_strut=6,
        n_per_strut_list=[5, 5, 6, 6, 6, 6, 5, 5],
        n_can=0,
        yaw_adj_deg= 1.070,
        pitch_adj_deg= -0.272,
        roll_adj_deg= -0.432,
        refine_subpix=False,
        display_scale=0.5,
        output_csv_basename="P2_TL2",
    )
