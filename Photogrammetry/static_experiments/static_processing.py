"""
Manual static stereo test with zoomable clicking + training patch export.
CSV-only output (no plotting). Each group starts fresh (no auto-complete cascade).
"""

import os
import json
import pickle
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
import pandas as pd

from Photogrammetry.stereo_photogrammetry_utils import _stereo_rectify_maps, triangulate_points
from Photogrammetry.marker_detection.marker_detection_utils import subpixel_cross_center


# ------------------------------ Helper I/O ------------------------------------

def _load_calibration(calib_pkl_path: str):
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
            meta.append({"file": path, "label": label, "group": k, "camera": "L", "cx": float(p[0]), "cy": float(p[1]), "patch": patch, "image_id": img_id})
        for p in ptsR:
            path = _save_patch(Rr, p, patch, out_dir, label, img_id, "R", extra_suffix=suffix)
            meta.append({"file": path, "label": label, "group": k, "camera": "R", "cx": float(p[0]), "cy": float(p[1]), "patch": patch, "image_id": img_id})

    meta_path = Path(out_dir, f"{img_id}_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[✓] Saved {len(meta)} patches to {out_dir}")
    return str(meta_path)


# ---------------------------- Zoomable click UI -------------------------------

class ZoomClickSession:
    """
    Zoomable/pannable click UI with correct coordinate mapping.
    - Uses cv2.getWindowImageRect() to map window <-> image pixels.
    - Stores points in ORIGINAL rectified image pixels.
    - collect(..., reset=True) clears points at the start of every group.
    Keys: left-click add, wheel zoom, right-drag pan, 'u' undo, 'r' refine, Enter accept, Esc cancel.
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

        status = f"zoom={self.zoom:.2f}  pts={len(self.points)}  refine={'ON' if self.refine else 'OFF'}"
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
            title += f"  [refine={'ON' if self.refine else 'OFF'}]"
            cv2.setWindowTitle(self.win, title)
            key = cv2.waitKey(20) & 0xFF
            if key in (ord('u'), ord('U')):
                if self.points: self.points.pop(); self._redraw()
            elif key in (ord('r'), ord('R')):
                self.refine = not self.refine; self._redraw()
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
    output_csv_basename: str = "static_manual_3d"
) -> str:

    os.makedirs("static_test_output", exist_ok=True)

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

    # Rectify
    (_, _, P1, P2, _), maps = _stereo_rectify_maps(K1, D1, K2, D2, R, T, size, rectify_alpha)
    map1x, map1y, map2x, map2y = maps
    Lr = cv2.remap(L,    map1x, map1y, cv2.INTER_LINEAR)
    Rr = cv2.remap(Rimg, map2x, map2y, cv2.INTER_LINEAR)

    # Click order: all LEFT, then all RIGHT
    left_ui  = ZoomClickSession("Rectified LEFT",  Lr, display_scale=display_scale, refine_subpix=refine_subpix)
    right_ui = ZoomClickSession("Rectified RIGHT", Rr, display_scale=display_scale, refine_subpix=refine_subpix)

    print(f"[LE-LEFT] Click {n_le} points, press ENTER.")
    le_L = left_ui.collect(n_expected=n_le, group_label="LE (LEFT)", reset=True)

    counts = n_per_strut_list if (n_per_strut_list and len(n_per_strut_list)==8) else [n_per_strut]*8
    struts_L: List[List[Tuple[float,float]]] = []
    for si in range(8):
        c = counts[si]
        print(f"[strut{si}-LEFT] Click {c} points, press ENTER.")
        sL = left_ui.collect(n_expected=c, group_label=f"strut{si} (LEFT)", reset=True)
        struts_L.append(sL)

    print(f"[CAN-LEFT] Click {n_can} points, press ENTER.")
    can_L = left_ui.collect(n_expected=n_can, group_label="CAN (LEFT)", reset=True)

    print(f"[LE-RIGHT] Click {n_le} points, press ENTER.")
    le_R = right_ui.collect(n_expected=n_le, group_label="LE (RIGHT)", reset=True)

    struts_R: List[List[Tuple[float,float]]] = []
    for si in range(8):
        c = counts[si]
        print(f"[strut{si}-RIGHT] Click {c} points, press ENTER.")
        sR = right_ui.collect(n_expected=c, group_label=f"strut{si} (RIGHT)", reset=True)
        struts_R.append(sR)

    print(f"[CAN-RIGHT] Click {n_can} points, press ENTER.")
    can_R = right_ui.collect(n_expected=n_can, group_label="CAN (RIGHT)", reset=True)

    left_ui.close(); right_ui.close()

    # Sanity
    if len(le_L) != len(le_R):
        raise RuntimeError(f"LE count mismatch: left {len(le_L)} vs right {len(le_R)}.")
    for si in range(8):
        if len(struts_L[si]) != len(struts_R[si]):
            raise RuntimeError(f"Strut {si} mismatch: left {len(struts_L[si])} vs right {len(struts_R[si])}.")
    if len(can_L) != len(can_R):
        raise RuntimeError(f"CAN mismatch: left {len(can_L)} vs right {len(can_R)}.")

    # Triangulate
    def _tri(lpts, rpts, label: str):
        l_lbl = [(x, y, label) for (x, y) in lpts]
        tri = triangulate_points(l_lbl, rpts, P1, P2)
        xyz = np.array([t[0] if isinstance(t, tuple) else t for t in tri], dtype=float)
        return xyz

    points_3d = {"LE": _tri(le_L, le_R, "LE")}
    for si in range(8):
        points_3d[f"strut{si}"] = _tri(struts_L[si], struts_R[si], f"strut{si}")
    points_3d["CAN"] = _tri(can_L, can_R, "CAN")

    # Save CSV (supports nested basename like "output/P1_S")
    rows = []
    for g, P in points_3d.items():
        for k, p in enumerate(P):
            rows.append({"group": g, "idx_in_group": int(k), "x": float(p[0]), "y": float(p[1]), "z": float(p[2])})
    df = pd.DataFrame(rows, columns=["group", "idx_in_group", "x", "y", "z"])
    csv_path = os.path.join("static_test_output", f"{output_csv_basename}.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)

    # Save training patches
    clicks = {"LE": (le_L, le_R)}
    for si in range(8):
        clicks[f"strut{si}"] = (struts_L[si], struts_R[si])
    clicks["CAN"] = (can_L, can_R)
    img_id = Path(output_csv_basename).name
    _ = _dump_training_from_clicks(Lr, Rr, clicks, out_dir=training_out_dir, patch=20, img_id=img_id)

    print(f"\n[✓] Triangulated {sum(len(v) for v in points_3d.values())} points.")
    print(f"[→] Saved CSV: {csv_path}")
    return csv_path


# ------------------------------ Script entry ----------------------------------

if __name__ == "__main__":
    CSV = static_manual_triangulation_test(
        calib_file="../Calibration/stereoscopic_calibration/stereo_calibration_output/stereo_calibration_ireland.pkl",
        left_image_path="left_input_static/L_P1_S.jpg",
        right_image_path="right_input_static/R_P1_S.jpg",
        n_le=22,
        n_per_strut=6,
        n_per_strut_list=[5,5,6,6,6,6,5,5],
        n_can=9,
        rectify_alpha=0.0,
        refine_subpix=False,
        display_scale=0.5,
        patch_size=20,
        output_csv_basename="P1_S"
    )
