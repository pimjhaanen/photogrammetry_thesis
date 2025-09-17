"""This code can be used to perform a single-camera calibration using an asymmetric circle grid.
It detects circle centers in calibration images, estimates intrinsic parameters, and optionally
visualizes detections. Useful for comparing calibration quality across different boards or lenses."""

import os, glob, pickle
import cv2
import numpy as np

def calibrate_asymmetric_circles(
    image_glob: str = 'Calibration_Videos_camera_2/circles_linear/*.jpg',
    pattern_size=(4, 11),
    circle_diameter_m: float = 2.5/100.0,
    visualize: bool = False,
    output_basename: str = 'calibration_circles_linear_camera_2'
):
    """RELEVANT FUNCTION INPUTS:
    - image_glob: folder containing calibration frames (e.g. frame_0001.jpg, …)
    - pattern_size: size of the asymmetric circle grid (cols, rows)
    - circle_diameter_m: physical diameter of circles in meters
    - visualize: if True, shows detected circle centers on the images
    - output_basename: filename (without extension) for the saved calibration results"""


    out_dir = 'single_calibration_output'
    os.makedirs(out_dir, exist_ok=True)

    # Prepare object points for asymmetric grid:
    # Even rows are offset by half a circle pitch in x.
    objp = np.zeros((np.prod(pattern_size), 3), np.float32)
    objp[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
    objp[:, 0] *= 2 * circle_diameter_m
    objp[:, 1] *= circle_diameter_m
    # Offset every other row by +circle_diameter
    for i in range(pattern_size[1]):
        if i % 2 == 1:
            row_slice = slice(i*pattern_size[0], (i+1)*pattern_size[0])
            objp[row_slice, 0] += circle_diameter_m

    objpoints, imgpoints = [], []
    images = sorted(glob.glob(image_glob))
    if not images:
        print(f"[circles] No images matched: {image_glob}")
        return None

    gray_shape = None

    for fname in images:
        img = cv2.imread(fname)
        if img is None:
            print(f"[circles] Could not read: {fname}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_shape = gray.shape[::-1]

        ret, centers = cv2.findCirclesGrid(
            gray, pattern_size, flags=cv2.CALIB_CB_ASYMMETRIC_GRID
        )

        if ret:
            objpoints.append(objp.copy())
            imgpoints.append(centers)
            msg = f"{fname}: Detected {len(centers)} circles"
            if visualize:
                vis = img.copy()
                cv2.drawChessboardCorners(vis, pattern_size, centers, True)
                cv2.imshow("Circles detection", cv2.resize(vis, None, fx=0.35, fy=0.35))
                cv2.waitKey(1)
        else:
            msg = f"{fname}: No circles detected"
        print(msg)

    if visualize:
        print("[circles] Close preview window to continue...")
        while True:
            if cv2.getWindowProperty("Circles detection", cv2.WND_PROP_VISIBLE) < 1:
                break
            if cv2.waitKey(100) == 27:
                break
        cv2.destroyAllWindows()

    if not objpoints:
        print("[circles] No valid detections, aborting.")
        return None

    # Calibrate
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray_shape, None, None
    )

    # Reprojection error
    mean_error = 0.0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        err = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += err
    mean_error /= len(objpoints)
    print("[circles] Reprojection error:", mean_error)

    # Save
    out_pkl = os.path.join(out_dir, f"{output_basename}.pkl")
    with open(out_pkl, "wb") as f:
        pickle.dump({"camera_matrix": mtx, "dist_coeffs": dist, "reprojection_error": mean_error}, f)
    print(f"[circles] Saved: {out_pkl}")

    return {"camera_matrix": mtx, "dist_coeffs": dist, "reprojection_error": mean_error}

if __name__ == "__main__":
    calibrate_asymmetric_circles(visualize=True)
