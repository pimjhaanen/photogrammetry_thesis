"""This code can be used to perform a single-camera calibration using a ChArUco board.
It combines ArUco marker detection with chessboard corner refinement for robust calibration.
You can adapt board dimensions, square size, and marker size to your printed board."""

import os, glob, pickle
import cv2

def calibrate_charuco(
    image_glob: str = 'Calibration_Videos_camera_2/charuco_wide/*.jpg',
    squaresX: int = 5,
    squaresY: int = 7,
    squareLength_m: float = 5.55/100.0,
    markerLength_m: float = 2.8/100.0,
    dictionary: int = cv2.aruco.DICT_5X5_100,
    visualize: bool = False,
    output_basename: str = 'calibration_charuco_wide_camera_2'
):
    """RELEVANT FUNCTION INPUTS:
    - image_glob: folder containing calibration frames (e.g. frame_0001.jpg, …)
    - squaresX, squaresY: number of squares in the ChArUco board (cols, rows)
    - squareLength_m: physical side length of each square in meters
    - markerLength_m: physical side length of each ArUco marker in meters
    - dictionary: predefined ArUco dictionary used for markers
    - visualize: if True, shows detected markers/corners on the images
    - output_basename: filename (without extension) for the saved calibration results"""


    out_dir = 'single_calibration_output'
    os.makedirs(out_dir, exist_ok=True)

    aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary)
    board = cv2.aruco.CharucoBoard_create(
        squaresX=squaresX, squaresY=squaresY,
        squareLength=squareLength_m, markerLength=markerLength_m,
        dictionary=aruco_dict
    )
    detector_params = cv2.aruco.DetectorParameters_create()

    images = sorted(glob.glob(image_glob))
    if not images:
        print(f"[charuco] No images matched: {image_glob}")
        return None

    all_corners, all_ids = [], []
    img_size = None

    for fname in images:
        img = cv2.imread(fname)
        if img is None:
            print(f"[charuco] Could not read: {fname}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_size = gray.shape[::-1]

        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=detector_params)

        num_aruco = 0 if ids is None else len(ids)
        num_charuco = 0

        if ids is not None and len(ids) > 0:
            ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                corners, ids, gray, board
            )
            if ret and charuco_ids is not None and len(charuco_ids) > 4:
                num_charuco = len(charuco_ids)
                all_corners.append(charuco_corners)
                all_ids.append(charuco_ids)

            if visualize:
                vis = img.copy()
                cv2.aruco.drawDetectedMarkers(vis, corners, ids)
                if num_charuco > 0:
                    cv2.aruco.drawDetectedCornersCharuco(vis, charuco_corners, charuco_ids)
                cv2.imshow("ChArUco detection", cv2.resize(vis, None, fx=0.35, fy=0.35))
                cv2.waitKey(1)
        else:
            if visualize:
                cv2.imshow("ChArUco detection", cv2.resize(img, None, fx=0.35, fy=0.35))
                cv2.waitKey(1)

        print(f"{fname}: {num_aruco} ArUco, {num_charuco} ChArUco")

    if visualize:
        print("[charuco] Close preview window to continue...")
        while True:
            if cv2.getWindowProperty("ChArUco detection", cv2.WND_PROP_VISIBLE) < 1:
                break
            if cv2.waitKey(100) == 27:
                break
        cv2.destroyAllWindows()

    if not all_corners or not all_ids:
        print("[charuco] No valid ChArUco corners found, aborting.")
        return None

    # Calibrate
    ret, mtx, dist, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
        charucoCorners=all_corners,
        charucoIds=all_ids,
        board=board,
        imageSize=img_size,
        cameraMatrix=None,
        distCoeffs=None
    )

    # Reprojection error over valid sets
    mean_error, valid_sets = 0.0, 0
    for i in range(len(all_corners)):
        c_corners = all_corners[i]
        c_ids = all_ids[i]
        if c_corners is None or c_ids is None or len(c_ids) < 1:
            continue
        obj_points = board.chessboardCorners[c_ids.flatten()].reshape(-1, 1, 3)  # (N,1,3)
        img_points = c_corners.reshape(-1, 1, 2)                                 # (N,1,2)
        imgpoints2, _ = cv2.projectPoints(obj_points, rvecs[i], tvecs[i], mtx, dist)
        err = cv2.norm(img_points, imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += err
        valid_sets += 1

    mean_error = (mean_error / valid_sets) if valid_sets > 0 else float('nan')
    print("[charuco] Reprojection error:", mean_error)

    # Save
    out_pkl = os.path.join(out_dir, f"{output_basename}.pkl")
    with open(out_pkl, "wb") as f:
        pickle.dump({"camera_matrix": mtx, "dist_coeffs": dist, "reprojection_error": mean_error}, f)
    print(f"[charuco] Saved: {out_pkl}")

    return {"camera_matrix": mtx, "dist_coeffs": dist, "reprojection_error": mean_error}

if __name__ == "__main__":
    calibrate_charuco(visualize=True)
