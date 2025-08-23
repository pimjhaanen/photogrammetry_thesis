import cv2
import numpy as np
import glob
import os
import json

with open("checkerboard_config.json", "r") as f:
    cfg = json.load(f)

CHECKERBOARD = tuple(cfg["CHECKERBOARD"])
square_size = cfg["square_size_mm"]

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= square_size

objpoints = []
imgpoints_left = []
imgpoints_right = []

images_left = sorted(glob.glob(cfg["left_image_dir"] + "/*.jpg"))
images_right = sorted(glob.glob(cfg["right_image_dir"] + "/*.jpg"))

for img_left_path, img_right_path in zip(images_left, images_right):
    imgL = cv2.imread(img_left_path)
    imgR = cv2.imread(img_right_path)
    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    retL, cornersL = cv2.findChessboardCorners(grayL, CHECKERBOARD, None)
    retR, cornersR = cv2.findChessboardCorners(grayR, CHECKERBOARD, None)

    if retL and retR:
        objpoints.append(objp)
        corners2L = cv2.cornerSubPix(grayL, cornersL, (11,11), (-1,-1), criteria)
        corners2R = cv2.cornerSubPix(grayR, cornersR, (11,11), (-1,-1), criteria)
        imgpoints_left.append(corners2L)
        imgpoints_right.append(corners2R)

        cv2.drawChessboardCorners(imgL, CHECKERBOARD, corners2L, retL)
        cv2.drawChessboardCorners(imgR, CHECKERBOARD, corners2R, retR)
        cv2.imwrite("calibration/output/cornersL_" + os.path.basename(img_left_path), imgL)
        cv2.imwrite("calibration/output/cornersR_" + os.path.basename(img_right_path), imgR)

ret, mtxL, distL, mtxR, distR, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpoints_left, imgpoints_right, None, None, None, None,
    grayL.shape[::-1], criteria=criteria, flags=cv2.CALIB_FIX_INTRINSIC
)

np.savez("calibration/output/stereo_params.npz",
         mtxL=mtxL, distL=distL, mtxR=mtxR, distR=distR, R=R, T=T, E=E, F=F)
print("Calibration complete. Results saved.")
