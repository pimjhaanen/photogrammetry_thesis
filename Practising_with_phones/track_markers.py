import cv2
import numpy as np
import os
from glob import glob
import matplotlib.pyplot as plt

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()

def detect_aruco(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    if ids is not None:
        for i in range(len(corners)):
            cv2.polylines(img, [np.int32(corners[i])], True, (0,255,0), 2)
            c = corners[i][0]
            cx = int(np.mean(c[:,0]))
            cy = int(np.mean(c[:,1]))
            cv2.putText(img, str(ids[i][0]), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
    return img

def batch_detect(folder):
    output = os.path.join(folder, "detected")
    os.makedirs(output, exist_ok=True)
    for f in sorted(glob(os.path.join(folder, "*.jpg"))):
        img = detect_aruco(f)
        cv2.imwrite(os.path.join(output, os.path.basename(f)), img)
    print("Detection complete.")

if __name__ == "__main__":
    batch_detect("calibration/images/left")
