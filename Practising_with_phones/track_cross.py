import cv2
import numpy as np
import os
from glob import glob

def track_cross(template_path, search_dir):
    template = cv2.imread(template_path, 0)
    w, h = template.shape[::-1]
    results = []

    for img_path in sorted(glob(os.path.join(search_dir, "*.jpg"))):
        img = cv2.imread(img_path, 0)
        res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        results.append((os.path.basename(img_path), max_loc, max_val))
    return results
