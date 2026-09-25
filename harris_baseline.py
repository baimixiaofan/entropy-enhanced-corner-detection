"""Harris 角点检测基线 (OpenCV 内置实现)."""

import cv2
import numpy as np

from entropy_corner import nms, refine_corners


def detect_harris_corners(gray, block_size=3, ksize=3, k=0.04,
                          nms_size=3, threshold_rel=0.05, min_distance=5):
    """Harris 角点检测.

    Args:
        gray: 单通道灰度图 (uint8).

    Returns:
        corners: (N, 2) 角点坐标 (x, y)
        R:       Harris 响应图
    """
    gray32 = np.float32(gray)
    R = cv2.cornerHarris(gray32, block_size, ksize, k)
    R = np.maximum(R, 0.0)
    thr = threshold_rel * float(R.max())
    corners = nms(R, nms_size, thr, min_distance)
    corners = refine_corners(gray, corners)
    return corners, R