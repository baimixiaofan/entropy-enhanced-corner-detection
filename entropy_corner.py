"""熵增强角点检测 (Entropy-Enhanced Corner Detection).

核心思想:
    R_new(p) = R0(p) * (1 + alpha * H(p))

- R0: 初始角点响应 (梯度积 |Ix*Iy| 或 Harris 响应)
- H : 邻域梯度方向熵, 归一化到 [0,1]
      角点处梯度方向分散 -> 熵大 -> 增强;
      边缘处梯度方向一致 -> 熵小 -> 抑制.
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class EntropyCornerParams:
    window: int = 15            # 邻域尺寸 (奇数)
    bins: int = 12              # 梯度方向分箱数
    alpha: float = 3.0          # 上下文增强系数
    r0_mode: str = "harris"     # 初始响应模式: "harris" | "grad"
    nms_size: int = 3           # 非极大值抑制窗口
    min_distance: int = 5       # 角点最小间距 (抑制同角点的重复检测)
    threshold_rel: float = 0.05  # 相对阈值 (响应最大值的比例)
    harris_k: float = 0.04      # 仅 r0_mode="harris" 时使用


def _initial_response(Ix, Iy, params):
    """计算初始角点响应 R0."""
    if params.r0_mode == "grad":
        # 梯度积: 角点处 |Ix|,|Iy| 同时大; 边缘处至少一个接近 0
        # 轻量平滑抑制噪声斑点
        R = cv2.GaussianBlur(np.abs(Ix * Iy), (3, 3), 0)
        return R
    if params.r0_mode == "harris":
        k = params.harris_k
        w = cv2.getGaussianKernel(3, -1)
        w = w @ w.T  # 3x3 高斯窗
        A = cv2.filter2D(Ix * Ix, -1, w)
        B = cv2.filter2D(Iy * Iy, -1, w)
        C = cv2.filter2D(Ix * Iy, -1, w)
        R = A * B - C * C - k * (A + B) ** 2
        return np.maximum(R, 0.0)
    raise ValueError(f"unknown r0_mode: {params.r0_mode}")


def direction_entropy(Ix, Iy, bins=12, window=15):
    """幅值加权的邻域梯度方向熵图, 归一化到 [0, 1].

    角度折叠到 [0, pi): 直线上边缘两侧梯度方向相差 pi,
    折叠后同向 -> 方向直方图集中 -> 熵低.
    """
    mag = np.hypot(Ix, Iy)
    theta = np.arctan2(Iy, Ix) % np.pi
    bin_map = np.clip((theta / (np.pi / bins)).astype(np.int32), 0, bins - 1)

    kernel = np.ones((window, window), np.float64)
    total = cv2.filter2D(mag, -1, kernel, borderType=cv2.BORDER_REFLECT)

    H = np.zeros_like(mag)
    for b in range(bins):
        mask = (bin_map == b).astype(np.float64) * mag
        count = cv2.filter2D(mask, -1, kernel, borderType=cv2.BORDER_REFLECT)
        p = count / np.maximum(total, 1e-12)
        with np.errstate(divide="ignore", invalid="ignore"):
            H -= np.where(p > 0, p * np.log(p), 0.0)
    return np.clip(H / np.log(bins), 0.0, 1.0)


def nms(response, size=3, threshold=0.0, min_distance=5):
    """非极大值抑制 + 最小距离去重, 返回角点坐标 (N, 2) 的 (x, y).

    1. 平顶 (多个像素响应相等) 只保留响应最大且最靠前的像素;
    2. 对保留下来的候选按响应降序贪心选取, 与已选角点距离
       < min_distance 的丢弃 (抑制同一角点周围的噪声局部极大值).
    """
    if size % 2 == 0:
        size += 1
    kernel = np.ones((size, size), np.uint8)
    local_max = cv2.dilate(response, kernel)
    mask = (response >= local_max - 1e-12) & (response >= threshold)
    num, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    cands = []
    for i in range(1, num):
        ys, xs = np.nonzero(labels == i)
        k = int(np.argmax(response[ys, xs]))
        cands.append((xs[k], ys[k], response[ys[k], xs[k]]))
    if not cands:
        return np.empty((0, 2), np.float32)
    cands.sort(key=lambda c: -c[2])
    picks = []
    for x, y, _ in cands:
        if picks and np.min(np.hypot(np.asarray(picks)[:, 0] - x,
                                     np.asarray(picks)[:, 1] - y)) < min_distance:
            continue
        picks.append((x, y))
    return np.asarray(picks, np.float32).reshape(-1, 2)


def refine_corners(gray, corners, win=3):
    """亚像素细化角点坐标 (cv2.cornerSubPix).

    噪声会使响应峰值偏离真角点 1~3px, 细化后收敛到角点交叉中心.
    """
    if len(corners) == 0:
        return corners
    c = corners.astype(np.float32).reshape(-1, 1, 2)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    refined = cv2.cornerSubPix(gray, c, (win, win), (-1, -1), criteria)
    return refined.reshape(-1, 2)


def detect_entropy_corners(gray, params=None):
    """熵增强角点检测主函数.

    Args:
        gray: 单通道灰度图 (uint8).
        params: EntropyCornerParams, None 时用默认参数.

    Returns:
        corners: (N, 2) 角点坐标 (x, y)
        R:       上下文增强后的响应图
        H:       方向熵图 (归一化 [0,1])
        R0:      初始响应图
    """
    if params is None:
        params = EntropyCornerParams()
    gray_u8 = gray.astype(np.uint8)
    gray = gray_u8.astype(np.float64)
    Ix = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    R0 = _initial_response(Ix, Iy, params)
    H = direction_entropy(Ix, Iy, params.bins, params.window)
    R = R0 * (1.0 + params.alpha * H)

    thr = params.threshold_rel * float(R.max())
    corners = nms(R, params.nms_size, thr, params.min_distance)
    corners = refine_corners(gray_u8, corners)
    return corners, R, H, R0