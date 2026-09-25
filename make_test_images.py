"""生成合成测试图 + 真值, 输出到 test_images/."""

import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "test_images")
SIZE = 60  # 棋盘格单格边长 (px)


def save(name, img, gt):
    os.makedirs(OUT, exist_ok=True)
    ok, buf = cv2.imencode(".png", img)  # imwrite 不支持中文路径, 用字节写入
    with open(os.path.join(OUT, name + ".png"), "wb") as f:
        f.write(buf.tobytes())
    np.save(os.path.join(OUT, name + "_gt.npy"), np.asarray(gt, np.float32))


def checkerboard():
    """7x7 方格棋盘格, 真值 = 6x6=36 个内部交叉点."""
    n = 7
    img = np.full((n * SIZE, n * SIZE), 255, np.uint8)
    for r in range(n):
        for c in range(n):
            if (r + c) % 2 == 0:
                img[r * SIZE:(r + 1) * SIZE, c * SIZE:(c + 1) * SIZE] = 0
    gt = [(c * SIZE, r * SIZE) for r in range(1, n) for c in range(1, n)]
    save("synthetic_checkerboard", img, gt)


def polygons():
    """三角形 + 矩形 + 五边形 + 圆形干扰项, 真值 = 多边形顶点.

    轻模糊消除二值图对角线的阶梯状像素伪影 (每个台阶都是微型角点).
    """
    img = np.full((600, 600), 255, np.uint8)
    shapes = []
    shapes.append(np.array([[90, 90], [210, 90], [150, 230]], np.int32))      # 三角形
    shapes.append(np.array([[300, 100], [520, 100], [520, 300], [300, 300]],
                           np.int32))                                          # 矩形
    ang = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, 6)[:5]                # 五边形
    pent = np.stack([200 + 90 * np.cos(ang), 460 + 90 * np.sin(ang)], 1)
    shapes.append(pent.astype(np.int32))
    for s in shapes:
        cv2.fillPoly(img, [s], 0)
    cv2.circle(img, (480, 480), 80, 0, 2)  # 圆形轮廓: 无角点干扰项
    img = cv2.GaussianBlur(img, (0, 0), 1.5)
    gt = [tuple(p) for s in shapes for p in s]
    save("synthetic_polygon", img, gt)


def simulated_photo():
    """透视棋盘格 + 高斯噪声 + 模糊, 模拟真实拍摄, 真值已知.

    棋盘格铺满画布 (无白边), 避免边框处产生假角点.
    """
    n = 8  # 每边方格数, 内部角点 7x7=49 个
    board = np.full((n * SIZE, n * SIZE), 255, np.uint8)
    for r in range(n):
        for c in range(n):
            if (r + c) % 2 == 0:
                board[r * SIZE:(r + 1) * SIZE, c * SIZE:(c + 1) * SIZE] = 0
    h, w = board.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[-30, -20], [990, -20], [970, 940], [-20, 940]])  # 超出画布, 无边框
    M = cv2.getPerspectiveTransform(src, dst)
    img = cv2.warpPerspective(board, M, (960, 920), flags=cv2.INTER_LINEAR)
    gt = []
    for r in range(1, n):
        for c in range(1, n):
            pt = M @ np.array([c * SIZE, r * SIZE, 1.0])
            gt.append((pt[0] / pt[2], pt[1] / pt[2]))
    rng = np.random.default_rng(0)
    img = np.clip(img.astype(np.float64) + rng.normal(0, 10, img.shape), 0, 255)
    img = cv2.GaussianBlur(img.astype(np.uint8), (3, 3), 0.8)
    save("simulated_photo_checkerboard", img, gt)


def low_contrast_photo():
    """弱对比棋盘格 + 强噪声: 角点响应接近噪声水平.

    考验熵增强是否能把真角点从噪声中抬升 (真角点方向熵高 -> 增强大).
    """
    n = 6  # 每边方格数, 内部角点 5x5=25 个
    board = np.full((n * SIZE, n * SIZE), 255, np.uint8)
    for r in range(n):
        for c in range(n):
            if (r + c) % 2 == 0:
                board[r * SIZE:(r + 1) * SIZE, c * SIZE:(c + 1) * SIZE] = 170
    h, w = board.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[-30, -20], [990, -20], [970, 940], [-20, 940]])
    M = cv2.getPerspectiveTransform(src, dst)
    img = cv2.warpPerspective(board, M, (960, 920), flags=cv2.INTER_LINEAR)
    gt = []
    for r in range(1, n):
        for c in range(1, n):
            pt = M @ np.array([c * SIZE, r * SIZE, 1.0])
            gt.append((pt[0] / pt[2], pt[1] / pt[2]))
    rng = np.random.default_rng(1)
    img = np.clip(img.astype(np.float64) + rng.normal(0, 12, img.shape), 0, 255)
    img = cv2.GaussianBlur(img.astype(np.uint8), (3, 3), 0.8)
    save("low_contrast_checkerboard", img, gt)


if __name__ == "__main__":
    checkerboard()
    polygons()
    simulated_photo()
    low_contrast_photo()
    print("saved to", OUT)