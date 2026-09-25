# 熵增强角点检测 (Entropy-Enhanced Corner Detection) 设计

日期: 2026-09-24

## 目标

实现基于邻域方向熵的上下文增强角点检测算法，并与 Harris 角点检测（OpenCV 内置）进行对比，输出可视化结果和定量指标。

## 核心公式

```
R_new(p) = R0(p) * (1 + α · H(p))
```

- R0: 初始响应，可选 |Ix·Iy|（梯度积）或 Harris 响应
- H: 邻域梯度方向熵，归一化到 [0, 1]
- 角点方向分散 → 熵大 → 增强；边缘方向一致 → 熵小 → 抑制

## 项目结构

```
E:\cv\角点检测新思路\
├── entropy_corner.py      # 熵增强算法
├── harris_baseline.py     # OpenCV Harris 基线
├── compare.py             # 对比：指标 + 可视化
├── make_test_images.py    # 生成合成测试图 + 真值
├── test_images/           # 测试图（*.png + *_gt.npy 真值）
└── results/               # 对比结果图 + metrics.txt
```

## 算法细节

### entropy_corner.py

参数 (EntropyCornerParams dataclass):
- window=15 邻域尺寸（奇数）
- bins=12 梯度方向分箱数
- alpha=3.0 增强系数
- r0_mode="harris"（"grad" | "harris"）
- nms_size=3, min_distance=5, threshold_rel=0.05

流程:
1. Sobel(ksize=3) 求 Ix, Iy
2. R0: grad 模式 = |Ix·Iy|（轻量高斯平滑）；harris 模式从零算 R = AB - C² - k(A+B)² (k=0.04)
3. θ = atan2(Iy, Ix) mod π（折叠到 [0,π)，边缘方向一致 → 熵低）
4. 幅值加权直方图: 每分箱 b 用 box filter 卷积 (bin_map==b)·mag，B 次滤波
5. H = −Σ p·log p（0·log0=0），除以 log(B) 归一化到 [0,1]
6. R = R0 · (1 + α·H)
7. NMS（3×3 膨胀比较 + 连通域平顶坍缩 + 最小距离抑制）+ 相对阈值
8. cornerSubPix 亚像素细化

返回: corners, R, H, R0

### harris_baseline.py

cv2.cornerHarris(gray32, blockSize=3, ksize=3, k=0.04) + 同样的 NMS + 相对阈值。

### make_test_images.py

- synthetic_checkerboard: 7×7 方格棋盘格，真值 = 6×6=36 个内部交叉点
- synthetic_polygon: 三角形+矩形+五边形+圆形干扰项，真值 = 多边形顶点
- simulated_photo_checkerboard: 透视棋盘格 + 高斯噪声(σ=10) + 模糊，模拟真实拍摄，真值 = 透视变换后的网格交叉点

真值保存为 `<name>_gt.npy`。

### compare.py

- 默认扫描 test_images/*.png；支持命令行传图片路径（真实图放入 test_images/ 即可）
- 指标（GT 存在时）: 角点数、TP/FP/FN（匹配半径 3px）、精确率、召回率、F1、平均定位误差；贪心最近邻匹配
- 真实图无 GT: 只统计角点数 + 定性图
- 输出: results/<name>_compare.png（6 面板：原图+GT | Harris 响应 | 熵增强响应 / 熵图 H | Harris 角点 | 熵增强角点）+ 控制台指标表 + results/metrics.txt

## 公平性

两算法使用相同的 NMS 窗口、最小距离抑制、相对阈值和亚像素细化。

## 环境

Python 3.14, OpenCV 4.13, NumPy 2.4, matplotlib。

## 实现要点与调试发现

1. **NMS 平顶坍缩**: 合成图上响应平顶 (多个像素相等) 会一个角点检出多个点, 用连通域+取最强解决; 噪声图上同一角点周围有多个局部极大值簇, 再叠加最小距离抑制 (min_distance=5)。
2. **亚像素细化**: 噪声使响应峰值偏离真角点 2~3px, 用 cornerSubPix 收敛回真位置 (两算法公平使用), 否则定位误差 ~2.4px 导致大量 3px 外误检。
3. **测试图设计**: 二值对角线是阶梯像素 (每个台阶是微型角点), 多边形图需轻模糊 (σ=1.5); 透视棋盘格必须铺满画布, 否则边框 (图案/背景交界) 整条线都是假角点。
4. **R0 选择**: grad 模式 (|Ix·Iy|) 无边缘拒绝能力, 噪声/阶梯处响应高且熵也高, 被 (1+αH) 进一步放大 -> 上千假角点; harris 模式作 R0 时算法=Harris×熵加权, 是公平的消融实验。
5. **熵增强的实际增益**: 真角点方向熵高 (H~0.7-0.9, 增强 ~3.3x), 噪声斑方向熵中等 (H~0.4, 增强 ~2.2x), 拉大了角点/噪声响应对比度 -> 同阈值下误检更少。

## 实测结果 (threshold_rel=0.05)

| 测试图 | Harris F1 | 熵增强 F1 |
|---|---|---|
| synthetic_checkerboard (36 GT) | 1.000 | 1.000 |
| synthetic_polygon (12 GT) | 1.000 | 1.000 |
| simulated_photo (49 GT) | 1.000 | 1.000 |
| low_contrast (25 GT, 弱对比+强噪声) | 0.321 (131角点) | **0.431** (91角点) |

低对比场景: 召回率同为 1.000, 熵增强误检减少 ~38% (106->66); 阈值 0.1 时误检减少 3.4 倍 (F1 0.847 vs 0.617)。