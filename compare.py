"""对比: 熵增强角点检测 vs Harris.

用法:
    python compare.py                 # 默认跑 test_images/ 下所有 png
    python compare.py 图片路径 [...]  # 指定图片 (真实图无真值, 只统计角点数)

输出:
    results/<name>_compare.png  6 面板对比图
    results/metrics.txt         指标汇总
"""

import glob
import os
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from entropy_corner import EntropyCornerParams, detect_entropy_corners
from harris_baseline import detect_harris_corners

HERE = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.join(HERE, "test_images")
RESULT_DIR = os.path.join(HERE, "results")
MATCH_RADIUS = 3.0  # GT 匹配半径 (px)


def match_corners(det, gt, radius=MATCH_RADIUS):
    """贪心最近邻匹配, 返回 (tp, fp, fn, 定位误差列表)."""
    det = np.asarray(det, np.float64)
    gt = np.asarray(gt, np.float64)
    if det.size == 0 or gt.size == 0:
        return 0, len(det), len(gt), []
    dists = np.linalg.norm(det[:, None, :] - gt[None, :, :], axis=2)
    matched_gt, dist_list = set(), []
    for i in range(len(det)):
        j = int(np.argmin(dists[i]))
        if dists[i, j] <= radius and j not in matched_gt:
            matched_gt.add(j)
            dist_list.append(dists[i, j])
    tp = len(dist_list)
    fp = len(det) - tp
    fn = len(gt) - len(matched_gt)
    loc_err = float(np.mean(dist_list)) if dist_list else float("nan")
    return tp, fp, fn, loc_err


def metrics(det, gt):
    if gt is None:
        return {"corners": len(det)}
    tp, fp, fn, loc_err = match_corners(det, gt)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"corners": len(det), "tp": tp, "fp": fp, "fn": fn,
            "precision": prec, "recall": rec, "f1": f1, "loc_err": loc_err}


def draw_corners(img, corners, color=(0, 0, 255)):
    out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for x, y in corners:
        cv2.circle(out, (int(round(x)), int(round(y))), 4, color, 2)
    return out


def make_figure(img, gt, hc, ec, R_h, R_e, H, name, m_h, m_e):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    titles = ["Original + GT", "Harris response", "Entropy-enhanced response",
              "Direction entropy H", f"Harris corners ({m_h['corners']})",
              f"Entropy corners ({m_e['corners']})"]

    axes[0, 0].imshow(img, cmap="gray")
    if gt is not None:
        gx, gy = gt[:, 0], gt[:, 1]
        axes[0, 0].plot(gx, gy, "g+", ms=8, mew=2, label="GT")
        axes[0, 0].legend(loc="upper right", fontsize=8)

    axes[0, 1].imshow(R_h, cmap="hot")
    axes[0, 2].imshow(R_e, cmap="hot")
    axes[1, 0].imshow(H, cmap="viridis", vmin=0, vmax=1)
    axes[1, 1].imshow(draw_corners(img, hc)[:, :, ::-1])
    axes[1, 2].imshow(draw_corners(img, ec)[:, :, ::-1])

    for ax, t in zip(axes.flat, titles):
        ax.set_title(t, fontsize=11)
        ax.axis("off")
    fig.suptitle(f"{name}  |  GT: {len(gt) if gt is not None else 'N/A'}", fontsize=13)
    fig.tight_layout()
    out = os.path.join(RESULT_DIR, name + "_compare.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def fmt_row(m):
    if "tp" not in m:
        return f"{m['corners']:>7d}"
    return (f"{m['corners']:>7d} {m['tp']:>5d} {m['fp']:>5d} {m['fn']:>5d} "
            f"{m['precision']:>9.3f} {m['recall']:>7.3f} {m['f1']:>6.3f} "
            f"{m['loc_err']:>8.3f}")


def imread_unicode(path):
    data = np.fromfile(path, dtype=np.uint8)  # imread 不支持中文路径
    return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)


def run_one(path, r0_mode="harris"):
    img = imread_unicode(path)
    if img is None:
        print(f"skip (cannot read): {path}")
        return None
    gt_path = os.path.splitext(path)[0] + "_gt.npy"
    gt = np.load(gt_path) if os.path.exists(gt_path) else None
    name = os.path.splitext(os.path.basename(path))[0]

    params = EntropyCornerParams(r0_mode=r0_mode)
    ec, R_e, H, _ = detect_entropy_corners(img, params)
    hc, R_h = detect_harris_corners(img)

    m_h = metrics(hc, gt)
    m_e = metrics(ec, gt)
    fig_path = make_figure(img, gt, hc, ec, R_h, R_e, H, name, m_h, m_e)
    return name, len(gt) if gt is not None else None, m_h, m_e, fig_path


def main(paths, r0_mode="harris"):
    os.makedirs(RESULT_DIR, exist_ok=True)
    if not paths:
        paths = sorted(glob.glob(os.path.join(TEST_DIR, "*.png")))
    if not paths:
        print("no images found; run make_test_images.py first")
        return

    print(f"entropy r0_mode = {r0_mode}")
    results = [r for p in paths if (r := run_one(p, r0_mode)) is not None]
    header = (f"{'image':<30} {'GT':>4} {'harris':>7} {'entropy':>7}  figure\n"
              f"{'-' * 78}")
    print(header)
    lines = [header]
    for name, n_gt, m_h, m_e, fig_path in results:
        line = (f"{name:<30} {str(n_gt) if n_gt is not None else 'N/A':>4} "
                f"{fmt_row(m_h):>30} {fmt_row(m_e):>30}  {fig_path}")
        print(line)
        lines.append(line)

    print("\n--- detail (tp/fp/fn/precision/recall/f1/loc_err) ---")
    print(f"{'image':<30} {'method':<8} {'corner':>7} {'tp':>5} {'fp':>5} "
          f"{'fn':>5} {'precision':>9} {'recall':>7} {'f1':>6} {'loc_err':>8}")
    detail = [f"{'image':<30} {'method':<8} {'corner':>7} {'tp':>5} {'fp':>5} "
              f"{'fn':>5} {'precision':>9} {'recall':>7} {'f1':>6} {'loc_err':>8}"]
    for name, n_gt, m_h, m_e, _ in results:
        for label, m in (("harris", m_h), ("entropy", m_e)):
            if "tp" not in m:
                continue
            line = (f"{name:<30} {label:<8} {m['corners']:>7d} {m['tp']:>5d} "
                    f"{m['fp']:>5d} {m['fn']:>5d} {m['precision']:>9.3f} "
                    f"{m['recall']:>7.3f} {m['f1']:>6.3f} {m['loc_err']:>8.3f}")
            print(line)
            detail.append(line)
    with open(os.path.join(RESULT_DIR, "metrics.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines + [""] + detail))


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    r0 = "harris"
    if "--r0" in sys.argv:
        r0 = sys.argv[sys.argv.index("--r0") + 1]
    main(args, r0_mode=r0)