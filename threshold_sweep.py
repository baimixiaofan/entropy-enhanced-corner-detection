import sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import compare
from entropy_corner import EntropyCornerParams, detect_entropy_corners
from harris_baseline import detect_harris_corners

img = compare.imread_unicode(r"test_images\low_contrast_checkerboard.png")
gt = np.load(r"test_images\low_contrast_checkerboard_gt.npy")
print("\n=== low contrast checkerboard (25 GT) ===")
print(f"{'thr':<6} {'harris':>32} | {'entropy':>32}")
for thr in [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]:
    hc, R_h = detect_harris_corners(img, threshold_rel=thr)
    ec, R_e, H, R0 = detect_entropy_corners(img, EntropyCornerParams(threshold_rel=thr))
    m_h = compare.metrics(hc, gt)
    m_e = compare.metrics(ec, gt)
    f = lambda m: f"n={m['corners']:>3} tp={m['tp']:>2} fp={m['fp']:>2} fn={m['fn']:>2} f1={m['f1']:.3f}"
    print(f"{thr:<6} {f(m_h):>32} | {f(m_e):>32}")