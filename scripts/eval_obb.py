# scripts/eval_obb.py
import numpy as np, cv2
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt

def parse_obb_label(path, w, h):
    """8 点标签 → (cx,cy,w,h,θdeg)，强制长边为 w"""
    objs = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            cls = int(parts[0])
            pts = np.array(parts[1:], dtype=np.float32).reshape(4, 2)
            pts[:, 0] *= w; pts[:, 1] *= h
            (cx, cy), (rw, rh), ang = cv2.minAreaRect(pts.astype(np.float32))
            if rh > rw:
                rw, rh = rh, rw; ang += 90
            objs.append((cls, cx, cy, rw, rh, ang))
    return objs

def angle_diff(a, b):
    d = abs(a - b) % 180
    return min(d, 180 - d)

model = YOLO("runs/obb/metal_v1/weights/best.pt")
img_dir = Path("data/dataset/images/val")
lbl_dir = Path("data/dataset/labels/val")

errors = []
for img_path in img_dir.glob("*.jpg"):
    img = cv2.imread(str(img_path))
    H, W = img.shape[:2]
    gts = parse_obb_label(lbl_dir / (img_path.stem + ".txt"), W, H)
    preds = model(img, verbose=False)[0].obb
    if preds is None or len(preds) == 0:
        continue
    pred_boxes = preds.xywhr.cpu().numpy()
    for gt in gts:
        d = np.linalg.norm(pred_boxes[:, :2] - np.array([gt[1], gt[2]]), axis=1)
        i = d.argmin()
        if d[i] > 50:
            continue
        errors.append(angle_diff(np.degrees(pred_boxes[i, 4]), gt[5]))

errors = np.array(errors)
print(f"N={len(errors)}, mean={errors.mean():.2f}°, "
      f"median={np.median(errors):.2f}°, P95={np.percentile(errors,95):.2f}°")
plt.hist(errors, bins=50); plt.xlabel("angle error (deg)")
plt.savefig("runs/obb/metal_v1/angle_err.png")
