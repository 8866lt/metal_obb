# deploy/postprocess.py
import sys
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO


class OBBDetector:
    def __init__(self, model_path, conf=0.3, iou=0.5):
        self.model = YOLO(model_path, task="obb")
        self.conf = conf
        self.iou = iou
        print(f"[OBBDetector] 加载模型: {model_path}")

    def infer(self, bgr):
        r = self.model(bgr, conf=self.conf, iou=self.iou, verbose=False)[0]
        if r.obb is None or len(r.obb) == 0:
            return []
        out = []
        for xywhr, conf in zip(r.obb.xywhr.cpu().numpy(),
                               r.obb.conf.cpu().numpy()):
            cx, cy, w, h, theta = xywhr
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            dx = np.array([-w/2,  w/2,  w/2, -w/2])
            dy = np.array([-h/2, -h/2,  h/2,  h/2])
            x = cx + dx*cos_t - dy*sin_t
            y = cy + dx*sin_t + dy*cos_t
            corners = np.stack([x, y], axis=1).astype(np.float32)
            out.append({"corners": corners, "xywhr": xywhr, "conf": float(conf)})
        return out


def refine_obb(bgr, corners_yolo, expand=15):
    H, W = bgr.shape[:2]
    pts = corners_yolo.astype(np.int32)
    x0 = max(0, pts[:, 0].min() - expand)
    y0 = max(0, pts[:, 1].min() - expand)
    x1 = min(W, pts[:, 0].max() + expand)
    y1 = min(H, pts[:, 1].max() + expand)

    roi = bgr[y0:y1, x0:x1]
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi_gray = cv2.GaussianBlur(roi_gray, (3, 3), 0)

    edges = cv2.Canny(roi_gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180,
                            threshold=30, minLineLength=20, maxLineGap=10)
    if lines is None or len(lines) < 4:
        return corners_yolo

    lines = lines.reshape(-1, 4)
    angles = np.mod(np.arctan2(lines[:, 3]-lines[:, 1],
                               lines[:, 2]-lines[:, 0]), np.pi)
    rect_angle = np.mod(np.arctan2(corners_yolo[1, 1]-corners_yolo[0, 1],
                                   corners_yolo[1, 0]-corners_yolo[0, 0]), np.pi)
    diff = np.abs(angles - rect_angle)
    diff = np.minimum(diff, np.pi - diff)
    long_lines  = lines[diff < np.pi/6]
    short_lines = lines[np.abs(diff - np.pi/2) < np.pi/6]

    if len(long_lines) < 2 or len(short_lines) < 2:
        return corners_yolo

    cx_local = corners_yolo.mean(axis=0) - np.array([x0, y0])
    long_dir  = np.array([np.cos(rect_angle), np.sin(rect_angle)])
    short_dir = np.array([-long_dir[1], long_dir[0]])

    def pick_two(ls, nd):
        mids = (ls[:, :2] + ls[:, 2:]) / 2
        proj = (mids - cx_local) @ nd
        return ls[proj.argmax()], ls[proj.argmin()]

    L1, L2 = pick_two(long_lines,  short_dir)
    S1, S2 = pick_two(short_lines, long_dir)

    def intersect(l1, l2):
        x1_, y1_, x2_, y2_ = l1
        x3_, y3_, x4_, y4_ = l2
        d = (x1_-x2_)*(y3_-y4_) - (y1_-y2_)*(x3_-x4_)
        if abs(d) < 1e-6:
            return None
        t = ((x1_-x3_)*(y3_-y4_) - (y1_-y3_)*(x3_-x4_)) / d
        return np.array([x1_ + t*(x2_-x1_), y1_ + t*(y2_-y1_)])

    raw = []
    for L in (L1, L2):
        for S in (S1, S2):
            p = intersect(L, S)
            if p is not None:
                raw.append(p)
    if len(raw) != 4:
        return corners_yolo
    raw = np.array(raw, dtype=np.float32)

    rh, rw = roi_gray.shape
    raw[:, 0] = np.clip(raw[:, 0], 1, rw - 2)
    raw[:, 1] = np.clip(raw[:, 1], 1, rh - 2)
    refined = raw.copy().reshape(-1, 1, 2)
    cv2.cornerSubPix(roi_gray, refined, (7, 7), (-1, -1),
                     (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001))
    refined = refined.reshape(-1, 2) + np.array([x0, y0])

    # 按原始顺序对齐
    used = [False] * 4
    out = np.zeros_like(corners_yolo)
    for i, p in enumerate(corners_yolo):
        d = np.linalg.norm(refined - p, axis=1)
        for j in d.argsort():
            if not used[j]:
                out[i] = refined[j]; used[j] = True; break
    return out


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  default="../runs/obb/metal_v1-2/weights/best.pt")
    parser.add_argument("--source", default="../data/dataset/images/val")
    parser.add_argument("--conf",   type=float, default=0.3)
    args = parser.parse_args()

    print(f"model : {args.model}")
    print(f"source: {args.source}")

    det = OBBDetector(args.model, conf=args.conf)

    src = Path(args.source)
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    files = [src] if src.is_file() else \
            sorted(f for f in src.iterdir() if f.suffix.lower() in exts)
    print(f"找到 {len(files)} 张图片")

    Path("output").mkdir(exist_ok=True)

    for img_path in files:
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            print(f"  ⚠️ 读取失败: {img_path}"); continue

        results = det.infer(bgr)
        if not results:
            print(f"{img_path.name}: 无检测结果"); continue

        best = max(results, key=lambda r: r["conf"])
        corners_raw = best["corners"]
        corners_ref = refine_obb(bgr, corners_raw)
        delta = np.linalg.norm(corners_ref - corners_raw, axis=1)

        print(f"\n{img_path.name}  conf={best['conf']:.3f}")
        print(f"  YOLO角点:  {np.round(corners_raw, 1).tolist()}")
        print(f"  精化角点:  {np.round(corners_ref, 1).tolist()}")
        print(f"  偏移(px):  {np.round(delta, 2).tolist()}")

        # 保存对比图
        vis = bgr.copy()
        cv2.polylines(vis, [corners_raw.astype(np.int32).reshape(-1,1,2)],
                      True, (0,200,0), 2)   # 绿：YOLO
        cv2.polylines(vis, [corners_ref.astype(np.int32).reshape(-1,1,2)],
                      True, (0,0,255), 2)   # 红：精化
        for pt in corners_ref.astype(np.int32):
            cv2.circle(vis, tuple(pt), 4, (0,0,255), -1)
        cv2.putText(vis, "Green=YOLO  Red=Refined",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
        out_path = Path("output") / img_path.name
        cv2.imwrite(str(out_path), vis)
        print(f"  保存: {out_path}")
