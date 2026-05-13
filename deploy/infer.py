# deploy/infer.py
"""
独立推理验证脚本
支持 .pt / .onnx / .engine 模型
用法:
    # 单张图片
    python infer.py --model ../runs/obb/metal_v1-2/weights/best.pt --source img.jpg

    # 目录（逐张显示）
    python infer.py --model best.pt --source ../data/dataset/images/val

    # 摄像头
    python infer.py --model best.pt --source 0

    # 不弹窗，只保存结果
    python infer.py --model best.pt --source val/ --save --no-show
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


# ──────────────────────────────────────────────
# 推理封装
# ──────────────────────────────────────────────
class OBBDetector:
    def __init__(self, model_path: str, conf: float = 0.3, iou: float = 0.5):
        self.model = YOLO(model_path, task="obb")
        self.conf = conf
        self.iou = iou
        print(f"[OBBDetector] 加载模型: {model_path}")

    def infer(self, bgr: np.ndarray) -> list[dict]:
        """
        返回列表，每个元素:
          corners : np.ndarray (4,2) float32  图像像素坐标，顺时针
          xywhr   : np.ndarray (5,)  (cx,cy,w,h,θ_rad)
          conf    : float
        """
        r = self.model(bgr, conf=self.conf, iou=self.iou, verbose=False)[0]
        if r.obb is None or len(r.obb) == 0:
            return []
        out = []
        for xywhr, conf in zip(r.obb.xywhr.cpu().numpy(),
                               r.obb.conf.cpu().numpy()):
            cx, cy, w, h, theta = xywhr
            corners = self._xywhr_to_corners(cx, cy, w, h, theta)
            out.append({
                "corners": corners,
                "xywhr":   xywhr,
                "conf":    float(conf),
            })
        return out

    @staticmethod
    def _xywhr_to_corners(cx, cy, w, h, theta) -> np.ndarray:
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        dx = np.array([-w/2,  w/2,  w/2, -w/2])
        dy = np.array([-h/2, -h/2,  h/2,  h/2])
        x = cx + dx * cos_t - dy * sin_t
        y = cy + dx * sin_t + dy * cos_t
        return np.stack([x, y], axis=1).astype(np.float32)


# ──────────────────────────────────────────────
# 可视化
# ──────────────────────────────────────────────
def draw_results(bgr: np.ndarray, results: list[dict], elapsed_ms: float) -> np.ndarray:
    vis = bgr.copy()
    for r in results:
        pts = r["corners"].astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        # 角点
        for pt in r["corners"].astype(np.int32):
            cv2.circle(vis, tuple(pt), 4, (0, 0, 255), -1)
        # 置信度标签
        cx, cy = r["corners"].mean(axis=0).astype(int)
        label = f"{r['conf']:.2f}"
        cv2.putText(vis, label, (cx - 20, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    # 耗时
    cv2.putText(vis, f"{elapsed_ms:.1f}ms  {len(results)} det",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    return vis


# ──────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

def run(args):
    det = OBBDetector(args.model, conf=args.conf)
    save_dir = Path(args.save_dir)
    if args.save:
        save_dir.mkdir(parents=True, exist_ok=True)

    source = args.source

    # ── 摄像头 ──
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t0 = time.perf_counter()
            results = det.infer(frame)
            elapsed = (time.perf_counter() - t0) * 1000
            vis = draw_results(frame, results, elapsed)
            if not args.no_show:
                cv2.imshow("OBB", vis)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        cap.release()
        cv2.destroyAllWindows()
        return

    # ── 图片 / 目录 ──
    src = Path(source)
    if src.is_file():
        files = [src]
    elif src.is_dir():
        files = sorted(f for f in src.iterdir() if f.suffix.lower() in IMG_EXTS)
    else:
        print(f"❌ source 不存在: {source}"); return

    print(f"共 {len(files)} 张图片")
    for img_path in files:
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            print(f"⚠️ 读取失败: {img_path}"); continue

        t0 = time.perf_counter()
        results = det.infer(bgr)
        elapsed = (time.perf_counter() - t0) * 1000

        print(f"{img_path.name}  {elapsed:.1f}ms  {len(results)} 个目标")
        for i, r in enumerate(results):
            angle_deg = np.degrees(r["xywhr"][4])
            print(f"  [{i}] conf={r['conf']:.3f}  "
                  f"cx={r['xywhr'][0]:.1f} cy={r['xywhr'][1]:.1f}  "
                  f"w={r['xywhr'][2]:.1f} h={r['xywhr'][3]:.1f}  "
                  f"θ={angle_deg:.1f}°")

        vis = draw_results(bgr, results, elapsed)

        if args.save:
            out_path = save_dir / img_path.name
            cv2.imwrite(str(out_path), vis)
            print(f"  → 保存: {out_path}")

        if not args.no_show:
            cv2.imshow("OBB - [任意键下一张 / q退出]", vis)
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                break

    cv2.destroyAllWindows()
    print("完成")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   default="../runs/obb/metal_v1-2/weights/best.pt")
    parser.add_argument("--source",  default="../data/dataset/images/val")
    parser.add_argument("--conf",    type=float, default=0.3)
    parser.add_argument("--save",    action="store_true", help="保存结果图片")
    parser.add_argument("--save-dir",default="output", dest="save_dir")
    parser.add_argument("--no-show", action="store_true", dest="no_show")
    args = parser.parse_args()
    run(args)
