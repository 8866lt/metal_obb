# scripts/train.py
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).parent.parent   # scripts/ 的上一级 = 项目根目录

if __name__ == "__main__":
    model = YOLO("yolo26s-obb.pt")
    model.train(
        data=str(ROOT / "data/dataset/data.yaml"),
        epochs=200,
        imgsz=960,
        batch=16,
        workers=8,
        device=0,

        optimizer="auto",
        lr0=0.005,
        cos_lr=True,
        warmup_epochs=5,
        patience=50,

        # ===== 金属件场景关键调参 =====
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.2,
        degrees=180,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        flipud=0.5,
        mosaic=0.8,
        mixup=0.0,
        close_mosaic=20,

        # ===== 损失权重 =====
        box=7.5,
        cls=0.5,
        dfl=1.5,

        val=True,
        plots=True,
        save_period=20,
        project=str(ROOT / "runs/obb"),
        name="metal_v1",
        exist_ok=True,
    )
