# scripts/transform.py
import json
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
import yaml

IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}


def convert_single_json(json_path, img_dir, label_dir, name2id):
    """转换单个 JSON 为 YOLO OBB txt（9 列：class + 8 个归一化坐标）"""
    with open(json_path, encoding='utf-8') as f:
        data = json.load(f)

    img_name = data.get("imagePath", json_path.stem + ".jpg")
    img_path = img_dir / img_name
    if not img_path.exists():
        for ext in IMG_EXTENSIONS:
            alt = img_dir / f"{img_path.stem}{ext}"
            if alt.exists():
                img_path = alt
                break
    if not img_path.exists():
        print(f"⚠️ 图片不存在: {img_path}")
        return None

    img_w = data["imageWidth"]
    img_h = data["imageHeight"]
    lines = []

    for shape in data.get("shapes", []):
        label = shape["label"]
        if label not in name2id:
            print(f"⚠️ 未知类别: {label}")
            continue
        class_id = name2id[label]
        points = shape["points"]

        if shape["shape_type"] == "rotation":
            # 旋转框：直接取 4 角点归一化（保留真实角度）→ 9 列 OBB
            if len(points) != 4:
                print(f"⚠️ rotation 点数异常: {len(points)}")
                continue
            coords = []
            for x, y in points:
                coords += [x / img_w, y / img_h]
            lines.append(f"{class_id} " + " ".join(f"{v:.6f}" for v in coords))

        elif shape["shape_type"] == "rectangle":
            # 普通矩形 → 转成水平 4 角点（θ=0 的 OBB）→ 9 列
            if len(points) == 2:
                x1, y1 = points[0]; x2, y2 = points[1]
            elif len(points) == 4:
                xs = [p[0] for p in points]; ys = [p[1] for p in points]
                x1, x2 = min(xs), max(xs)
                y1, y2 = min(ys), max(ys)
            else:
                print(f"⚠️ 坐标点数量异常: {len(points)}")
                continue
            if abs(x2 - x1) == 0 or abs(y2 - y1) == 0:
                print(f"⚠️ 跳过无效标注 (宽度或高度为0): {label}")
                continue
            # 左上 → 右上 → 右下 → 左下
            corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            coords = []
            for x, y in corners:
                coords += [x / img_w, y / img_h]
            lines.append(f"{class_id} " + " ".join(f"{v:.6f}" for v in coords))

    txt_path = label_dir / f"{img_path.stem}.txt"
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path.write_text("\n".join(lines))
    return img_path, txt_path


def main(json_dir, img_dir, output_dir, class_file, split=0.2):
    print("=" * 60)
    print("开始转换 X-AnyLabeling 标注（OBB 模式，9 列输出）")
    print("=" * 60)

    with open(class_file, encoding='utf-8') as f:
        names = [line.strip() for line in f if line.strip()]
    id2name = {i: n for i, n in enumerate(names)}
    name2id = {n: i for i, n in enumerate(names)}
    print(f"✅ 加载类别: {name2id}")

    json_path = Path(json_dir)
    img_path  = Path(img_dir)
    print(f"\nJSON目录: {json_path} | 存在: {json_path.exists()}")
    print(f"图片目录: {img_path} | 存在: {img_path.exists()}")

    json_files = list(json_path.glob("*.json"))
    print(f"\n找到JSON文件数: {len(json_files)}")
    if not json_files:
        print("❌ 未找到 JSON 文件"); return

    img_files = [f for f in img_path.glob("*.*") if f.suffix.lower() in IMG_EXTENSIONS]
    print(f"找到图片文件数: {len(img_files)} | 格式: {set(f.suffix for f in img_files)}")
    if not img_files:
        print("❌ 未找到图片文件"); return

    output = Path(output_dir)
    for d in ["images/train", "images/val", "labels/train", "labels/val"]:
        (output / d).mkdir(parents=True, exist_ok=True)
    print(f"\n✅ 输出目录: {output}")

    temp_label_dir = output / "labels_temp"
    temp_label_dir.mkdir(exist_ok=True)
    img_paths = []
    for jf in json_files:
        result = convert_single_json(jf, img_path, temp_label_dir, name2id)
        if result:
            img_paths.append(result[0])

    print(f"\n转换成功总数: {len(img_paths)}")
    if not img_paths:
        print("❌ 未成功转换任何数据"); return

    temp_img_dir = output / "images_temp"
    temp_img_dir.mkdir(exist_ok=True)
    for img in img_paths:
        shutil.copy(img, temp_img_dir)
    print(f"✅ 图片已拷贝到临时目录: {len(list(temp_img_dir.glob('*')))} 张")

    all_imgs = [f for f in temp_img_dir.glob("*.*") if f.suffix.lower() in IMG_EXTENSIONS]
    if len(all_imgs) < 2:
        print("❌ 至少需要 2 张图片"); return

    train_imgs, val_imgs = train_test_split(all_imgs, test_size=split, random_state=42)
    print(f"划分结果: 训练集 {len(train_imgs)} 张, 验证集 {len(val_imgs)} 张")

    for img in train_imgs:
        shutil.move(str(img), output / "images/train")
        lbl = temp_label_dir / f"{img.stem}.txt"
        if lbl.exists():
            shutil.move(str(lbl), output / "labels/train")

    for img in val_imgs:
        shutil.move(str(img), output / "images/val")
        lbl = temp_label_dir / f"{img.stem}.txt"
        if lbl.exists():
            shutil.move(str(lbl), output / "labels/val")

    temp_img_dir.rmdir()
    remaining = list(temp_label_dir.glob("*"))
    if remaining:
        print(f"⚠️ 临时标签目录还有 {len(remaining)} 个文件未处理")
    else:
        temp_label_dir.rmdir()

    yaml_content = {
        "path":  str(output.absolute()),
        "train": "images/train",
        "val":   "images/val",
        "nc":    len(id2name),
        "names": [id2name[i] for i in range(len(id2name))]
    }
    (output / "data.yaml").write_text(
        yaml.dump(yaml_content, sort_keys=False, allow_unicode=True),
        encoding='utf-8'
    )

    print("\n" + "=" * 60)
    print("✅ 转换完成！")
    print(f"  训练集: {len(train_imgs)} 张")
    print(f"  验证集: {len(val_imgs)} 张")
    print(f"  配置文件: {output / 'data.yaml'}")
    print("=" * 60)


if __name__ == "__main__":
    ROOT = Path(__file__).parent.parent
    main(
        json_dir=str(ROOT / "data/raw"),
        img_dir=str(ROOT / "data/raw"),
        output_dir=str(ROOT / "data/dataset"),
        class_file=str(ROOT / "data/class.txt"),
        split=0.2,
    )
