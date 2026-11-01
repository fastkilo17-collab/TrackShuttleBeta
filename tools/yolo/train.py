# -*- coding: utf-8 -*-
"""
ขั้นที่ 2: เทรน YOLO ขนาดเล็กตรวจลูกขนไก่ แล้วส่งออกเป็น ONNX / TFLite
ใช้ชุดข้อมูลจาก autolabel.py   (ต้อง pip install ultralytics)
"""
import shutil
from pathlib import Path

import yaml


def write_yaml(dataset_dir):
    d = Path(dataset_dir).resolve()
    p = d / "data.yaml"
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump({"path": str(d), "train": "images/train", "val": "images/val",
                        "names": {0: "shuttlecock"}}, f, allow_unicode=True)
    return p


def count_images(dataset_dir):
    d = Path(dataset_dir)
    out = {}
    for sp in ("train", "val"):
        imgs = list((d / "images" / sp).glob("*.jpg"))
        pos = sum(1 for p in imgs if (d / "labels" / sp / f"{p.stem}.txt").stat().st_size > 0)
        out[sp] = (len(imgs), pos)
    return out


def train(dataset_dir, project, name="shuttle", base="yolov8n.pt", epochs=80,
          imgsz=640, batch=16, patience=20, device=None):
    """คืน path ของ best.pt"""
    from ultralytics import YOLO

    c = count_images(dataset_dir)
    print(f"ชุดข้อมูล: train {c['train'][0]} ภาพ (บวก {c['train'][1]}), val {c['val'][0]} ภาพ (บวก {c['val'][1]})")
    if c["train"][1] < 100 or c["val"][1] < 20:
        raise SystemExit("ข้อมูลน้อยเกินไปที่จะเทรน — กลับไปเพิ่มคลิปใน autolabel ก่อน")

    data = write_yaml(dataset_dir)
    model = YOLO(base)
    model.train(
        data=str(data), epochs=epochs, imgsz=imgsz, batch=batch, patience=patience,
        project=str(Path(project).resolve()), name=name, exist_ok=True, workers=2, device=device,
        # ลูกเล็กมาก: ไม่ย่อภาพแรง ไม่พลิกแนวตั้ง ไม่หมุน
        scale=0.3, degrees=0.0, flipud=0.0, fliplr=0.5, mosaic=1.0, close_mosaic=10,
        seed=0, plots=True,
    )
    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    m = YOLO(str(best)).val(data=str(data), imgsz=imgsz)
    print(f"ผลบนชุด val: precision {m.box.mp:.3f}  recall {m.box.mr:.3f}  mAP50 {m.box.map50:.3f}")
    return str(best)


def export(best_pt, imgsz=(384, 640), dest=None):
    """ส่งออกขนาดภาพกว้าง 640 x สูง 384 (16:9 ปัดเป็นพหุคูณของ 32) ประหยัดกว่าภาพจัตุรัส 640x640
    คืน dict รูปแบบ -> path  ที่ส่งออกสำเร็จ (ถ้าอันไหนล้มจะพิมพ์เหตุผลแล้วข้าม)"""
    from ultralytics import YOLO

    m = YOLO(best_pt)
    outs = {}
    for fmt, kw in (("onnx", dict(opset=12, simplify=True)), ("tflite", dict())):
        try:
            p = m.export(format=fmt, imgsz=list(imgsz), **kw)
            outs[fmt] = str(p)
            print("ส่งออก", fmt, "สำเร็จ:", p)
        except Exception as e:  # noqa: BLE001
            print("ส่งออก", fmt, "ไม่สำเร็จ:", repr(e)[:300])
    if dest:
        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy(best_pt, dest / "best.pt")
        for fmt, p in outs.items():
            src = Path(p)
            if src.is_dir():
                shutil.copytree(src, dest / src.name, dirs_exist_ok=True)
            else:
                shutil.copy(src, dest / src.name)
    return outs


def preview_predictions(best_pt, dataset_dir, n=12, imgsz=640, conf=0.25, seed=0, cols=3):
    """สุ่มภาพ val มาให้โมเดลที่เทรนแล้วทำนาย วงกลมแดง = ที่โมเดลเจอ ใช้ตรวจด้วยตาว่าเจอลูกจริงหรือเจอสิ่งอื่น
    คืน (path ของแผ่นภาพ, จำนวนภาพที่เจออย่างน้อย 1 จุด)"""
    import random

    import cv2
    import numpy as np
    from ultralytics import YOLO

    imgs = sorted((Path(dataset_dir) / "images" / "val").glob("*.jpg"))
    random.Random(seed).shuffle(imgs)
    m = YOLO(best_pt)
    tiles, hit = [], 0
    for p in imgs[:n]:
        r = m.predict(str(p), imgsz=imgsz, conf=conf, verbose=False)[0]
        im = cv2.imread(str(p))
        H, W = im.shape[:2]
        t = cv2.resize(im, (480, 270))
        boxes = r.boxes.xyxy.cpu().numpy()
        for b in boxes:
            cv2.circle(t, (int((b[0] + b[2]) / 2 * 480 / W), int((b[1] + b[3]) / 2 * 270 / H)), 14, (0, 0, 255), 2)
        hit += int(len(boxes) > 0)
        tiles.append(t)
    if not tiles:
        raise SystemExit("ไม่มีภาพ val")
    th, tw = tiles[0].shape[:2]
    rows = []
    for k in range(0, len(tiles), cols):
        row = tiles[k:k + cols]
        row += [np.zeros((th, tw, 3), np.uint8)] * (cols - len(row))
        rows.append(np.hstack(row))
    out = Path(dataset_dir) / "pred_val.jpg"
    cv2.imwrite(str(out), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 90])
    return str(out), hit
