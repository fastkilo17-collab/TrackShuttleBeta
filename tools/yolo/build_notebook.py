# -*- coding: utf-8 -*-
"""สร้าง MDSportLive_yolo_shuttle.ipynb จาก autolabel.py + train.py (รัน: python tools/yolo/build_notebook.py)"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def src(name):
    with open(os.path.join(HERE, name), encoding="utf-8") as f:
        return f.read()


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(True)}


cells = [
    md("""
# เทรนตัวตรวจจับลูกขนไก่เบา ๆ (YOLO) จากครู TrackNet

**ก่อนเริ่ม:** เมนู *รันไทม์ → เปลี่ยนประเภทรันไทม์ → T4 GPU* (ใช้ตอนเทรน) แล้วกดรันทีละช่องตามลำดับ

เตรียมใน Google Drive โฟลเดอร์ `MyDrive/shuttle/` :
- `videos/`  คลิปสนามของคุณ 3–5 คลิป (.mp4) แนวนอน ยิ่งหลากหลายแสง/มุมยิ่งดี และถ้ามีคลิปจากกล้องดิบที่ไม่มีโอเวอร์เลย์สกอร์ ยิ่งดีกว่า
- `tracknet.tflite`  ไฟล์ครู (GitHub → Actions → รันล่าสุด → artifact `tracknet-tflite`)
"""),
    code("""
!nvidia-smi -L
!pip -q install ultralytics
"""),
    code("""
from google.colab import drive
drive.mount('/content/drive')

BASE = '/content/drive/MyDrive/shuttle'
VIDEOS = f'{BASE}/videos'
TRACKNET = f'{BASE}/tracknet.tflite'
WORK = '/content/work'

import os, glob, shutil
os.makedirs(VIDEOS, exist_ok=True)
os.makedirs(WORK, exist_ok=True)
print('คลิป:', [os.path.basename(p) for p in sorted(glob.glob(VIDEOS + '/*'))])
print('มีไฟล์ครู tracknet.tflite:', os.path.exists(TRACKNET))
"""),
    code("%%writefile autolabel.py\n" + src("autolabel.py")),
    code("%%writefile train.py\n" + src("train.py")),
    md("""
## ขั้นที่ 1: ติดป้ายอัตโนมัติ
รอบแรกลองคลิปสั้นก่อน (`max_frames=600` ≈ 20 วินาทีต่อคลิป) ใช้ CPU จึงอาจช้า ช่องนี้จะบอกเวลาที่เหลือ
ถ้าคลิปของคุณไม่มีโอเวอร์เลย์เหมือนตัวอย่าง ให้ตั้ง `masks=[]` หรือแก้โซนให้ตรง (สัดส่วนของภาพ x0,y0,x1,y1)
"""),
    code("""
import importlib, autolabel as A
importlib.reload(A)

DS = f'{WORK}/dataset'
# กู้แคชการตรวจจากรอบก่อน (ถ้ามี) เพื่อไม่ต้องรันครูซ้ำ
if os.path.isdir(f'{BASE}/cache') and not os.path.isdir(f'{DS}/cache'):
    shutil.copytree(f'{BASE}/cache', f'{DS}/cache')

total = A.run(VIDEOS, TRACKNET, DS, stride=2, max_frames=600)

# เก็บแคชไว้ใน Drive
shutil.copytree(f'{DS}/cache', f'{BASE}/cache', dirs_exist_ok=True)
"""),
    md("""
## ตรวจป้ายด้วยตา (สำคัญที่สุด)
ป้ายมาจาก TrackNet ซึ่งผิดได้ ถ้าป้ายผิด โมเดลที่เทรนก็จะผิดตาม
- **ภาพซูม:** กรอบแดงต้องคร่อมลูกขนไก่จริง ถ้าเห็นกรอบอยู่บนพื้น/เสื้อ/แสงสะท้อนเกินครึ่ง อย่าเพิ่งเทรน → ปรับ `thr` (เช่น 0.6), `static_frac`, `masks` แล้วรันช่องข้างบนซ้ำ (เร็วเพราะมีแคช)
- **ภาพลบ:** ต้องไม่เห็นลูกขนไก่ในภาพเหล่านี้เลย
"""),
    code("""
from IPython.display import Image, display
print('ภาพซูมรอบลูก (กรอบแดง = ป้ายที่จะใช้เทรน)')
display(Image(A.contact_sheet(DS, n=24, mode='crop')))
print('ทั้งเฟรม (วงกลมแดง = ตำแหน่งลูก)')
display(Image(A.contact_sheet(DS, n=12, mode='full', cols=3)))
print('ภาพลบ (ต้องไม่มีลูก)')
display(Image(A.contact_sheet(DS, n=12, negatives=True)))
"""),
    md("""
## ขั้นที่ 2: เทรน
ใช้เวลาราว 10–40 นาทีบน T4 (ขึ้นกับจำนวนภาพ) เมื่อป้ายดูถูกต้องแล้วจึงรัน
"""),
    code("""
import train as T
importlib.reload(T)
best = T.train(DS, project=f'{WORK}/runs', epochs=80, imgsz=640)
"""),
    md("""
## ดูผลทำนาย
**ตัวเลข precision/recall ข้างบนวัดว่า "ตรงกับป้ายของ TrackNet" ไม่ใช่ความแม่นจริง** ต้องดูภาพต่อไปนี้ด้วยตา:
วงกลมแดง = จุดที่โมเดลเจอ ดูว่าเป็นลูกจริงกี่ภาพ และมีภาพที่ลูกอยู่แต่โมเดลไม่เจอกี่ภาพ
"""),
    code("""
path, hit = T.preview_predictions(best, DS, n=12, imgsz=640, conf=0.25)
print(f'เจออย่างน้อย 1 จุดใน {hit}/12 ภาพ')
display(Image(path))
"""),
    md("""
## ส่งออกและเก็บลง Drive
ส่งออกที่ขนาด 640x384 (16:9) ได้ ONNX และ TFLite ถ้าอันไหนส่งออกไม่สำเร็จ ช่องนี้จะบอกเหตุผล ไม่ต้องกังวล
ไฟล์จะอยู่ที่ `MyDrive/shuttle/model_out/`
"""),
    code("""
outs = T.export(best, imgsz=(384, 640), dest=f'{BASE}/model_out')
print(outs)
"""),
    md("""
## ขั้นต่อไป
ส่งไฟล์ในโฟลเดอร์ `model_out` (อย่างน้อย `best.pt` และ `.onnx`/`.tflite` ที่ได้) กลับมาให้ Claude
เพื่อนำเข้าแอป ShuttleTrack และวัดความเร็วต่อเฟรมบนมือถือของคุณจริง
"""),
]

nb = {
    "cells": cells,
    "metadata": {
        "colab": {"provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "accelerator": "GPU",
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
out = os.path.join(HERE, "MDSportLive_yolo_shuttle.ipynb")
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("เขียน", out)
