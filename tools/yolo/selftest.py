# -*- coding: utf-8 -*-
"""ทดสอบตรรกะของ autolabel.py ด้วยวิดีโอสังเคราะห์ + ครูจำลอง (ไม่ต้องใช้ TensorFlow)
รัน: python tools/yolo/selftest.py"""
import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import autolabel as A  # noqa: E402

W, H, N = 640, 360, 300
STATIC = (100, 300)          # จุดสว่างนิ่งขนาดเล็ก = ตัวหลอก
OVERLAY = (580, 60)          # จุดสว่างในโซนแผนที่มุมขวาบน = ตัวหลอก (อยู่ใน mask)
PAUSE = range(100, 200)      # ช่วงพักแรลลี่ ไม่มีลูก (ยาวกว่า 2 เท่าของ neg_margin)


def ball_pos(i):
    t = i / N
    return int(40 + 520 * t), int(260 - 120 * np.sin(np.pi * (i % 50) / 50))  # y>=140 พ้นโซนกราฟิก


def make_video(path):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (W, H))
    rng = np.random.default_rng(1)
    for i in range(N):
        fr = (rng.random((H, W, 3)) * 40).astype(np.uint8)
        cv2.circle(fr, STATIC, 1, (255, 255, 255), -1)
        cv2.circle(fr, OVERLAY, 3, (255, 255, 255), -1)
        if i not in PAUSE:
            cv2.circle(fr, ball_pos(i), 3, (255, 255, 255), -1)
        vw.write(fr)
    vw.release()


class StubTeacher:
    """ฮีตแมป = พิกเซลสว่างมากในเฟรมล่าสุด"""
    def predict(self, g0, g1, g2):
        return (g2 > 0.9).astype(np.float32)


def main():
    tmp = Path(tempfile.mkdtemp())
    vid = tmp / "สนาม test 1.mp4"
    make_video(vid)
    cfg = {**A.CFG, "stride": 2, "box_min": 12}
    cp = str(tmp / "out" / "cache" / "c.json")
    c = A.detect_video(vid, StubTeacher(), stride=2, thr=0.5, cache_path=cp, masks=cfg["masks"])
    c2 = A.detect_video(vid, StubTeacher(), stride=2, thr=0.5, cache_path=cp, masks=cfg["masks"])
    assert c2["dets"].keys() == c["dets"].keys(), "แคชต้องให้ผลเหมือนเดิม"

    total = A.build_dataset([vid], [c], tmp / "out", cfg)
    print("total:", total)

    out = tmp / "out"
    labs = {sp: sorted((out / "labels" / sp).glob("*.txt")) for sp in ("train", "val")}
    n_pos = total.get("pos_train", 0) + total.get("pos_val", 0)
    assert n_pos > 40, f"ภาพบวกน้อยผิดปกติ {n_pos}"

    split_at = N * (1 - cfg["val_frac"])
    for sp, files in labs.items():
        for lp in files:
            i = int(lp.stem.split("_")[-1])
            assert (i >= split_at) == (sp == "val"), "แบ่ง train/val ต้องตามเวลา"
            assert i not in PAUSE or lp.stat().st_size == 0, "ช่วงพักต้องไม่มีป้ายบวก"
            if lp.stat().st_size:
                _, cx, cy, bw, bh = (float(v) for v in lp.read_text().split())
                bx, by = ball_pos(i)
                assert abs(cx * W - bx) < 6 and abs(cy * H - by) < 6, f"ป้ายผิดที่ เฟรม {i}"
                assert 0 < bw < 0.1 and 0 < bh < 0.1
                img = cv2.imread(str(out / "images" / sp / f"{lp.stem}.jpg"))
                assert img is not None
    n_neg = total.get("neg_train", 0) + total.get("neg_val", 0)
    assert n_neg > 0, "ต้องมีภาพลบจากช่วงพัก"
    for sp, files in labs.items():
        for lp in files:
            if lp.stat().st_size == 0:
                assert int(lp.stem.split("_")[-1]) in PAUSE, "ภาพลบต้องมาจากช่วงพักเท่านั้น"

    pos, plaus, st = A.filter_dets(c["dets"], cfg)
    print("filter stats:", st)
    # โซนกราฟิกถูกปิดบนฮีตแมปแล้ว: ลูกที่บินผ่านโซนนั้นจะหายไป แต่ตัวหลอกต้องไม่ถูกเลือกแทนลูกที่อยู่นอกโซน
    assert not any(d is not None and abs(d[0] / A.TN_W * W - OVERLAY[0]) < 6 and abs(d[1] / A.TN_H * H - OVERLAY[1]) < 6
                   for d in c["dets"].values()), "ต้องไม่เลือกจุดในโซนกราฟิก"
    assert all(abs(d[0] / A.TN_W * W - STATIC[0]) > 8 or abs(d[1] / A.TN_H * H - STATIC[1]) > 8
               for d in plaus.values()), "ต้องตัดจุดนิ่งออก"

    for kw in (dict(mode="crop"), dict(mode="full"), dict(negatives=True)):
        p = A.contact_sheet(out, n=12, **kw)
        assert Path(p).exists() and cv2.imread(p) is not None
    print("PASS")


if __name__ == "__main__":
    main()
