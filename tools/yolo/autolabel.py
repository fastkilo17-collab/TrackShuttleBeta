# -*- coding: utf-8 -*-
"""
ขั้นที่ 1: ติดป้ายตำแหน่งลูกขนไก่อัตโนมัติ ด้วย TrackNet (tracknet.tflite) เป็น "ครู"
แล้วสร้างชุดข้อมูลรูปแบบ YOLO (images/ labels/ แยก train/val) สำหรับเทรนโมเดลเล็ก

ไหลของงาน
  วิดีโอ -> TrackNet (3 เฟรม, ปิดโซนกราฟิกบนฮีตแมป) -> จุดลูกต่อเฟรม (เก็บแคชเป็น JSON)
         -> กรอง: โซนกราฟิก, จุดที่นิ่งซ้ำ, จุดโดดเดี่ยวที่ไม่มีเพื่อนบ้านต่อเนื่อง
         -> บันทึกภาพ + ป้าย YOLO (+ภาพลบที่ไม่มีลูก)  -> contact_sheet() ให้ตรวจด้วยตา

ข้อควรรู้: ความแม่นของป้ายถูกจำกัดโดย TrackNet เอง จึงต้องเปิดดู contact_sheet ทุกครั้งก่อนเทรน
"""
import glob
import json
import os
import random
import re
import shutil
import time
from bisect import bisect_left
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np

TN_W, TN_H = 512, 288  # ขนาดอินพุตของ TrackNet
VIDEO_EXT = (".mp4", ".mov", ".mkv", ".avi", ".m4v", ".3gp")

CFG = dict(
    stride=2,            # ตรวจทุก N เฟรม (ยิ่งมาก ยิ่งเร็ว ข้อมูลยิ่งน้อย)
    max_frames=None,     # จำกัดจำนวนเฟรมต่อคลิป (None = ทั้งคลิป)
    thr=0.5,             # เกณฑ์ฮีตแมปของ TrackNet
    # โซนกราฟิกที่ต้องไม่นับ (สัดส่วนของภาพ x0,y0,x1,y1) ค่าเริ่มต้นตรงกับโอเวอร์เลย์ MDSportLive
    masks=[
        (0.69, 0.02, 1.00, 0.34),  # แผนที่สนามมุมขวาบน
        (0.00, 0.78, 0.11, 1.00),  # โลโก้ซ้ายล่าง
        (0.22, 0.82, 0.78, 1.00),  # แถบสกอร์ล่างกลาง
    ],
    static_cell=16,      # ขนาดช่องตารางหาจุดนิ่ง (พิกเซลที่ 512x288)
    static_frac=0.02,    # ช่องที่มีจุดซ้ำเกินสัดส่วนนี้ของเฟรมที่ตรวจ = จุดนิ่ง (ไม่ใช่ลูก)
    static_min=6,        # และต้องซ้ำอย่างน้อยกี่ครั้ง
    max_gap=6,           # ต้องมีจุดเพื่อนบ้านภายในกี่เฟรม
    max_speed=45.0,      # ลูกเคลื่อนได้มากสุดกี่พิกเซล (ที่ 512x288) ต่อเฟรม
    neg_ratio=0.15,      # สัดส่วนภาพลบ (ไม่มีลูก) เทียบกับภาพบวก
    neg_margin=30,       # ภาพลบต้องห่างจากทุกการตรวจจับที่น่าเชื่อถืออย่างน้อยกี่เฟรม (ลูกที่บินผ่านโซนกราฟิกจะมองไม่เห็น จึงต้องเผื่อกว้าง)
    val_frac=0.2,        # ท้ายคลิปเท่านี้เป็นชุดตรวจสอบ (แบ่งตามเวลา ไม่สุ่ม เพื่อกันข้อมูลรั่ว)
    save_width=1280,     # ความกว้างภาพที่บันทึก
    jpg_quality=92,
    box_scale=1.2,       # ขยายกรอบจากขนาด blob ของฮีตแมป
    box_min=12,          # ขนาดกรอบต่ำสุด/สูงสุด (พิกเซลที่ความกว้าง 1280)
    box_max=48,
    seed=0,
)


# ---------------------------------------------------------------- ครู: TrackNet TFLite
class TFLiteTeacher:
    def __init__(self, path, threads=2):
        try:
            import tensorflow as tf
            Interpreter = tf.lite.Interpreter
        except Exception:  # noqa: BLE001
            from ai_edge_litert.interpreter import Interpreter
        self.it = Interpreter(model_path=str(path), num_threads=threads)
        self.it.allocate_tensors()
        self.i = self.it.get_input_details()[0]
        self.o = self.it.get_output_details()[0]
        s = self.i["shape"]
        self.nchw = int(s[1]) == 3  # [1,3,H,W] หรือ [1,H,W,3]

    def predict(self, g0, g1, g2):
        """g*: float32 (288,512) ค่า 0..1 เก่า->ใหม่. คืนฮีตแมป (288,512) ของเฟรมที่ 3"""
        st = np.stack([g0, g1, g2], axis=0 if self.nchw else -1)[None].astype(np.float32)
        self.it.set_tensor(self.i["index"], st)
        self.it.invoke()
        return self.it.get_tensor(self.o["index"]).reshape(TN_H, TN_W)


def gray_small(frame):
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (TN_W, TN_H))
    return g.astype(np.float32) / 255.0


def largest_blob(heat, thr=0.5):
    """เหมือนแอปและ predict.py ต้นฉบับ: blob ที่กรอบใหญ่สุด -> (cx, cy, w, h) หรือ None"""
    m = (heat > thr).astype(np.uint8)
    if not m.any():
        return None
    n, _, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=4)
    best, best_area = None, 0
    for i in range(1, n):
        x, y, w, h = (int(v) for v in stats[i][:4])
        if w * h > best_area:
            best_area = w * h
            best = (x + w / 2.0, y + h / 2.0, float(w), float(h))
    return best


# ---------------------------------------------------------------- ขั้น A: ตรวจทั้งคลิป (แคช)
def _mask_array(masks):
    """ภาพ 288x512: 0 ในโซนกราฟิก, 1 ที่อื่น (ใช้คูณฮีตแมป เพื่อไม่ให้กราฟิกสว่างๆ แย่งชนะลูกจริง)"""
    m = np.ones((TN_H, TN_W), np.float32)
    for x0, y0, x1, y1 in masks or []:
        m[int(y0 * TN_H):int(np.ceil(y1 * TN_H)), int(x0 * TN_W):int(np.ceil(x1 * TN_W))] = 0.0
    return m


def detect_video(video, teacher, stride=2, max_frames=None, thr=0.5, cache_path=None, log_every=50, masks=None):
    video = str(video)
    params = dict(stride=stride, thr=thr, max_frames=max_frames, size=os.path.getsize(video),
                  masks=[list(m) for m in (masks or [])])
    keep_mask = _mask_array(masks)
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            c = json.load(f)
        if c.get("params") == params:
            print(f"  ใช้แคช {os.path.basename(cache_path)}")
            c["dets"] = {int(k): v for k, v in c["dets"].items()}
            return c

    cap = cv2.VideoCapture(video)
    win = deque(maxlen=3)
    dets, idx, W, H, done = {}, -1, 0, 0, 0
    est = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    t0 = time.time()
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        idx += 1
        if max_frames and idx >= max_frames:
            break
        if idx == 0:
            H, W = fr.shape[:2]
        win.append(gray_small(fr))
        if len(win) == 3 and idx % stride == 0:
            b = largest_blob(teacher.predict(*win) * keep_mask, thr)
            dets[idx] = None if b is None else [float(v) for v in b]
            done += 1
            if done % log_every == 0:
                el = time.time() - t0
                total = (min(est, max_frames) if max_frames else est) // stride
                eta = f" เหลือราว {(total - done) * el / done / 60:.1f} นาที" if total > done else ""
                print(f"  ตรวจแล้ว {done}/{total or '?'} ครั้ง ใช้ {el / 60:.1f} นาที{eta}")
    cap.release()
    res = dict(params=params, n=idx + 1, w=W, h=H, stride=stride, dets=dets)
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(res, f)
    return res


# ---------------------------------------------------------------- ขั้น B: กรอง
def _in_masks(nx, ny, masks):
    return any(x0 <= nx <= x1 and y0 <= ny <= y1 for x0, y0, x1, y1 in masks)


def filter_dets(dets, cfg):
    """คืน (positives, plausible, stats)
    positives = ผ่านทุกตัวกรอง (ใช้ติดป้าย)
    plausible = ผ่านโซนกราฟิก+จุดนิ่ง (ใช้กันไม่ให้สุ่มภาพลบใกล้ที่น่าจะมีลูก)"""
    stats = {}
    pts = {i: d for i, d in dets.items() if d is not None}
    stats["raw"] = len(pts)

    pts = {i: d for i, d in pts.items() if not _in_masks(d[0] / TN_W, d[1] / TN_H, cfg["masks"])}
    stats["after_mask"] = len(pts)

    cell = cfg["static_cell"]
    limit = max(cfg["static_min"], cfg["static_frac"] * max(1, len(dets)))
    cnt = Counter((int(d[0] // cell), int(d[1] // cell)) for d in pts.values())
    pts = {i: d for i, d in pts.items() if cnt[(int(d[0] // cell), int(d[1] // cell))] <= limit}
    stats["after_static"] = len(pts)
    plausible = dict(pts)

    idxs = sorted(pts)
    keep = {}
    for k, i in enumerate(idxs):
        ok = False
        for step in (-1, 1):
            j = k + step
            while 0 <= j < len(idxs) and abs(idxs[j] - i) <= cfg["max_gap"]:
                gap = abs(idxs[j] - i)
                dx = pts[idxs[j]][0] - pts[i][0]
                dy = pts[idxs[j]][1] - pts[i][1]
                if (dx * dx + dy * dy) ** 0.5 <= cfg["max_speed"] * gap:
                    ok = True
                    break
                j += step
            if ok:
                break
        if ok:
            keep[i] = pts[i]
    stats["after_temporal"] = len(keep)
    return keep, plausible, stats


# ---------------------------------------------------------------- ขั้น C: บันทึกชุดข้อมูล
def _safe(name):
    s = re.sub(r"[^0-9A-Za-z_-]+", "_", Path(name).stem).strip("_")
    return s or "clip"


def build_dataset(videos, caches, out_dir, cfg):
    out = Path(out_dir)
    for sub in ("images", "labels"):
        shutil.rmtree(out / sub, ignore_errors=True)
        for sp in ("train", "val"):
            (out / sub / sp).mkdir(parents=True, exist_ok=True)

    rnd = random.Random(cfg["seed"])
    total = Counter()
    for video, c in zip(videos, caches):
        dets = c["dets"]
        pos, plaus, st = filter_dets(dets, cfg)

        pl_idx = sorted(plaus)
        neg_pool = []
        for i in dets:
            if i in plaus:  # เฟรมที่ถูกตัดทิ้งเป็นจุดนิ่ง/โซนกราฟิกก็นับเป็นผู้สมัครภาพลบได้
                continue
            k = bisect_left(pl_idx, i)
            near = min((abs(pl_idx[j] - i) for j in (k - 1, k) if 0 <= j < len(pl_idx)), default=10 ** 9)
            if near > cfg["neg_margin"]:
                neg_pool.append(i)
        n_neg = min(len(neg_pool), int(round(len(pos) * cfg["neg_ratio"])))
        neg = set(rnd.sample(neg_pool, n_neg)) if n_neg > 0 else set()

        want = {**{i: pos[i] for i in pos}, **{i: None for i in neg}}
        W, H, n = c["w"], c["h"], c["n"]
        save_w = min(cfg["save_width"], W)
        save_h = int(round(H * save_w / W))
        k_ref = save_w / 1280.0
        stem = _safe(video)
        split_at = n * (1.0 - cfg["val_frac"])
        got = Counter()

        cap = cv2.VideoCapture(str(video))
        i = -1
        while want:
            ok, fr = cap.read()
            if not ok:
                break
            i += 1
            if i not in want:
                continue
            d = want.pop(i)
            split = "val" if i >= split_at else "train"
            name = f"{stem}_{i:06d}"
            img = fr if save_w == W else cv2.resize(fr, (save_w, save_h), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(out / "images" / split / f"{name}.jpg"), img,
                        [cv2.IMWRITE_JPEG_QUALITY, cfg["jpg_quality"]])
            lp = out / "labels" / split / f"{name}.txt"
            if d is None:
                lp.write_text("")
                got[f"neg_{split}"] += 1
            else:
                side = max(d[2], d[3]) * (save_w / TN_W) * cfg["box_scale"]
                side = min(max(side, cfg["box_min"] * k_ref), cfg["box_max"] * k_ref)
                cx, cy = d[0] / TN_W, d[1] / TN_H
                lp.write_text(f"0 {cx:.6f} {cy:.6f} {side / save_w:.6f} {side / save_h:.6f}\n")
                got[f"pos_{split}"] += 1
        cap.release()

        print(f"  {os.path.basename(str(video))}: ตรวจได้ {st['raw']} -> ตัดโซนกราฟิก {st['after_mask']} "
              f"-> ตัดจุดนิ่ง {st['after_static']} -> ตัดจุดโดดเดี่ยว {st['after_temporal']} "
              f"| บวก train/val {got['pos_train']}/{got['pos_val']} ลบ {got['neg_train'] + got['neg_val']}")
        total.update(got)
    return dict(total)


# ---------------------------------------------------------------- ตรวจด้วยตา
def contact_sheet(out_dir, n=24, mode="crop", cols=6, seed=0, out_name=None, negatives=False):
    """รวมภาพตัวอย่างเป็นแผ่นเดียวให้ตรวจด้วยตา
    mode='crop' = ซูมรอบลูก, 'full' = ทั้งเฟรมวงกลมแดงที่ลูก
    negatives=True = สุ่มภาพ "ไม่มีลูก" มาดูทั้งเฟรม (ต้องไม่เห็นลูกขนไก่ในภาพเหล่านี้)"""
    out = Path(out_dir)
    want_empty = bool(negatives)
    labels = [p for p in sorted((out / "labels").glob("*/*.txt")) if (p.stat().st_size == 0) == want_empty]
    if not labels:
        raise SystemExit("ไม่มีภาพที่จะแสดง — ลองลด stride หรือเช็ก masks/thr")
    random.Random(seed).shuffle(labels)
    if negatives:
        mode, cols = "full", min(cols, 3)
    tiles = []
    for lp in labels[:n]:
        img = cv2.imread(str(out / "images" / lp.parent.name / f"{lp.stem}.jpg"))
        H, W = img.shape[:2]
        if negatives:
            cx = cy = bw = bh = 0.0
        else:
            _, cx, cy, bw, bh = (float(v) for v in lp.read_text().split()[:5])
            cx, cy, bw, bh = cx * W, cy * H, bw * W, bh * H
        if mode == "crop":
            half, s = 48, 3
            pad = cv2.copyMakeBorder(img, half, half, half, half, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            x, y = int(round(cx)), int(round(cy))
            t = pad[y:y + 2 * half, x:x + 2 * half]
            t = cv2.resize(t, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
            p1 = (int((half - bw / 2) * s), int((half - bh / 2) * s))
            p2 = (int((half + bw / 2) * s), int((half + bh / 2) * s))
            cv2.rectangle(t, p1, p2, (0, 0, 255), 1)
        else:
            sc = 480.0 / W
            t = cv2.resize(img, (480, int(round(H * sc))))
            if not negatives:
                cv2.circle(t, (int(cx * sc), int(cy * sc)), 14, (0, 0, 255), 2)
            t = cv2.resize(t, (480, 270))
        cv2.putText(t, lp.stem[-6:], (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        tiles.append(t)
    th, tw = tiles[0].shape[:2]
    rows = []
    for r in range(0, len(tiles), cols):
        row = tiles[r:r + cols]
        row += [np.zeros((th, tw, 3), np.uint8)] * (cols - len(row))
        rows.append(np.hstack(row))
    sheet = np.vstack(rows)
    path = out / (out_name or ("contact_neg.jpg" if negatives else f"contact_{mode}.jpg"))
    cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return str(path)


# ---------------------------------------------------------------- ทางเข้าหลัก
def run(videos_dir, tracknet_tflite, out_dir, **overrides):
    cfg = {**CFG, **overrides}
    videos = sorted(p for p in glob.glob(os.path.join(videos_dir, "*"))
                    if p.lower().endswith(VIDEO_EXT))
    if not videos:
        raise SystemExit(f"ไม่พบวิดีโอใน {videos_dir}")
    teacher = TFLiteTeacher(tracknet_tflite)
    caches = []
    for v in videos:
        print("คลิป:", os.path.basename(v))
        caches.append(detect_video(v, teacher, cfg["stride"], cfg["max_frames"], cfg["thr"],
                                   cache_path=os.path.join(out_dir, "cache", _safe(v) + ".json"),
                                   masks=cfg["masks"]))
    print("สร้างชุดข้อมูล...")
    total = build_dataset(videos, caches, out_dir, cfg)
    pos = total.get("pos_train", 0) + total.get("pos_val", 0)
    print(f"รวม: บวก {pos} (train {total.get('pos_train', 0)} / val {total.get('pos_val', 0)}) "
          f"ลบ {total.get('neg_train', 0) + total.get('neg_val', 0)}")
    if pos < 300:
        print("คำเตือน: ภาพบวกน้อยกว่า ~300 เทรนแล้วน่าจะไม่เสถียร — เพิ่มคลิป/ลด stride")
    return total
