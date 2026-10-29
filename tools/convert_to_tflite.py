"""
แปลงน้ำหนัก TrackNet (PyTorch) -> TFLite สำหรับแอป

ใช้กับ TrackNetV2/V3 สำหรับแบดมินตัน (เช่น repo qaz812345/TrackNetV3 หรือ ChgygLin/TrackNetV2-pytorch)
ต้องแก้ 2 จุดที่มี TODO ให้ตรงกับ repo ที่ใช้:
  1) import คลาสโมเดล
  2) โหลด checkpoint

วิธีรัน:
  pip install torch onnx onnx2tf tensorflow
  python convert_to_tflite.py --weights best.pt --height 288 --width 512 --out ../app/src/main/assets/tracknet.tflite

ข้อกำหนดของแอป:
  - อินพุต: 3 เฟรม RGB ต่อกัน = 9 แชนเนล ค่า 0..1  (NCHW [1,9,H,W] หรือ NHWC [1,H,W,9] ก็ได้ แอปตรวจเอง)
  - เอาต์พุต: ฮีตแมป 3 แชนเนล ค่า 0..1 (ต้องผ่าน sigmoid แล้ว)
  ถ้าโมเดลของคุณใช้ 8 เฟรม (TrackNetV3) หรือมี background median ให้แก้ frameCount ใน TrackNetDetector.kt
"""
import argparse
import glob
import os
import shutil
import subprocess

import torch


def load_model(weights: str):
    # TODO 1: แก้ import ให้ตรงกับ repo ของคุณ
    from model import TrackNet  # type: ignore

    model = TrackNet(in_dim=9, out_dim=3)
    # TODO 2: แก้การโหลด checkpoint ให้ตรง
    ckpt = torch.load(weights, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt.get("model", ckpt)) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state)
    return model.eval()


class WithSigmoid(torch.nn.Module):
    """กันกรณีโมเดลคืน logits"""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        y = self.m(x)
        return y if bool(((y >= 0) & (y <= 1)).all()) else torch.sigmoid(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--height", type=int, default=288)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--out", default="tracknet.tflite")
    a = ap.parse_args()

    model = WithSigmoid(load_model(a.weights)).eval()
    dummy = torch.rand(1, 9, a.height, a.width)

    onnx_path = "tracknet.onnx"
    torch.onnx.export(model, dummy, onnx_path, opset_version=13,
                      input_names=["input"], output_names=["output"])

    subprocess.check_call(["onnx2tf", "-i", onnx_path, "-o", "tf_out"])

    candidates = sorted(glob.glob("tf_out/*float32.tflite")) or sorted(glob.glob("tf_out/*.tflite"))
    if not candidates:
        raise SystemExit("onnx2tf ไม่สร้างไฟล์ .tflite")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    shutil.copy(candidates[0], a.out)
    print("เสร็จ ->", a.out)


if __name__ == "__main__":
    main()
