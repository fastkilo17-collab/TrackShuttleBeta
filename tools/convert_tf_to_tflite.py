"""
แปลงน้ำหนัก TrackNet (TensorFlow 2 / Keras 2) -> TFLite

ที่มาโมเดล+น้ำหนัก: https://github.com/Chang-Chia-Chi/TrackNet-Badminton-Tracking-tensorflow2 (MIT)
  tools/tracknet_tf/TrackNet.py            โค้ดโมเดลต้นฉบับ (ห้ามแก้ ชื่อเลเยอร์ต้องตรงกับ checkpoint)
  tools/tracknet_tf/weights/TrackNet.*     น้ำหนักที่เทรนแล้ว

สเปกโมเดล (ต้องตรงกับ TrackNetDetector.kt):
  อินพุต  [1, 3, 288, 512]  = 3 เฟรมต่อเนื่อง ขาวดำ ค่า 0..1 (NCHW)
  เอาต์พุต [1, 1, 288, 512] = แผนที่ความน่าจะเป็นของลูก ณ เฟรมสุดท้าย

ต้องใช้ tensorflow 2.15.x (Keras 2) เพราะ Keras 3 โหลด checkpoint รูปแบบนี้ไม่ได้
  pip install tensorflow-cpu==2.15.1
  python tools/convert_tf_to_tflite.py --out app/src/main/assets/tracknet.tflite
"""
import argparse
import os
import sys

import numpy as np
import tensorflow as tf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "tracknet_tf"))
from TrackNet import ResNet_Track  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=os.path.join(HERE, "tracknet_tf", "weights", "TrackNet"))
    ap.add_argument("--height", type=int, default=288)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--out", default="tracknet.tflite")
    a = ap.parse_args()

    shape = (3, a.height, a.width)
    model = ResNet_Track(input_shape=shape)
    model.build(input_shape=(None,) + shape)

    status = model.load_weights(a.weights)
    status.expect_partial()  # ไม่สนตัวแปรของ optimizer
    try:
        status.assert_existing_objects_matched()
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"โหลดน้ำหนักไม่ครบ/ไม่ตรงโมเดล: {e}")
    print("โหลดน้ำหนักสำเร็จ จำนวนพารามิเตอร์:", model.count_params())

    @tf.function(input_signature=[tf.TensorSpec([1, *shape], tf.float32, name="input")])
    def serve(x):
        return {"output": model(x, training=False)}

    concrete = serve.get_concrete_function()
    conv = tf.lite.TFLiteConverter.from_concrete_functions([concrete], model)
    tflite = conv.convert()

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "wb") as f:
        f.write(tflite)
    print(f"เขียน {a.out} ({len(tflite) / 1e6:.1f} MB)")

    # ตรวจ: TFLite ต้องรันได้และรูปร่างเอาต์พุตตรงสเปก
    it = tf.lite.Interpreter(model_content=tflite)
    it.allocate_tensors()
    i_d, o_d = it.get_input_details()[0], it.get_output_details()[0]
    print("input ", i_d["shape"], i_d["dtype"])
    print("output", o_d["shape"], o_d["dtype"])

    rng = np.random.default_rng(0)
    x = rng.random((1, *shape), dtype=np.float32)
    it.set_tensor(i_d["index"], x)
    it.invoke()
    y_lite = it.get_tensor(o_d["index"])
    assert y_lite.size == a.height * a.width, f"รูปร่างเอาต์พุตผิด: {y_lite.shape}"

    # เทียบกับ Keras (อาจรันบน CPU ไม่ได้ถ้า TF ไม่รองรับ NCHW ก็แค่เตือน)
    try:
        y_ref = serve(tf.constant(x))["output"].numpy()
        diff = float(np.max(np.abs(y_ref.reshape(-1) - y_lite.reshape(-1))))
        print("ความต่างสูงสุด Keras vs TFLite:", diff)
        if diff > 1e-2:
            raise SystemExit("ผลของ TFLite ไม่ตรงกับ Keras")
    except tf.errors.OpError as e:
        print("ข้ามการเทียบกับ Keras (รันบน CPU ไม่ได้):", str(e)[:200])


if __name__ == "__main__":
    main()
