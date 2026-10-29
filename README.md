# ShuttleTrack — วิเคราะห์ลูกขนไก่จากวิดีโอด้วย TrackNet (Android)

เฟส 1: เลือกวิดีโอในเครื่อง → TrackNet (TFLite รันบนมือถือ) → แสดงจุดลูกและหางวิถีทีละเฟรม → ส่งออก CSV
เฟส 2 (อนาคต): กล้องสด — `TrackNetDetector.detect(3 เฟรม)` ไม่ผูกกับวิดีโอ ใช้ต่อกับ CameraX ImageAnalysis ได้เลย

## สร้าง APK

### วิธี A: Android Studio
1. เปิดโฟลเดอร์นี้ด้วย Android Studio (Hedgehog ขึ้นไป) ให้ sync Gradle
2. รันสคริปต์แปลงโมเดลตามหัวข้อ "โมเดล" ด้านล่าง
3. Build > Build APK(s) → ได้ `app/build/outputs/apk/debug/app-debug.apk`

### วิธี B: GitHub Actions (ไม่ต้องลงอะไรในเครื่อง)
แท็บ Actions → Build APK → Run workflow → ดาวน์โหลด artifact `ShuttleTrack-debug-apk`

## โมเดล
ใช้ TrackNet TensorFlow 2 ของ [Chang-Chia-Chi](https://github.com/Chang-Chia-Chi/TrackNet-Badminton-Tracking-tensorflow2) (MIT)
พร้อมน้ำหนักที่เทรนแล้วใน `tools/tracknet_tf/` — GitHub Actions จะแปลงเป็น `tracknet.tflite` และฝังลง APK ให้อัตโนมัติ
(สเปก: อินพุต 3 เฟรมขาวดำ 512x288, เอาต์พุตแผนที่ลูก ณ เฟรมสุดท้าย)

แปลงเองในเครื่อง: `pip install tensorflow-cpu==2.15.1` แล้ว
`python tools/convert_tf_to_tflite.py --out app/src/main/assets/tracknet.tflite`

## การใช้งาน
เปิดแอป → "เลือกวิดีโอ" → รอวิเคราะห์ (จำกัด 900 เฟรมแรก; ผลเริ่มตั้งแต่เฟรมที่ 3) → เลื่อนแถบดูผลทีละเฟรม → "ส่งออก CSV"
(CSV: frame, time_s, visible, x_norm, y_norm)

## ข้อจำกัดที่ควรรู้
- ยังไม่ได้ทดสอบบนเครื่องจริง/ยังไม่ได้คอมไพล์ในที่นี้ — รอบแรกอาจต้องแก้เล็กน้อย
- ดึงเฟรมด้วย MediaMetadataRetriever ซึ่งช้าและไม่แม่นระดับเฟรมเป๊ะ เหมาะกับการทดลอง; ถ้าจะจริงจังควรเปลี่ยนเป็น MediaCodec
- ยังไม่มีขั้นกรองวิถี/ตรวจจุดตกพื้น (ต่อยอดจาก CSV ได้)
