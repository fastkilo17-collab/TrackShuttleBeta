# ShuttleTrack — วิเคราะห์ลูกขนไก่จากวิดีโอด้วย TrackNet (Android)

เฟส 1: เลือกวิดีโอในเครื่อง → TrackNet (TFLite รันบนมือถือ) → แสดงจุดลูกและหางวิถีทีละเฟรม → ส่งออก CSV
เฟส 2 (อนาคต): กล้องสด — `TrackNetDetector.detect(3 เฟรม)` ไม่ผูกกับวิดีโอ ใช้ต่อกับ CameraX ImageAnalysis ได้เลย

## สร้าง APK

### วิธี A: Android Studio
1. เปิดโฟลเดอร์นี้ด้วย Android Studio (Hedgehog ขึ้นไป) ให้ sync Gradle
2. ทำขั้นตอน "เตรียมโมเดล" ด้านล่าง
3. Build > Build APK(s) → ได้ `app/build/outputs/apk/debug/app-debug.apk`

### วิธี B: GitHub Actions (ไม่ต้องลงอะไรในเครื่อง)
อัปโหลดโปรเจกต์ขึ้น GitHub (รวมไฟล์ `tracknet.tflite` ใน assets) → แท็บ Actions → Build APK → ดาวน์โหลด artifact

## เตรียมโมเดล (ต้องทำก่อนสร้าง APK)
แอปไม่แถมน้ำหนักโมเดล เพราะต้องตรงกับ repo/ไลเซนส์ที่คุณเลือก
1. เอา TrackNetV2/V3 (PyTorch) สำหรับแบดมินตัน พร้อม checkpoint
2. แก้ 2 จุด TODO ใน `tools/convert_to_tflite.py` แล้วรัน
   `python convert_to_tflite.py --weights best.pt --out ../app/src/main/assets/tracknet.tflite`
3. ข้อกำหนด: อินพุต 3 เฟรม RGB (9 แชนเนล, 0..1) ขนาด 512x288; เอาต์พุตฮีตแมป 3 ภาพ

ถ้าโมเดลคุณใช้จำนวนเฟรมอื่น (เช่น V3 ใช้ 8 เฟรม + background) ให้แก้ `frameCount` ใน `TrackNetDetector.kt` และลูปใน `VideoAnalyzer.kt`

## การใช้งาน
เปิดแอป → "เลือกวิดีโอ" → รอวิเคราะห์ (จำกัด 900 เฟรมแรก) → เลื่อนแถบดูผลทีละเฟรม → "ส่งออก CSV"
(CSV: frame, time_s, visible, x_norm, y_norm)

## ข้อจำกัดที่ควรรู้
- ยังไม่ได้ทดสอบบนเครื่องจริง/ยังไม่ได้คอมไพล์ในที่นี้ — รอบแรกอาจต้องแก้เล็กน้อย
- ดึงเฟรมด้วย MediaMetadataRetriever ซึ่งช้าและไม่แม่นระดับเฟรมเป๊ะ เหมาะกับการทดลอง; ถ้าจะจริงจังควรเปลี่ยนเป็น MediaCodec
- ยังไม่มีขั้นกรองวิถี/ตรวจจุดตกพื้น (ต่อยอดจาก CSV ได้)
