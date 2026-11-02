package com.example.shuttletrack

import android.content.Context
import android.graphics.PointF
import android.media.Image
import android.net.Uri

/**
 * ถอดรหัสวิดีโอต่อเนื่อง -> ภาพสี RGB ค่า 0..1 ขนาด outW x outH แบบ letterbox (คงสัดส่วน เติมขอบสีเทา 114)
 * ให้ตรงกับที่ ultralytics ใช้ตอนเทรน/ทำนาย (YOLO 640x384)
 *
 * ย่อภาพลงด้วยการเฉลี่ยกล่อง k x k เมื่อต้นฉบับใหญ่กว่าเอาต์พุต >= 1.5 เท่า (กัน aliasing ของลูกที่เล็กมาก
 * ตอนเทรนภาพถูกย่อด้วย INTER_AREA) สีใช้ BT.601 ช่วงจำกัด เหมือน OpenCV ที่ใช้ทำชุดข้อมูล
 */
class RgbLetterboxDecoder(
    context: Context,
    uri: Uri,
    val outW: Int,
    val outH: Int,
    private val nchw: Boolean
) {
    private val reader = YuvVideoReader(context, uri)
    val fps: Float get() = reader.fps
    val estimatedFrames: Int get() = reader.estimatedFrames

    private val n = outW * outH
    private val inside = BooleanArray(n)
    private val ix = IntArray(n)
    private val iy = IntArray(n)
    private val wx = FloatArray(n)
    private val wy = FloatArray(n)
    private val xn = IntArray(n)
    private val yn = IntArray(n)
    private var tX = IntArray(0)
    private var tY = IntArray(0)
    private var taps = 1

    private var tCw = -1
    private var tCh = -1
    private var tRot = -1
    private var yArr = ByteArray(0)
    private var uArr = ByteArray(0)
    private var vArr = ByteArray(0)
    private val out = FloatArray(3 * n)

    // พารามิเตอร์ letterbox (พิกเซลในภาพ outW x outH)
    var padX = 0f
        private set
    var padY = 0f
        private set
    var contentW = 1f
        private set
    var contentH = 1f
        private set

    /** onFrame(index, rgb) rgb ถูกใช้ซ้ำทุกเฟรม ต้องใช้ให้เสร็จก่อนกลับจาก callback; wants=false จะข้ามการแปลงเฟรมนั้น */
    fun decode(
        maxFrames: Int,
        isCancelled: () -> Boolean,
        wants: (Int) -> Boolean = { true },
        onFrame: (Int, FloatArray) -> Unit
    ): Int = reader.read(maxFrames, isCancelled, wants) { i, img, rot -> onFrame(i, convert(img, rot)) }

    /** พิกัดพิกเซลในภาพ letterbox -> พิกัด 0..1 ของเฟรมวิดีโอ (หลังหมุน) หรือ null ถ้าอยู่ในขอบเติม */
    fun normalize(x: Float, y: Float): PointF? {
        val u = (x - padX) / contentW
        val v = (y - padY) / contentH
        return if (u in 0f..1f && v in 0f..1f) PointF(u, v) else null
    }

    private fun buildTables(cw: Int, ch: Int, rot: Int) {
        val swap = rot == 90 || rot == 270
        val rw = if (swap) ch else cw
        val rh = if (swap) cw else ch
        val s = minOf(outW.toFloat() / rw, outH.toFloat() / rh)
        contentW = rw * s
        contentH = rh * s
        padX = (outW - contentW) / 2f
        padY = (outH - contentH) / 2f

        val scale = 1f / s // พิกเซลต้นฉบับต่อพิกเซลเอาต์พุต
        val k = if (scale >= 2.5f) 3 else if (scale >= 1.5f) 2 else 1
        taps = k * k
        if (k > 1) {
            tX = IntArray(taps * n)
            tY = IntArray(taps * n)
        }

        for (oy in 0 until outH) for (ox in 0 until outW) {
            val i = oy * outW + ox
            val px = ox + 0.5f - padX
            val py = oy + 0.5f - padY
            if (px < 0f || py < 0f || px >= contentW || py >= contentH) {
                inside[i] = false
                continue
            }
            inside[i] = true
            val u = px / contentW
            val v = py / contentH
            val sx = srcX(u, v, rot, cw, ch)
            val sy = srcY(u, v, rot, cw, ch)
            val fx = (sx - 0.5f).coerceIn(0f, (cw - 1).toFloat())
            val fy = (sy - 0.5f).coerceIn(0f, (ch - 1).toFloat())
            val x0 = fx.toInt()
            val y0 = fy.toInt()
            ix[i] = x0; iy[i] = y0
            wx[i] = fx - x0; wy[i] = fy - y0
            xn[i] = if (fx - x0 >= 0.5f) minOf(x0 + 1, cw - 1) else x0
            yn[i] = if (fy - y0 >= 0.5f) minOf(y0 + 1, ch - 1) else y0

            if (k > 1) {
                var t = 0
                for (a in 0 until k) for (b in 0 until k) {
                    val uu = u + ((b + 0.5f) / k - 0.5f) / contentW
                    val vv = v + ((a + 0.5f) / k - 0.5f) / contentH
                    tX[i * taps + t] = srcX(uu, vv, rot, cw, ch).toInt().coerceIn(0, cw - 1)
                    tY[i * taps + t] = srcY(uu, vv, rot, cw, ch).toInt().coerceIn(0, ch - 1)
                    t++
                }
            }
        }
        tCw = cw; tCh = ch; tRot = rot
    }

    private fun convert(img: Image, rotation: Int): FloatArray {
        val crop = img.cropRect
        val cw = crop.width()
        val ch = crop.height()
        if (cw != tCw || ch != tCh || rotation != tRot) buildTables(cw, ch, rotation)

        val pY = img.planes[0]
        val pU = img.planes[1]
        val pV = img.planes[2]
        yArr = fillBytes(pY.buffer, yArr)
        uArr = fillBytes(pU.buffer, uArr)
        vArr = fillBytes(pV.buffer, vArr)
        val yRs = pY.rowStride
        val yPs = pY.pixelStride
        val uvRs = pU.rowStride
        val uvPs = pU.pixelStride
        val padVal = 114f / 255f
        val inv = 1f / 255f
        val left = crop.left
        val top = crop.top

        for (i in 0 until n) {
            var r: Float
            var g: Float
            var b: Float
            if (!inside[i]) {
                r = padVal; g = padVal; b = padVal
            } else {
                val yv: Float
                if (taps == 1) {
                    val x0 = ix[i]
                    val y0 = iy[i]
                    val x1 = if (x0 + 1 < cw) x0 + 1 else x0
                    val y1 = if (y0 + 1 < ch) y0 + 1 else y0
                    val r0 = (top + y0) * yRs
                    val r1 = (top + y1) * yRs
                    val c0 = (left + x0) * yPs
                    val c1 = (left + x1) * yPs
                    val a = (yArr[r0 + c0].toInt() and 0xFF).toFloat()
                    val bb = (yArr[r0 + c1].toInt() and 0xFF).toFloat()
                    val c = (yArr[r1 + c0].toInt() and 0xFF).toFloat()
                    val d = (yArr[r1 + c1].toInt() and 0xFF).toFloat()
                    val tp = a + (bb - a) * wx[i]
                    val bt = c + (d - c) * wx[i]
                    yv = tp + (bt - tp) * wy[i]
                } else {
                    var sum = 0
                    val base = i * taps
                    for (t in 0 until taps) {
                        sum += yArr[(top + tY[base + t]) * yRs + (left + tX[base + t]) * yPs].toInt() and 0xFF
                    }
                    yv = sum.toFloat() / taps
                }
                val ci = ((top + yn[i]) shr 1) * uvRs + ((left + xn[i]) shr 1) * uvPs
                val u = (uArr[ci].toInt() and 0xFF) - 128
                val v = (vArr[ci].toInt() and 0xFF) - 128
                val yy = 1.164f * (yv - 16f)
                r = ((yy + 1.596f * v) * inv).coerceIn(0f, 1f)
                g = ((yy - 0.392f * u - 0.813f * v) * inv).coerceIn(0f, 1f)
                b = ((yy + 2.017f * u) * inv).coerceIn(0f, 1f)
            }
            if (nchw) {
                out[i] = r; out[n + i] = g; out[2 * n + i] = b
            } else {
                out[3 * i] = r; out[3 * i + 1] = g; out[3 * i + 2] = b
            }
        }
        return out
    }
}
