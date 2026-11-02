package com.example.shuttletrack

import android.content.Context
import android.media.Image
import android.net.Uri

/**
 * ถอดรหัสวิดีโอต่อเนื่องแล้วเอาเฉพาะแชนเนลความสว่าง (Y) ย่อเป็น outW x outH ค่า 0..1 (สำหรับ TrackNet)
 * รองรับวิดีโอที่หมุน (rotation metadata) โดยสุ่มตัวอย่างในพิกัดหลังหมุน
 */
class GrayFrameDecoder(
    context: Context,
    uri: Uri,
    private val outW: Int,
    private val outH: Int
) {
    private val reader = YuvVideoReader(context, uri)
    val fps: Float get() = reader.fps
    val estimatedFrames: Int get() = reader.estimatedFrames

    private var tCw = -1
    private var tCh = -1
    private var tRot = -1
    private val n = outW * outH
    private val ix = IntArray(n)
    private val iy = IntArray(n)
    private val wx = FloatArray(n)
    private val wy = FloatArray(n)
    private var yArr = ByteArray(0)

    /** คืนจำนวนเฟรมที่ถอดได้ onFrame(index, gray[outW*outH]) */
    fun decode(maxFrames: Int, isCancelled: () -> Boolean, onFrame: (Int, FloatArray) -> Unit): Int =
        reader.read(maxFrames, isCancelled, { true }) { i, img, rot -> onFrame(i, toGray(img, rot)) }

    private fun buildTables(cw: Int, ch: Int, rot: Int) {
        for (oy in 0 until outH) for (ox in 0 until outW) {
            val u = (ox + 0.5f) / outW
            val v = (oy + 0.5f) / outH
            val sx = srcX(u, v, rot, cw, ch)
            val sy = srcY(u, v, rot, cw, ch)
            val fx = (sx - 0.5f).coerceIn(0f, (cw - 1).toFloat())
            val fy = (sy - 0.5f).coerceIn(0f, (ch - 1).toFloat())
            val x0 = fx.toInt()
            val y0 = fy.toInt()
            val i = oy * outW + ox
            ix[i] = x0; iy[i] = y0
            wx[i] = fx - x0; wy[i] = fy - y0
        }
        tCw = cw; tCh = ch; tRot = rot
    }

    private fun toGray(img: Image, rotation: Int): FloatArray {
        val crop = img.cropRect
        val cw = crop.width()
        val ch = crop.height()
        if (cw != tCw || ch != tCh || rotation != tRot) buildTables(cw, ch, rotation)

        val plane = img.planes[0]
        val rowStride = plane.rowStride
        val pixStride = plane.pixelStride
        yArr = fillBytes(plane.buffer, yArr)
        val arr = yArr

        val out = FloatArray(n)
        val inv = 1f / 255f
        for (i in 0 until n) {
            val x0 = ix[i]
            val y0 = iy[i]
            val x1 = if (x0 + 1 < cw) x0 + 1 else x0
            val y1 = if (y0 + 1 < ch) y0 + 1 else y0
            val r0 = (crop.top + y0) * rowStride
            val r1 = (crop.top + y1) * rowStride
            val c0 = (crop.left + x0) * pixStride
            val c1 = (crop.left + x1) * pixStride
            val a = (arr[r0 + c0].toInt() and 0xFF).toFloat()
            val b = (arr[r0 + c1].toInt() and 0xFF).toFloat()
            val c = (arr[r1 + c0].toInt() and 0xFF).toFloat()
            val d = (arr[r1 + c1].toInt() and 0xFF).toFloat()
            val fx = wx[i]
            val fy = wy[i]
            val top = a + (b - a) * fx
            val bot = c + (d - c) * fx
            out[i] = (top + (bot - top) * fy) * inv
        }
        return out
    }
}
