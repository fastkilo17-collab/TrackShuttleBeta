package com.example.shuttletrack

import android.content.Context
import android.media.Image
import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.os.Build

/**
 * ถอดรหัสวิดีโอต่อเนื่องด้วย MediaCodec แล้วเอาเฉพาะแชนเนลความสว่าง (Y) ย่อเป็น outW x outH ค่า 0..1
 * เร็วกว่า MediaMetadataRetriever.getFrameAtTime ต่อเฟรมหลายเท่า (ไม่ต้อง seek/ถอดรหัสซ้ำ และไม่ผ่าน Bitmap/RGB)
 * รองรับวิดีโอที่หมุน (rotation metadata) โดยสุ่มตัวอย่างในพิกัดหลังหมุน
 */
class GrayFrameDecoder(
    private val context: Context,
    private val uri: Uri,
    private val outW: Int,
    private val outH: Int
) {
    val fps: Float
    val estimatedFrames: Int

    private var tCw = -1
    private var tCh = -1
    private var tRot = -1
    private val n = outW * outH
    private val ix = IntArray(n)
    private val iy = IntArray(n)
    private val wx = FloatArray(n)
    private val wy = FloatArray(n)

    init {
        var f = 0f
        var total = 0
        val mmr = MediaMetadataRetriever()
        try {
            mmr.setDataSource(context, uri)
            val durMs = mmr.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLongOrNull() ?: 0L
            if (Build.VERSION.SDK_INT >= 28) {
                val fc = mmr.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_FRAME_COUNT)?.toIntOrNull() ?: 0
                if (fc > 0 && durMs > 0) f = fc * 1000f / durMs
            }
            if (f < 1f) {
                f = mmr.extractMetadata(MediaMetadataRetriever.METADATA_KEY_CAPTURE_FRAMERATE)
                    ?.toFloatOrNull() ?: 0f
            }
            if (f < 1f) f = 30f
            total = (durMs / 1000f * f).toInt()
        } finally {
            mmr.release()
        }
        fps = f
        estimatedFrames = total
    }

    /** คืนจำนวนเฟรมที่ถอดได้ onFrame(index, gray[outW*outH]) */
    fun decode(maxFrames: Int, isCancelled: () -> Boolean, onFrame: (Int, FloatArray) -> Unit): Int {
        val ex = MediaExtractor()
        var codec: MediaCodec? = null
        try {
            ex.setDataSource(context, uri, null)
            var track = -1
            for (i in 0 until ex.trackCount) {
                val mime = ex.getTrackFormat(i).getString(MediaFormat.KEY_MIME) ?: continue
                if (mime.startsWith("video/")) { track = i; break }
            }
            require(track >= 0) { "ไม่พบแทร็กวิดีโอ" }
            ex.selectTrack(track)
            val fmt = ex.getTrackFormat(track)
            val rotation = if (fmt.containsKey(MediaFormat.KEY_ROTATION)) fmt.getInteger(MediaFormat.KEY_ROTATION) else 0

            codec = MediaCodec.createDecoderByType(fmt.getString(MediaFormat.KEY_MIME)!!)
            codec.configure(fmt, null, null, 0)
            codec.start()

            val info = MediaCodec.BufferInfo()
            var inputDone = false
            var eos = false
            var count = 0
            while (!eos && count < maxFrames && !isCancelled()) {
                if (!inputDone) {
                    val ii = codec.dequeueInputBuffer(5_000)
                    if (ii >= 0) {
                        val buf = codec.getInputBuffer(ii)!!
                        val sz = ex.readSampleData(buf, 0)
                        if (sz < 0) {
                            codec.queueInputBuffer(ii, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                            inputDone = true
                        } else {
                            codec.queueInputBuffer(ii, 0, sz, ex.sampleTime, 0)
                            ex.advance()
                        }
                    }
                }
                val oi = codec.dequeueOutputBuffer(info, 5_000)
                if (oi >= 0) {
                    if (info.size > 0) {
                        val img = codec.getOutputImage(oi)
                            ?: throw IllegalStateException("เครื่องนี้ไม่ให้อ่านภาพจากตัวถอดรหัสตรงๆ")
                        try {
                            onFrame(count, toGray(img, rotation))
                            count++
                        } finally {
                            img.close()
                        }
                    }
                    codec.releaseOutputBuffer(oi, false)
                    if (info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) eos = true
                }
            }
            return count
        } finally {
            try { codec?.stop() } catch (_: Exception) {}
            codec?.release()
            ex.release()
        }
    }

    private fun buildTables(cw: Int, ch: Int, rot: Int) {
        for (oy in 0 until outH) for (ox in 0 until outW) {
            val u = (ox + 0.5f) / outW
            val v = (oy + 0.5f) / outH
            val sx: Float
            val sy: Float
            when (rot) {
                90 -> { sx = v * cw; sy = (1f - u) * ch }
                180 -> { sx = (1f - u) * cw; sy = (1f - v) * ch }
                270 -> { sx = (1f - v) * cw; sy = u * ch }
                else -> { sx = u * cw; sy = v * ch }
            }
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
        val buf = plane.buffer
        val arr = ByteArray(buf.remaining())
        buf.get(arr)

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
