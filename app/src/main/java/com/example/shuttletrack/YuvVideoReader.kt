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
 * ถอดรหัสวิดีโอต่อเนื่องด้วย MediaCodec แล้วส่งเฟรม YUV (android.media.Image) พร้อมมุมหมุนให้ผู้ใช้แปลงเอง
 * เร็วกว่า MediaMetadataRetriever.getFrameAtTime ต่อเฟรมหลายเท่า (ไม่ต้อง seek/ถอดรหัสซ้ำ)
 */
class YuvVideoReader(private val context: Context, private val uri: Uri) {
    val fps: Float
    val estimatedFrames: Int

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

    /**
     * onImage(index, image, rotationDegrees) ถูกเรียกต่อเฟรม; image ใช้ได้เฉพาะภายใน callback
     * wants(index) = false จะข้ามเฟรมนั้นโดยไม่เรียก onImage (ยังต้องถอดรหัสเพื่อไปเฟรมถัดไป)
     * คืนจำนวนเฟรมที่ถอดได้
     */
    fun read(
        maxFrames: Int,
        isCancelled: () -> Boolean,
        wants: (Int) -> Boolean,
        onImage: (Int, Image, Int) -> Unit
    ): Int {
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
                        if (wants(count)) {
                            val img = codec.getOutputImage(oi)
                                ?: throw IllegalStateException("เครื่องนี้ไม่ให้อ่านภาพจากตัวถอดรหัสตรงๆ")
                            try {
                                onImage(count, img, rotation)
                            } finally {
                                img.close()
                            }
                        }
                        count++
                    }
                    codec.releaseOutputBuffer(oi, false)
                    if (info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) eos = true
                }
            }
            return count
        } finally {
            try { codec?.stop() } catch (e: Exception) { /* ignore */ }
            codec?.release()
            ex.release()
        }
    }
}

/** แปลงพิกัดผลลัพธ์ (u,v ใน 0..1 ของภาพหลังหมุน) เป็นพิกัดในภาพต้นฉบับก่อนหมุน */
internal fun srcX(u: Float, v: Float, rot: Int, cw: Int, ch: Int): Float = when (rot) {
    90 -> v * cw
    180 -> (1f - u) * cw
    270 -> (1f - v) * cw
    else -> u * cw
}

internal fun srcY(u: Float, v: Float, rot: Int, cw: Int, ch: Int): Float = when (rot) {
    90 -> (1f - u) * ch
    180 -> (1f - v) * ch
    270 -> u * ch
    else -> v * ch
}

/** คัดลอกเนื้อใน ByteBuffer ลงอาเรย์ที่ใช้ซ้ำ (ขยายเมื่อจำเป็น) */
internal fun fillBytes(buf: java.nio.ByteBuffer, dst: ByteArray): ByteArray {
    buf.rewind()
    val rem = buf.remaining()
    val arr = if (dst.size >= rem) dst else ByteArray(rem)
    buf.get(arr, 0, rem)
    return arr
}
