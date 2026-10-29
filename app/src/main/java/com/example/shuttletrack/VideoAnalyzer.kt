package com.example.shuttletrack

import android.content.Context
import android.graphics.Bitmap
import android.graphics.PointF
import android.media.MediaMetadataRetriever
import android.net.Uri

/** ผลวิเคราะห์: จุดลูกต่อเฟรม (พิกเซลเทียบภาพ inW x inH) null = ไม่เจอ */
class AnalysisResult(
    val uri: Uri,
    val fps: Float,
    val inW: Int,
    val inH: Int,
    val points: List<PointF?>
) {
    val frameCount get() = points.size
    val detected get() = points.count { it != null }
}

object VideoAnalyzer {

    /**
     * หน้าต่างเลื่อนทีละเฟรม: เฟรม (i-2, i-1, i) -> ตำแหน่งลูกในเฟรม i
     * สองเฟรมแรกไม่มีผล (null)
     * ควรเรียกจากเธรดเบื้องหลัง
     */
    fun analyze(
        context: Context,
        uri: Uri,
        detector: TrackNetDetector,
        maxFrames: Int = 900,
        onProgress: (Int, Int) -> Unit,
        isCancelled: () -> Boolean = { false }
    ): AnalysisResult {
        val mmr = MediaMetadataRetriever()
        try {
            mmr.setDataSource(context, uri)
            val durationMs = mmr.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLongOrNull() ?: 0L
            val fps = mmr.extractMetadata(MediaMetadataRetriever.METADATA_KEY_CAPTURE_FRAMERATE)
                ?.toFloatOrNull()?.takeIf { it > 1f } ?: 30f

            val total = minOf(maxFrames, (durationMs / 1000f * fps).toInt())
            val points = ArrayList<PointF?>(total)
            val window = ArrayDeque<Bitmap>()

            for (i in 0 until total) {
                if (isCancelled()) break
                val bmp = frameAt(mmr, i, fps, detector.inW, detector.inH)
                if (bmp == null) {
                    window.clear()
                    points.add(null)
                } else {
                    window.addLast(bmp)
                    if (window.size > 3) window.removeFirst().recycle()
                    points.add(if (window.size == 3) detector.detect(window.toList()) else null)
                }
                if (i % 3 == 0) onProgress(i + 1, total)
            }
            window.forEach { it.recycle() }
            onProgress(points.size, total)
            return AnalysisResult(uri, fps, detector.inW, detector.inH, points)
        } finally {
            mmr.release()
        }
    }

    /** ดึงเฟรมที่ index แล้วย่อเป็น w x h (ยืดตามต้นฉบับ TrackNet ที่ resize ตรงๆ) */
    fun frameAt(mmr: MediaMetadataRetriever, index: Int, fps: Float, w: Int, h: Int): Bitmap? {
        val timeUs = (index * 1_000_000f / fps).toLong()
        val src = mmr.getFrameAtTime(timeUs, MediaMetadataRetriever.OPTION_CLOSEST) ?: return null
        if (src.width == w && src.height == h) return src
        val out = Bitmap.createScaledBitmap(src, w, h, true)
        if (out !== src) src.recycle()
        return out
    }
}
