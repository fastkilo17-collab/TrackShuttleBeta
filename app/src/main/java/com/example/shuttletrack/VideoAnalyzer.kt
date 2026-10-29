package com.example.shuttletrack

import android.content.Context
import android.graphics.Bitmap
import android.graphics.PointF
import android.media.MediaMetadataRetriever
import android.net.Uri

/** ผลวิเคราะห์: จุดลูกต่อเฟรม (พิกเซลเทียบภาพ inW x inH) */
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
     * อ่านวิดีโอทีละ 3 เฟรม -> TrackNet -> จุดลูก
     * onProgress(done, total) ถูกเรียกจากเธรดที่เรียก analyze (ควรเรียกจากเธรดเบื้องหลัง)
     * isCancelled ใช้หยุดกลางคัน
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
            var i = 0
            while (i + 3 <= total) {
                if (isCancelled()) break
                val frames = (0 until 3).map { j -> frameAt(mmr, i + j, fps, detector.inW, detector.inH) }
                if (frames.any { it == null }) {
                    repeat(3) { points.add(null) }
                } else {
                    points.addAll(detector.detect(frames.map { it!! }))
                }
                i += 3
                onProgress(i, total)
            }
            return AnalysisResult(uri, fps, detector.inW, detector.inH, points)
        } finally {
            mmr.release()
        }
    }

    fun frameAt(mmr: MediaMetadataRetriever, index: Int, fps: Float, w: Int, h: Int): Bitmap? {
        val timeUs = (index * 1_000_000f / fps).toLong()
        return mmr.getScaledFrameAtTime(timeUs, MediaMetadataRetriever.OPTION_CLOSEST, w, h)
    }
}
