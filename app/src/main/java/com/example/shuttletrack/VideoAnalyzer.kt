package com.example.shuttletrack

import android.content.Context
import android.graphics.Bitmap
import android.graphics.PointF
import android.media.MediaMetadataRetriever
import android.net.Uri

/** ผลวิเคราะห์: จุดลูกต่อเฟรม (พิกเซลเทียบภาพ inW x inH) null = ไม่เจอ/ไม่ได้ตรวจ */
class AnalysisResult(
    val uri: Uri,
    val fps: Float,
    val inW: Int,
    val inH: Int,
    val points: List<PointF?>,
    val attempted: Int,        // จำนวนครั้งที่รันโมเดลจริง
    val msDecodePerFrame: Float,
    val msInferPerRun: Float,
    val gpu: Boolean
) {
    val frameCount get() = points.size
    val detected get() = points.count { it != null }
}

object VideoAnalyzer {

    /**
     * หน้าต่าง 3 เฟรมเลื่อนไปตามวิดีโอ: เฟรม (i-2, i-1, i) -> ตำแหน่งลูกในเฟรม i
     * detectEvery = 1 ตรวจทุกเฟรม, 3 ตรวจเฟรมเว้นเฟรม (เร็วขึ้น ~3 เท่า แต่จุดห่างขึ้น)
     * ควรเรียกจากเธรดเบื้องหลัง
     */
    fun analyze(
        context: Context,
        uri: Uri,
        detector: TrackNetDetector,
        maxFrames: Int = 900,
        detectEvery: Int = 1,
        onProgress: (Int, Int) -> Unit,
        isCancelled: () -> Boolean = { false }
    ): AnalysisResult {
        val dec = GrayFrameDecoder(context, uri, detector.inW, detector.inH)
        val total = minOf(maxFrames, dec.estimatedFrames).coerceAtLeast(1)
        val window = ArrayDeque<FloatArray>()
        val points = ArrayList<PointF?>()
        var inferNs = 0L
        var attempted = 0

        val t0 = System.nanoTime()
        dec.decode(maxFrames, isCancelled) { i, gray ->
            window.addLast(gray)
            if (window.size > 3) window.removeFirst()
            var p: PointF? = null
            if (window.size == 3 && (i - 2) % detectEvery == 0) {
                val t = System.nanoTime()
                p = detector.detect(window[0], window[1], window[2])
                inferNs += System.nanoTime() - t
                attempted++
            }
            points.add(p)
            if (i % 3 == 0) onProgress(i + 1, total)
        }
        val totalNs = System.nanoTime() - t0
        onProgress(points.size, maxOf(total, points.size))

        val n = maxOf(1, points.size)
        val msDecode = (totalNs - inferNs) / 1e6f / n
        val msInfer = if (attempted > 0) inferNs / 1e6f / attempted else 0f
        return AnalysisResult(uri, dec.fps, detector.inW, detector.inH, points, attempted, msDecode, msInfer, detector.gpuActive)
    }

    /** ดึงเฟรมเดียวมาแสดงตอนเลื่อนดู (ช้าได้ ไม่กระทบการวิเคราะห์) ย่อเป็น w x h */
    fun frameAt(mmr: MediaMetadataRetriever, index: Int, fps: Float, w: Int, h: Int): Bitmap? {
        val timeUs = (index * 1_000_000f / fps).toLong()
        val src = mmr.getFrameAtTime(timeUs, MediaMetadataRetriever.OPTION_CLOSEST) ?: return null
        if (src.width == w && src.height == h) return src
        val out = Bitmap.createScaledBitmap(src, w, h, true)
        if (out !== src) src.recycle()
        return out
    }
}
