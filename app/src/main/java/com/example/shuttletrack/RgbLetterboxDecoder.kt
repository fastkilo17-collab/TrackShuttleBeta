package com.example.shuttletrack

import android.content.Context
import android.graphics.PointF
import android.net.Uri

/** ถอดรหัสวิดีโอต่อเนื่อง -> ภาพ RGB letterbox (ใช้ RgbLetterboxConverter ตัวเดียวกับโหมดกล้องสด) */
class RgbLetterboxDecoder(
    context: Context,
    uri: Uri,
    val outW: Int,
    val outH: Int,
    nchw: Boolean
) {
    private val reader = YuvVideoReader(context, uri)
    private val conv = RgbLetterboxConverter(outW, outH, nchw)
    val fps: Float get() = reader.fps
    val estimatedFrames: Int get() = reader.estimatedFrames

    /** onFrame(index, rgb) rgb ถูกใช้ซ้ำทุกเฟรม; wants=false จะข้ามการแปลงเฟรมนั้น */
    fun decode(
        maxFrames: Int,
        isCancelled: () -> Boolean,
        wants: (Int) -> Boolean = { true },
        onFrame: (Int, FloatArray) -> Unit
    ): Int = reader.read(maxFrames, isCancelled, wants) { i, img, rot -> onFrame(i, conv.convert(img, rot)) }

    fun normalize(x: Float, y: Float): PointF? = conv.normalize(x, y)
}
