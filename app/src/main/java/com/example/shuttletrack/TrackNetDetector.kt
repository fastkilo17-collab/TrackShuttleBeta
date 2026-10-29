package com.example.shuttletrack

import android.content.Context
import android.graphics.Bitmap
import android.graphics.PointF
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

/**
 * ตัวตรวจจับลูกขนไก่ด้วย TrackNet (V2: รับ 3 เฟรม RGB = 9 แชนเนล -> ฮีตแมป 3 ภาพ)
 *
 * คลาสนี้ไม่รู้จักวิดีโอหรือกล้อง รับแค่ "บิตแมป 3 ใบต่อเนื่องกัน" จึงใช้ซ้ำกับกล้องสดได้ในอนาคต
 * (CameraX ImageAnalysis -> เก็บ 3 เฟรมล่าสุด -> detect())
 */
class TrackNetDetector(context: Context, modelAsset: String = "tracknet.tflite") : AutoCloseable {

    private val interpreter: Interpreter
    val inW: Int
    val inH: Int
    private val inputNhwc: Boolean
    private val outputNhwc: Boolean
    private val frameCount = 3

    var threshold = 0.5f

    init {
        val fd = context.assets.openFd(modelAsset)
        val model = FileInputStream(fd.fileDescriptor).channel
            .map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
        interpreter = Interpreter(model, Interpreter.Options().setNumThreads(4))

        val inShape = interpreter.getInputTensor(0).shape()   // [1,H,W,9] หรือ [1,9,H,W]
        val outShape = interpreter.getOutputTensor(0).shape() // [1,H,W,3] หรือ [1,3,H,W]
        inputNhwc = inShape[3] == frameCount * 3
        outputNhwc = outShape[3] == frameCount
        inH = if (inputNhwc) inShape[1] else inShape[2]
        inW = if (inputNhwc) inShape[2] else inShape[3]
    }

    /** frames: บิตแมป 3 ใบต่อเนื่อง ขนาด inW x inH. คืนตำแหน่งลูก (พิกเซลในภาพ inW x inH) ต่อเฟรม หรือ null ถ้าไม่เจอ */
    fun detect(frames: List<Bitmap>): List<PointF?> {
        require(frames.size == frameCount) { "ต้องใช้ $frameCount เฟรม" }

        val input = ByteBuffer.allocateDirect(4 * frameCount * 3 * inH * inW).order(ByteOrder.nativeOrder())
        val fb = input.asFloatBuffer()
        val plane = inH * inW
        val data = FloatArray(frameCount * 3 * plane)
        val px = IntArray(plane)

        for (f in 0 until frameCount) {
            frames[f].getPixels(px, 0, inW, 0, 0, inW, inH)
            for (i in 0 until plane) {
                val p = px[i]
                val r = ((p shr 16) and 0xFF) / 255f
                val g = ((p shr 8) and 0xFF) / 255f
                val b = (p and 0xFF) / 255f
                if (inputNhwc) {
                    val base = i * frameCount * 3 + f * 3
                    data[base] = r; data[base + 1] = g; data[base + 2] = b
                } else {
                    data[(f * 3) * plane + i] = r
                    data[(f * 3 + 1) * plane + i] = g
                    data[(f * 3 + 2) * plane + i] = b
                }
            }
        }
        fb.put(data)
        input.rewind()

        val output = ByteBuffer.allocateDirect(4 * frameCount * plane).order(ByteOrder.nativeOrder())
        interpreter.run(input, output)
        output.rewind()
        val heat = FloatArray(frameCount * plane)
        output.asFloatBuffer().get(heat)

        return List(frameCount) { k -> findBall(heat, k, plane) }
    }

    private fun heatAt(h: FloatArray, k: Int, x: Int, y: Int, plane: Int): Float =
        if (outputNhwc) h[(y * inW + x) * frameCount + k] else h[k * plane + y * inW + x]

    /** หายอดสูงสุดของฮีตแมป แล้วหาจุดศูนย์กลางถ่วงน้ำหนักรอบยอด */
    private fun findBall(h: FloatArray, k: Int, plane: Int): PointF? {
        var best = 0f; var bx = -1; var by = -1
        for (y in 0 until inH) for (x in 0 until inW) {
            val v = heatAt(h, k, x, y, plane)
            if (v > best) { best = v; bx = x; by = y }
        }
        if (best < threshold || bx < 0) return null

        val r = 6
        var sw = 0f; var sx = 0f; var sy = 0f
        for (y in maxOf(0, by - r)..minOf(inH - 1, by + r))
            for (x in maxOf(0, bx - r)..minOf(inW - 1, bx + r)) {
                val v = heatAt(h, k, x, y, plane)
                if (v >= threshold * 0.5f) { sw += v; sx += v * x; sy += v * y }
            }
        return if (sw > 0f) PointF(sx / sw, sy / sw) else PointF(bx.toFloat(), by.toFloat())
    }

    override fun close() = interpreter.close()
}
