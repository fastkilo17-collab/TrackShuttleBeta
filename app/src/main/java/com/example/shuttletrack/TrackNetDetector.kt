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
 * ตัวตรวจจับลูกขนไก่ด้วย TrackNet (Chang-Chia-Chi TF2)
 *   อินพุต : 3 เฟรมต่อเนื่อง ขาวดำ 0..1 ขนาด inW x inH (512x288)
 *   เอาต์พุต: แผนที่ความน่าจะเป็นของลูก "ณ เฟรมสุดท้ายของ 3 เฟรม"
 *
 * ไม่รู้จักวิดีโอหรือกล้อง รับแค่บิตแมป 3 ใบ จึงต่อกับ CameraX ในอนาคตได้
 */
class TrackNetDetector(context: Context, modelAsset: String = "tracknet.tflite") : AutoCloseable {

    private val interpreter: Interpreter
    val inW: Int
    val inH: Int
    private val nchw: Boolean
    private val frameCount = 3

    private val input: ByteBuffer
    private val output: ByteBuffer
    private val inData: FloatArray
    private val heat: FloatArray
    private val px: IntArray
    private val visited: BooleanArray
    private val stack: IntArray

    var threshold = 0.5f

    init {
        val fd = context.assets.openFd(modelAsset)
        val model = FileInputStream(fd.fileDescriptor).channel
            .map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
        interpreter = Interpreter(model, Interpreter.Options().setNumThreads(4))

        val s = interpreter.getInputTensor(0).shape() // [1,3,H,W] หรือ [1,H,W,3]
        nchw = s[1] == frameCount
        inH = if (nchw) s[2] else s[1]
        inW = if (nchw) s[3] else s[2]

        val plane = inH * inW
        input = ByteBuffer.allocateDirect(4 * frameCount * plane).order(ByteOrder.nativeOrder())
        output = ByteBuffer.allocateDirect(4 * plane).order(ByteOrder.nativeOrder())
        inData = FloatArray(frameCount * plane)
        heat = FloatArray(plane)
        px = IntArray(plane)
        visited = BooleanArray(plane)
        stack = IntArray(plane)
    }

    /**
     * frames: บิตแมป 3 ใบต่อเนื่อง ขนาด inW x inH (เก่า -> ใหม่)
     * คืนตำแหน่งลูกในเฟรมที่ 3 (พิกเซลของภาพ inW x inH) หรือ null ถ้าไม่เจอ
     */
    fun detect(frames: List<Bitmap>): PointF? {
        require(frames.size == frameCount) { "ต้องใช้ $frameCount เฟรม" }
        val plane = inH * inW

        for (f in 0 until frameCount) {
            frames[f].getPixels(px, 0, inW, 0, 0, inW, inH)
            for (i in 0 until plane) {
                val p = px[i]
                val r = (p shr 16) and 0xFF
                val g = (p shr 8) and 0xFF
                val b = p and 0xFF
                val gray = (0.299f * r + 0.587f * g + 0.114f * b) / 255f
                if (nchw) inData[f * plane + i] = gray else inData[i * frameCount + f] = gray
            }
        }
        input.rewind()
        input.asFloatBuffer().put(inData)
        input.rewind()
        output.rewind()
        interpreter.run(input, output)
        output.rewind()
        output.asFloatBuffer().get(heat)

        return largestBlobCenter(heat)
    }

    /** ตามต้นฉบับ: threshold -> หา blob ที่กรอบใหญ่สุด -> จุดกึ่งกลางกรอบ */
    private fun largestBlobCenter(h: FloatArray): PointF? {
        java.util.Arrays.fill(visited, false)
        var bestArea = 0
        var bx = 0f
        var by = 0f
        for (start in h.indices) {
            if (visited[start] || h[start] <= threshold) continue
            var sp = 0
            stack[sp++] = start
            visited[start] = true
            var minX = Int.MAX_VALUE; var maxX = -1
            var minY = Int.MAX_VALUE; var maxY = -1
            while (sp > 0) {
                val cur = stack[--sp]
                val x = cur % inW
                val y = cur / inW
                if (x < minX) minX = x
                if (x > maxX) maxX = x
                if (y < minY) minY = y
                if (y > maxY) maxY = y
                if (x > 0) push(cur - 1, h, sp).also { sp = it }
                if (x < inW - 1) push(cur + 1, h, sp).also { sp = it }
                if (y > 0) push(cur - inW, h, sp).also { sp = it }
                if (y < inH - 1) push(cur + inW, h, sp).also { sp = it }
            }
            val w = maxX - minX + 1
            val hh = maxY - minY + 1
            val area = w * hh
            if (area > bestArea) {
                bestArea = area
                bx = minX + w / 2f
                by = minY + hh / 2f
            }
        }
        return if (bestArea > 0) PointF(bx, by) else null
    }

    private fun push(idx: Int, h: FloatArray, sp: Int): Int {
        if (visited[idx] || h[idx] <= threshold) return sp
        visited[idx] = true
        stack[sp] = idx
        return sp + 1
    }

    override fun close() = interpreter.close()
}
