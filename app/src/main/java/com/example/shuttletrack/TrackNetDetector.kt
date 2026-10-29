package com.example.shuttletrack

import android.content.Context
import android.graphics.PointF
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

/**
 * ตัวตรวจจับลูกขนไก่ด้วย TrackNet (Chang-Chia-Chi TF2)
 *   อินพุต : 3 เฟรมต่อเนื่อง ขาวดำ 0..1 ขนาด inW x inH (512x288)
 *   เอาต์พุต: แผนที่ความน่าจะเป็นของลูก "ณ เฟรมสุดท้ายของ 3 เฟรม"
 *
 * รับภาพขาวดำเป็น FloatArray ตรงๆ (ไม่ผูกกับวิดีโอ/Bitmap) จึงต่อกับ CameraX (แชนเนล Y) ได้ในอนาคต
 * ต้องสร้างและเรียก detect() จากเธรดเดียวกัน (GPU delegate ผูกกับเธรด)
 */
class TrackNetDetector(
    context: Context,
    modelAsset: String = "tracknet.tflite",
    useGpu: Boolean = false
) : AutoCloseable {

    private val interpreter: Interpreter
    private var gpu: GpuDelegate? = null
    val gpuActive: Boolean get() = gpu != null

    val inW: Int
    val inH: Int
    private val nchw: Boolean
    private val frameCount = 3

    private val input: ByteBuffer
    private val output: ByteBuffer
    private val heat: FloatArray
    private val visited: BooleanArray
    private val stack: IntArray

    var threshold = 0.5f

    init {
        val fd = context.assets.openFd(modelAsset)
        val model = FileInputStream(fd.fileDescriptor).channel
            .map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)

        var created: Interpreter? = null
        if (useGpu) {
            try {
                val d = GpuDelegate(GpuDelegate.Options().setPrecisionLossAllowed(true))
                try {
                    created = Interpreter(model, Interpreter.Options().addDelegate(d))
                    gpu = d
                } catch (t: Throwable) {
                    d.close()
                }
            } catch (_: Throwable) {
                // เครื่องไม่รองรับ GPU -> ใช้ CPU
            }
        }
        interpreter = created ?: Interpreter(model, Interpreter.Options().setNumThreads(4))

        val s = interpreter.getInputTensor(0).shape() // [1,3,H,W] หรือ [1,H,W,3]
        nchw = s[1] == frameCount
        inH = if (nchw) s[2] else s[1]
        inW = if (nchw) s[3] else s[2]

        val plane = inH * inW
        input = ByteBuffer.allocateDirect(4 * frameCount * plane).order(ByteOrder.nativeOrder())
        output = ByteBuffer.allocateDirect(4 * plane).order(ByteOrder.nativeOrder())
        heat = FloatArray(plane)
        visited = BooleanArray(plane)
        stack = IntArray(plane)
    }

    /** g0,g1,g2: ภาพขาวดำ inW*inH ค่า 0..1 เก่า -> ใหม่. คืนตำแหน่งลูกในเฟรมที่ 3 หรือ null */
    fun detect(g0: FloatArray, g1: FloatArray, g2: FloatArray): PointF? {
        val plane = inH * inW
        input.rewind()
        val fb = input.asFloatBuffer()
        if (nchw) {
            fb.put(g0); fb.put(g1); fb.put(g2)
        } else {
            val tmp = FloatArray(frameCount * plane)
            for (i in 0 until plane) {
                tmp[i * 3] = g0[i]; tmp[i * 3 + 1] = g1[i]; tmp[i * 3 + 2] = g2[i]
            }
            fb.put(tmp)
        }
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
                if (x > 0) sp = push(cur - 1, h, sp)
                if (x < inW - 1) sp = push(cur + 1, h, sp)
                if (y > 0) sp = push(cur - inW, h, sp)
                if (y < inH - 1) sp = push(cur + inW, h, sp)
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

    override fun close() {
        interpreter.close()
        gpu?.close()
    }
}
