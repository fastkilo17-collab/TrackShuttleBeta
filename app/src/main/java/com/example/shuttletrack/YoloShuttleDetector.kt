package com.example.shuttletrack

import android.content.Context
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

/** ผลตรวจ: กึ่งกลางและขนาดกล่อง เป็นพิกเซลของภาพอินพุตโมเดล (inW x inH) */
class YoloDet(val cx: Float, val cy: Float, val w: Float, val h: Float, val conf: Float)

/**
 * ตัวตรวจจับลูกขนไก่ YOLOv8n ที่เทรนเอง (ส่งออกจาก ultralytics: อินพุต [1,3,384,640] เอาต์พุต [1,5,5040])
 * ดูภาพสีเดียวทีละภาพ จึงไม่ต้องซ้อน 3 เฟรม และเบากว่า TrackNet
 */
class YoloShuttleDetector(
    context: Context,
    modelAsset: String = "shuttle_yolo.tflite",
    useGpu: Boolean = false
) : AutoCloseable {

    private var interpreter: Interpreter
    private var gpu: GpuDelegate? = null
    val gpuActive: Boolean get() = gpu != null

    val inW: Int
    val inH: Int
    val nchw: Boolean
    private val boxMajor: Boolean   // เอาต์พุต [1,N,5] (true) หรือ [1,5,N] (false)
    private val nAnchors: Int

    private val input: ByteBuffer
    private val output: ByteBuffer
    private val outArr: FloatArray

    /** ค่าความมั่นใจขั้นต่ำ */
    var conf = 0.25f

    init {
        val fd = context.assets.openFd(modelAsset)
        val model: MappedByteBuffer = FileInputStream(fd.fileDescriptor).channel
            .map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)

        val cpu = { Interpreter(model, Interpreter.Options().setNumThreads(4)) }
        var made: Interpreter? = null
        if (useGpu) {
            try {
                val d = GpuDelegate(GpuDelegate.Options().setPrecisionLossAllowed(true))
                try {
                    val ip = Interpreter(model, Interpreter.Options().addDelegate(d))
                    warmUp(ip)  // ถ้า GPU ใช้ไม่ได้จริงจะล้มตรงนี้ แล้วถอยไป CPU
                    made = ip
                    gpu = d
                } catch (t: Throwable) {
                    d.close()
                }
            } catch (t: Throwable) {
                // เครื่องไม่รองรับ GPU -> CPU
            }
        }
        interpreter = made ?: cpu()

        val s = interpreter.getInputTensor(0).shape()      // [1,3,H,W] หรือ [1,H,W,3]
        nchw = s[1] == 3
        inH = if (nchw) s[2] else s[1]
        inW = if (nchw) s[3] else s[2]

        val o = interpreter.getOutputTensor(0).shape()     // [1,5,N] หรือ [1,N,5]
        boxMajor = o[1] != 5 && o[2] == 5
        nAnchors = if (boxMajor) o[1] else o[2]

        input = ByteBuffer.allocateDirect(4 * 3 * inW * inH).order(ByteOrder.nativeOrder())
        output = ByteBuffer.allocateDirect(4 * 5 * nAnchors).order(ByteOrder.nativeOrder())
        outArr = FloatArray(5 * nAnchors)
    }

    private fun warmUp(ip: Interpreter) {
        val s = ip.getInputTensor(0).shape()
        val o = ip.getOutputTensor(0).shape()
        val inB = ByteBuffer.allocateDirect(4 * s.fold(1) { a, b -> a * b }).order(ByteOrder.nativeOrder())
        val outB = ByteBuffer.allocateDirect(4 * o.fold(1) { a, b -> a * b }).order(ByteOrder.nativeOrder())
        ip.run(inB, outB)
    }

    /**
     * rgb: ภาพ letterbox ค่า 0..1 เรียงตามรูปแบบอินพุตของโมเดล (RgbLetterboxDecoder สร้างให้)
     * คืนกล่องที่มั่นใจสุดที่ผ่านเกณฑ์ หรือ null
     */
    fun detect(rgb: FloatArray): YoloDet? {
        input.rewind()
        input.asFloatBuffer().put(rgb)
        input.rewind()
        output.rewind()
        interpreter.run(input, output)
        output.rewind()
        output.asFloatBuffer().get(outArr)

        var best = -1
        var bc = conf
        for (i in 0 until nAnchors) {
            val c = if (boxMajor) outArr[i * 5 + 4] else outArr[4 * nAnchors + i]
            if (c > bc) { bc = c; best = i }
        }
        if (best < 0) return null

        var cx: Float; var cy: Float; var w: Float; var h: Float
        if (boxMajor) {
            cx = outArr[best * 5]; cy = outArr[best * 5 + 1]; w = outArr[best * 5 + 2]; h = outArr[best * 5 + 3]
        } else {
            cx = outArr[best]; cy = outArr[nAnchors + best]
            w = outArr[2 * nAnchors + best]; h = outArr[3 * nAnchors + best]
        }
        // ultralytics ส่งออก ONNX เป็นพิกเซล แต่ TFLite บางรุ่นเป็นสัดส่วน 0..1: ถ้าทุกค่าเล็กกว่า ~1.5 ถือเป็นสัดส่วน
        if (cx <= 1.5f && cy <= 1.5f && w <= 1.5f && h <= 1.5f) {
            cx *= inW; w *= inW; cy *= inH; h *= inH
        }
        return YoloDet(cx, cy, w, h, bc)
    }

    override fun close() {
        interpreter.close()
        gpu?.close()
    }
}
