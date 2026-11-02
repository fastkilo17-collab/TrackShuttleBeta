package com.example.shuttletrack

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.PointF
import android.os.Bundle
import android.util.Size
import android.view.Gravity
import android.view.View
import android.view.ViewGroup.LayoutParams.MATCH_PARENT
import android.view.ViewGroup.LayoutParams.WRAP_CONTENT
import android.widget.CheckBox
import android.widget.FrameLayout
import android.widget.LinearLayout
import android.widget.SeekBar
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import java.util.concurrent.Executors

/** วาดวิถีลูกทับพรีวิวกล้อง (พรีวิวใช้ FIT_CENTER จึงคำนวณพื้นที่ภาพจากสัดส่วนเฟรม) */
class LiveOverlay(context: android.content.Context) : View(context) {
    @Volatile var trail: List<PointF> = emptyList()   // พิกัด 0..1 ของเฟรม เก่า -> ใหม่
    @Volatile var aspect = 16f / 9f
    private val line = Paint(Paint.ANTI_ALIAS_FLAG).apply { strokeWidth = 5f; style = Paint.Style.STROKE }
    private val ring = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.RED; strokeWidth = 6f; style = Paint.Style.STROKE
    }

    override fun onDraw(c: Canvas) {
        val t = trail
        if (t.isEmpty()) return
        var w = width.toFloat()
        var h = w / aspect
        if (h > height) { h = height.toFloat(); w = h * aspect }
        val ox = (width - w) / 2f
        val oy = (height - h) / 2f
        for (i in 1 until t.size) {
            line.color = Color.argb(60 + 195 * i / t.size, 255, 200, 0)
            c.drawLine(ox + t[i - 1].x * w, oy + t[i - 1].y * h, ox + t[i].x * w, oy + t[i].y * h, line)
        }
        val p = t.last()
        c.drawCircle(ox + p.x * w, oy + p.y * h, 22f, ring)
    }
}

@androidx.camera.core.ExperimentalGetImage
class LiveActivity : ComponentActivity() {

    private val analysisExec = Executors.newSingleThreadExecutor()
    private lateinit var previewView: PreviewView
    private lateinit var overlay: LiveOverlay
    private lateinit var stats: TextView
    private lateinit var gpuBox: CheckBox

    @Volatile private var conf = 0.25f
    @Volatile private var wantGpu = true
    @Volatile private var gpuChanged = false

    // เฉพาะเธรดวิเคราะห์
    private var det: YoloShuttleDetector? = null
    private var conv: RgbLetterboxConverter? = null
    private val trail = ArrayDeque<Pair<Long, PointF>>()
    private var emaConv = 0f
    private var emaInfer = 0f
    private var lastFrameNs = 0L
    private var emaFps = 0f
    private var lastUiNs = 0L
    private var hits = 0
    private var total = 0

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val root = FrameLayout(this).apply { setBackgroundColor(Color.BLACK) }
        previewView = PreviewView(this).apply { scaleType = PreviewView.ScaleType.FIT_CENTER }
        overlay = LiveOverlay(this)
        root.addView(previewView, FrameLayout.LayoutParams(MATCH_PARENT, MATCH_PARENT))
        root.addView(overlay, FrameLayout.LayoutParams(MATCH_PARENT, MATCH_PARENT))

        val panel = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(Color.argb(150, 0, 0, 0))
            setPadding(24, 12, 24, 12)
        }
        stats = TextView(this).apply { setTextColor(Color.WHITE); text = "กำลังเริ่มกล้อง..." }
        gpuBox = CheckBox(this).apply {
            text = "GPU"; setTextColor(Color.WHITE); isChecked = true
            setOnCheckedChangeListener { _, on -> wantGpu = on; gpuChanged = true }
        }
        val confLabel = TextView(this).apply { setTextColor(Color.WHITE); text = "ความมั่นใจขั้นต่ำ 0.25" }
        val confBar = SeekBar(this).apply {
            max = 90; progress = 20      // 0.05 .. 0.95
            setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
                override fun onProgressChanged(s: SeekBar?, p: Int, fromUser: Boolean) {
                    conf = 0.05f + p / 100f
                    confLabel.text = "ความมั่นใจขั้นต่ำ %.2f".format(java.util.Locale.US, conf)
                }
                override fun onStartTrackingTouch(s: SeekBar?) {}
                override fun onStopTrackingTouch(s: SeekBar?) {}
            })
        }
        panel.addView(stats, LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT))
        panel.addView(gpuBox, LinearLayout.LayoutParams(WRAP_CONTENT, WRAP_CONTENT))
        panel.addView(confLabel, LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT))
        panel.addView(confBar, LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT))
        root.addView(panel, FrameLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT, Gravity.BOTTOM))
        setContentView(root)

        if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) startCamera()
        else requestPermissions(arrayOf(Manifest.permission.CAMERA), REQ_CAM)
    }

    override fun onRequestPermissionsResult(code: Int, perms: Array<out String>, res: IntArray) {
        super.onRequestPermissionsResult(code, perms, res)
        if (code == REQ_CAM && res.firstOrNull() == PackageManager.PERMISSION_GRANTED) startCamera()
        else stats.text = "ต้องอนุญาตกล้องก่อนจึงจะใช้โหมดสดได้"
    }

    private fun startCamera() {
        val fut = ProcessCameraProvider.getInstance(this)
        fut.addListener({
            val provider = fut.get()
            val preview = Preview.Builder().setTargetResolution(Size(1280, 720)).build()
                .also { it.setSurfaceProvider(previewView.surfaceProvider) }
            val analysis = ImageAnalysis.Builder()
                .setTargetResolution(Size(1280, 720))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
                .build()
            analysis.setAnalyzer(analysisExec) { proxy -> analyze(proxy) }
            provider.unbindAll()
            provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis)
        }, ContextCompat.getMainExecutor(this))
    }

    private fun analyze(proxy: ImageProxy) {
        try {
            val img = proxy.image ?: return
            if (det == null || gpuChanged) {
                gpuChanged = false
                det?.close()
                det = YoloShuttleDetector(this, useGpu = wantGpu)
                conv = RgbLetterboxConverter(det!!.inW, det!!.inH, det!!.nchw)
            }
            val d = det!!
            val c = conv!!
            d.conf = conf

            val t0 = System.nanoTime()
            val rgb = c.convert(img, proxy.imageInfo.rotationDegrees)
            val t1 = System.nanoTime()
            val r = d.detect(rgb)
            val t2 = System.nanoTime()

            emaConv = ema(emaConv, (t1 - t0) / 1e6f)
            emaInfer = ema(emaInfer, (t2 - t1) / 1e6f)
            if (lastFrameNs != 0L) emaFps = ema(emaFps, 1e9f / (t2 - lastFrameNs).coerceAtLeast(1))
            lastFrameNs = t2
            total++

            val now = System.nanoTime()
            if (r != null) {
                c.normalize(r.cx, r.cy)?.let { trail.addLast(now to it); hits++ }
            }
            while (trail.isNotEmpty() && now - trail.first().first > 600_000_000L) trail.removeFirst()
            overlay.aspect = c.contentW / c.contentH
            overlay.trail = trail.map { it.second }
            overlay.postInvalidate()

            if (now - lastUiNs > 250_000_000L) {
                lastUiNs = now
                val txt = "%.0f fps · แปลงภาพ %.0f ms + โมเดล %.0f ms (%s)\nเจอลูก %d/%d เฟรม%s".format(
                    emaFps, emaConv, emaInfer, if (d.gpuActive) "GPU" else "CPU",
                    hits, total, if (r != null) "  ✓ %.2f".format(r.conf) else ""
                )
                runOnUiThread { stats.text = txt }
            }
        } catch (e: Exception) {
            runOnUiThread { stats.text = "ผิดพลาด: ${e.message}" }
        } finally {
            proxy.close()
        }
    }

    private fun ema(old: Float, v: Float) = if (old == 0f) v else old * 0.9f + v * 0.1f

    override fun onDestroy() {
        analysisExec.execute { det?.close(); det = null }
        analysisExec.shutdown()
        super.onDestroy()
    }

    companion object { private const val REQ_CAM = 2 }
}
