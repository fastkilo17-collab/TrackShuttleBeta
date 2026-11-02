package com.example.shuttletrack

import android.app.Activity
import android.content.ContentValues
import android.content.Intent
import android.graphics.Color
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.os.Handler
import android.os.Looper
import android.provider.MediaStore
import android.view.Gravity
import android.view.ViewGroup.LayoutParams.MATCH_PARENT
import android.view.ViewGroup.LayoutParams.WRAP_CONTENT
import android.widget.Button
import android.widget.CheckBox
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.SeekBar
import android.widget.TextView
import android.widget.Toast
import java.io.File
import java.util.Locale
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

class MainActivity : Activity() {

    private val ui = Handler(Looper.getMainLooper())
    private val worker = Executors.newSingleThreadExecutor()
    private val cancel = AtomicBoolean(false)

    private lateinit var pickBtn: Button
    private lateinit var exportBtn: Button
    private lateinit var liveBtn: Button
    private lateinit var yoloBox: CheckBox
    private lateinit var gpuBox: CheckBox
    private lateinit var fastBox: CheckBox
    private lateinit var status: TextView
    private lateinit var progress: ProgressBar
    private lateinit var seek: SeekBar
    private lateinit var view: ResultView

    private var result: AnalysisResult? = null
    private var retriever: MediaMetadataRetriever? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(24, 24, 24, 24)
            setBackgroundColor(Color.WHITE)
        }
        pickBtn = Button(this).apply { text = "เลือกวิดีโอเพื่อวิเคราะห์" }
        yoloBox = CheckBox(this).apply { text = "ใช้โมเดล YOLO ที่เทรนเอง (เบา/เร็ว)"; isChecked = true }
        gpuBox = CheckBox(this).apply { text = "ใช้ GPU (เร็วกว่า ถ้าเครื่องรองรับ)"; isChecked = true }
        fastBox = CheckBox(this).apply { text = "โหมดเร็ว: ตรวจทุก 3 เฟรม" }
        status = TextView(this).apply { text = "ยังไม่ได้เลือกวิดีโอ"; setPadding(0, 16, 0, 8) }
        progress = ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal).apply { max = 100 }
        view = ResultView(this)
        seek = SeekBar(this).apply { isEnabled = false }
        exportBtn = Button(this).apply { text = "ส่งออก CSV"; isEnabled = false }
        liveBtn = Button(this).apply { text = "โหมดกล้องสด (ทดลอง)" }

        root.addView(pickBtn, LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT))
        root.addView(liveBtn, LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT))
        root.addView(yoloBox, LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT))
        root.addView(gpuBox, LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT))
        root.addView(fastBox, LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT))
        root.addView(status, LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT))
        root.addView(progress, LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT))
        root.addView(view, LinearLayout.LayoutParams(MATCH_PARENT, 0, 1f))
        root.addView(seek, LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT).apply { gravity = Gravity.CENTER })
        root.addView(exportBtn, LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT))
        setContentView(root)

        pickBtn.setOnClickListener {
            startActivityForResult(
                Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
                    addCategory(Intent.CATEGORY_OPENABLE)
                    type = "video/*"
                }, REQ_VIDEO
            )
        }
        exportBtn.setOnClickListener { exportCsv() }
        liveBtn.setOnClickListener { startActivity(Intent(this, LiveActivity::class.java)) }
        seek.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(s: SeekBar?, p: Int, fromUser: Boolean) {
                if (fromUser) showFrame(p)
            }
            override fun onStartTrackingTouch(s: SeekBar?) {}
            override fun onStopTrackingTouch(s: SeekBar?) {}
        })
    }

    @Deprecated("Deprecated in Java")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == REQ_VIDEO && resultCode == RESULT_OK) data?.data?.let { startAnalysis(it) }
    }

    private fun startAnalysis(uri: Uri) {
        pickBtn.isEnabled = false
        exportBtn.isEnabled = false
        seek.isEnabled = false
        progress.progress = 0
        status.text = "กำลังโหลดโมเดล..."
        cancel.set(false)
        val wantGpu = gpuBox.isChecked
        val wantYolo = yoloBox.isChecked
        val every = if (fastBox.isChecked) 3 else 1

        worker.execute {
            try {
                val onProgress: (Int, Int) -> Unit = { done, total ->
                    ui.post {
                        progress.progress = if (total > 0) done * 100 / total else 0
                        status.text = "วิเคราะห์แล้ว $done / $total เฟรม"
                    }
                }
                val res = if (wantYolo) {
                    YoloShuttleDetector(this, useGpu = wantGpu).use { det ->
                        VideoAnalyzer.analyzeYolo(
                            this, uri, det, detectEvery = every,
                            onProgress = onProgress, isCancelled = { cancel.get() }
                        )
                    }
                } else {
                    TrackNetDetector(this, useGpu = wantGpu).use { det ->
                        VideoAnalyzer.analyze(
                            this, uri, det, detectEvery = every,
                            onProgress = onProgress, isCancelled = { cancel.get() }
                        )
                    }
                }
                ui.post { onAnalysisDone(res) }
            } catch (e: java.io.FileNotFoundException) {
                fail("ไม่พบไฟล์โมเดลในแอป (${e.message}) — ดู README")
            } catch (e: Exception) {
                fail("เกิดข้อผิดพลาด: ${e.message}")
            }
        }
    }

    private fun fail(msg: String) = ui.post {
        status.text = msg
        pickBtn.isEnabled = true
    }

    private fun onAnalysisDone(res: AnalysisResult) {
        result = res
        retriever?.release()
        retriever = MediaMetadataRetriever().apply { setDataSource(this@MainActivity, res.uri) }
        view.points = res.points
        view.srcW = res.inW
        view.srcH = res.inH
        val pct = if (res.attempted > 0) res.detected * 100 / res.attempted else 0
        status.text = "เสร็จ [${res.model}]: เจอลูก ${res.detected}/${res.attempted} ครั้งที่ตรวจ ($pct%)\n" +
            "ถอดรหัส %.1f ms/เฟรม · โมเดล %.0f ms/ครั้ง (%s)".format(
                res.msDecodePerFrame, res.msInferPerRun, if (res.gpu) "GPU" else "CPU")
        seek.max = maxOf(0, res.frameCount - 1)
        seek.progress = 0
        seek.isEnabled = res.frameCount > 0
        exportBtn.isEnabled = res.frameCount > 0
        pickBtn.isEnabled = true
        showFrame(0)
    }

    private fun showFrame(index: Int) {
        val res = result ?: return
        val mmr = retriever ?: return
        view.frame = VideoAnalyzer.frameAt(mmr, index, res.fps, res.inW, res.inH)
        view.currentIndex = index
        view.invalidate()
    }

    private fun exportCsv() {
        val res = result ?: return
        val sb = StringBuilder("frame,time_s,visible,x_norm,y_norm\n")
        res.points.forEachIndexed { i, p ->
            val t = i / res.fps
            if (p == null) sb.append(String.format(Locale.US, "%d,%.3f,0,,\n", i, t))
            else sb.append(String.format(Locale.US, "%d,%.3f,1,%.4f,%.4f\n", i, t, p.x / res.inW, p.y / res.inH))
        }
        val name = "shuttle_${res.model.lowercase(Locale.US)}_${System.currentTimeMillis()}.csv"
        try {
            val where: String
            if (Build.VERSION.SDK_INT >= 29) {
                val cv = ContentValues().apply {
                    put(MediaStore.Downloads.DISPLAY_NAME, name)
                    put(MediaStore.Downloads.MIME_TYPE, "text/csv")
                    put(MediaStore.Downloads.RELATIVE_PATH, Environment.DIRECTORY_DOWNLOADS)
                }
                val uri = contentResolver.insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, cv)
                    ?: throw java.io.IOException("สร้างไฟล์ใน Downloads ไม่ได้")
                contentResolver.openOutputStream(uri)!!.use { it.write(sb.toString().toByteArray()) }
                where = "โฟลเดอร์ Downloads: $name"
            } else {
                val f = File(getExternalFilesDir(null) ?: filesDir, name)
                f.writeText(sb.toString())
                where = f.absolutePath
            }
            Toast.makeText(this, "บันทึกที่ $where", Toast.LENGTH_LONG).show()
        } catch (e: Exception) {
            Toast.makeText(this, "บันทึก CSV ไม่สำเร็จ: ${e.message}", Toast.LENGTH_LONG).show()
        }
    }

    override fun onDestroy() {
        cancel.set(true)
        retriever?.release()
        worker.shutdownNow()
        super.onDestroy()
    }

    companion object {
        private const val REQ_VIDEO = 1
    }
}
