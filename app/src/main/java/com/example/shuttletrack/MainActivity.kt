package com.example.shuttletrack

import android.app.Activity
import android.content.Intent
import android.graphics.Color
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.view.Gravity
import android.view.ViewGroup.LayoutParams.MATCH_PARENT
import android.view.ViewGroup.LayoutParams.WRAP_CONTENT
import android.widget.Button
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.SeekBar
import android.widget.TextView
import android.widget.Toast
import java.io.File
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

class MainActivity : Activity() {

    private val ui = Handler(Looper.getMainLooper())
    private val worker = Executors.newSingleThreadExecutor()
    private val cancel = AtomicBoolean(false)

    private lateinit var pickBtn: Button
    private lateinit var exportBtn: Button
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
        status = TextView(this).apply { text = "ยังไม่ได้เลือกวิดีโอ"; setPadding(0, 16, 0, 8) }
        progress = ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal).apply { max = 100 }
        view = ResultView(this)
        seek = SeekBar(this).apply { isEnabled = false }
        exportBtn = Button(this).apply { text = "ส่งออก CSV"; isEnabled = false }

        root.addView(pickBtn, LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT))
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

        worker.execute {
            try {
                val detector = TrackNetDetector(this)
                detector.use { det ->
                    val res = VideoAnalyzer.analyze(
                        this, uri, det,
                        onProgress = { done, total ->
                            ui.post {
                                progress.progress = if (total > 0) done * 100 / total else 0
                                status.text = "วิเคราะห์แล้ว $done / $total เฟรม"
                            }
                        },
                        isCancelled = { cancel.get() }
                    )
                    ui.post { onAnalysisDone(res) }
                }
            } catch (e: java.io.FileNotFoundException) {
                fail("ไม่พบไฟล์โมเดล app/src/main/assets/tracknet.tflite — ดู README ขั้นตอนแปลงโมเดล")
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
        val pct = if (res.frameCount > 0) res.detected * 100 / res.frameCount else 0
        status.text = "เสร็จ: เจอลูก ${res.detected}/${res.frameCount} เฟรม ($pct%)"
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
        val dir = getExternalFilesDir(null) ?: filesDir
        val file = File(dir, "shuttle_track_${System.currentTimeMillis()}.csv")
        file.bufferedWriter().use { w ->
            w.write("frame,time_s,visible,x_norm,y_norm\n")
            res.points.forEachIndexed { i, p ->
                val t = i / res.fps
                if (p == null) w.write("$i,%.3f,0,,\n".format(t))
                else w.write("$i,%.3f,1,%.4f,%.4f\n".format(t, p.x / res.inW, p.y / res.inH))
            }
        }
        Toast.makeText(this, "บันทึกที่ ${file.absolutePath}", Toast.LENGTH_LONG).show()
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
