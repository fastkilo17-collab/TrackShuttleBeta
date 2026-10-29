package com.example.shuttletrack

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.PointF
import android.graphics.RectF
import android.view.View

/** แสดงเฟรมวิดีโอ + จุดลูก + หางวิถี (ย้อนหลัง trailLength เฟรม) */
class ResultView(context: Context) : View(context) {

    var frame: Bitmap? = null
    var points: List<PointF?> = emptyList()
    var currentIndex = 0
    var srcW = 512
    var srcH = 288
    var trailLength = 15

    private val dot = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.YELLOW; style = Paint.Style.FILL }
    private val ring = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.RED; style = Paint.Style.STROKE; strokeWidth = 4f
    }
    private val trail = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.CYAN; style = Paint.Style.STROKE; strokeWidth = 4f
    }

    override fun onDraw(canvas: Canvas) {
        canvas.drawColor(Color.BLACK)
        val bmp = frame ?: return
        val scale = minOf(width.toFloat() / bmp.width, height.toFloat() / bmp.height)
        val dw = bmp.width * scale
        val dh = bmp.height * scale
        val ox = (width - dw) / 2f
        val oy = (height - dh) / 2f
        canvas.drawBitmap(bmp, null, RectF(ox, oy, ox + dw, oy + dh), null)

        fun map(p: PointF) = PointF(ox + p.x / srcW * dw, oy + p.y / srcH * dh)

        var prev: PointF? = null
        val from = maxOf(0, currentIndex - trailLength)
        for (i in from..minOf(currentIndex, points.lastIndex)) {
            val p = points[i]?.let { map(it) }
            if (p != null && prev != null) canvas.drawLine(prev.x, prev.y, p.x, p.y, trail)
            if (p != null) prev = p
        }
        points.getOrNull(currentIndex)?.let {
            val p = map(it)
            canvas.drawCircle(p.x, p.y, 6f, dot)
            canvas.drawCircle(p.x, p.y, 16f, ring)
        }
    }
}
