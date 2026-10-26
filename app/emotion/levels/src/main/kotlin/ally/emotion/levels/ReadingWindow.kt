package ally.emotion.levels

import java.time.LocalDateTime
import kotlin.time.Duration
import kotlin.time.toJavaDuration

/** The last `span` of expression readings; a "no reading" (null) is never stored, so it never counts (D-23 rule 3). */
class ReadingWindow(
    span: Duration,
) {
    private val span: java.time.Duration = span.toJavaDuration()
    private val readings: ArrayDeque<Pair<LocalDateTime, Expression>> = ArrayDeque()

    fun add(
        at: LocalDateTime,
        expression: Expression?,
    ) {
        if (expression == null) return
        readings.addLast(at to expression)
        while (readings.first().first.isBefore(at.minus(span))) readings.removeFirst()
    }

    fun validCount(now: LocalDateTime): Int = inWindow(now).size

    /** Negative readings over valid readings in the window; 0.0 when there are none. */
    fun negativeShare(now: LocalDateTime): Double {
        val valid = inWindow(now)
        if (valid.isEmpty()) return 0.0
        return valid.count { it.second.isNegative }.toDouble() / valid.size
    }

    fun clear() = readings.clear()

    private fun inWindow(now: LocalDateTime): List<Pair<LocalDateTime, Expression>> {
        val from = now.minus(span)
        return readings.filter { !it.first.isBefore(from) && !it.first.isAfter(now) }
    }
}
