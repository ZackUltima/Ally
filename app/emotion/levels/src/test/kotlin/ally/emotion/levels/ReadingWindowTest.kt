package ally.emotion.levels

import org.junit.Assert.assertEquals
import org.junit.Test
import java.time.LocalDateTime
import kotlin.time.Duration.Companion.minutes

class ReadingWindowTest {
    private val t0: LocalDateTime = LocalDateTime.of(2026, 11, 15, 9, 0, 0)

    @Test fun `no reading never counts`() {
        val w = ReadingWindow(10.minutes)
        repeat(30) { w.add(t0.plusSeconds(it.toLong()), null) }
        val now = t0.plusSeconds(30)
        assertEquals(0, w.validCount(now))
        assertEquals(0.0, w.negativeShare(now), 1e-9)
    }

    @Test fun `readings older than 10 min drop out`() {
        val w = ReadingWindow(10.minutes)
        repeat(20) { w.add(t0, Expression.SADNESS) }
        assertEquals(20, w.validCount(t0.plusMinutes(10)))
        assertEquals(0, w.validCount(t0.plusMinutes(10).plusSeconds(1)))
    }

    @Test fun `the share is negative over valid`() {
        val w = ReadingWindow(10.minutes)
        repeat(12) { w.add(t0.plusSeconds(it.toLong()), Expression.SADNESS) }
        repeat(8) { w.add(t0.plusSeconds(12L + it), Expression.NEUTRAL) }
        w.add(t0.plusSeconds(20), null)
        val now = t0.plusSeconds(21)
        assertEquals(20, w.validCount(now))
        assertEquals(0.60, w.negativeShare(now), 1e-9)
    }

    @Test fun `old readings leave the share as well as the count`() {
        val w = ReadingWindow(10.minutes)
        repeat(10) { w.add(t0, Expression.ANGER) }
        repeat(10) { w.add(t0.plusMinutes(9), Expression.HAPPINESS) }
        val later = t0.plusMinutes(11)
        assertEquals(10, w.validCount(later))
        assertEquals(0.0, w.negativeShare(later), 1e-9)
    }

    @Test fun `clear empties the window`() {
        val w = ReadingWindow(10.minutes)
        repeat(5) { w.add(t0, Expression.FEAR) }
        w.clear()
        assertEquals(0, w.validCount(t0))
    }
}
