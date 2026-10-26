package ally.emotion.levels

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test
import java.time.LocalDate

class BaselineTest {
    private val today: LocalDate = LocalDate.of(2026, 11, 15)
    private val s = LevelSettings()

    private fun day(
        daysAgo: Long,
        valid: Int,
        negative: Int,
    ): FaceCounts = FaceCounts(today.minusDays(daysAgo), valid, negative)

    private val sevenGood: List<FaceCounts> =
        listOf(
            day(1, 20, 10),
            day(2, 100, 10),
            day(3, 50, 5),
            day(4, 30, 3),
            day(5, 20, 2),
            day(6, 40, 4),
            day(7, 40, 6),
        )

    @Test fun `off with 6 qualifying days`() {
        assertNull(baselineShare(sevenGood.take(6), emptySet(), today, s))
    }

    @Test fun `on with 7, pooled over valid readings`() {
        assertEquals(40.0 / 300.0, baselineShare(sevenGood, emptySet(), today, s)!!, 1e-9)
    }

    @Test fun `low days, thin days and today are left out`() {
        val bad =
            listOf(
                day(8, 200, 190), // a low day
                day(9, 19, 19), // thin: 19 valid readings
                day(0, 200, 200), // today, still running
            )
        val low = setOf(today.minusDays(8))
        assertNull(baselineShare(sevenGood.take(6) + bad, low, today, s))
        assertEquals(40.0 / 300.0, baselineShare(sevenGood + bad, low, today, s)!!, 1e-9)
    }

    @Test fun `days older than 14 are left out`() {
        val old = day(15, 50, 5)
        assertNull(baselineShare(sevenGood.take(6) + old, emptySet(), today, s))
        val edge = day(14, 50, 5)
        // six good days (260 valid, 34 negative) plus the edge day (50, 5)
        assertEquals(39.0 / 310.0, baselineShare(sevenGood.take(6) + edge, emptySet(), today, s)!!, 1e-9)
    }

    @Test(expected = IllegalArgumentException::class)
    fun `two rows for one day are refused`() {
        baselineShare(sevenGood + day(1, 20, 5), emptySet(), today, s)
    }

    @Test fun `trigger share is 60 percent or baseline plus 20`() {
        assertEquals(0.60, triggerShare(null, s), 1e-9)
        assertEquals(0.60, triggerShare(0.30, s), 1e-9)
        assertEquals(0.65, triggerShare(0.45, s), 1e-9)
    }
}
