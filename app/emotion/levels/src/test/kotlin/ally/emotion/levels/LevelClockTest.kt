package ally.emotion.levels

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.time.LocalDate
import java.time.LocalDateTime
import kotlin.time.Duration.Companion.seconds

class LevelClockTest {
    @Test fun `an earlier time is treated as the last time and reported`() {
        val d = LevelDriver()
        d.readings(Expression.SADNESS, 3)
        val last: LocalDateTime = d.now
        val earlier: LocalDateTime = last.minusMinutes(5)
        val out = d.send(Tick(earlier))
        assertEquals(listOf<LevelCommand>(ClockWentBack(last, earlier)), out)
        assertEquals(LevelState.Idle, d.rules.state)
        assertEquals(FaceCounts(last.toLocalDate(), 3, 3), d.store.counts(last.toLocalDate()))
    }

    @Test fun `rules go on from the last time after a clock set back`() {
        val d = LevelDriver()
        d.readings(Expression.NEUTRAL, 8)
        d.send(Tick(d.now.minusHours(3)))
        val out = d.readings(Expression.SADNESS, 12)
        assertEquals(listOf(Line.CHECK_IN), out.lines())
    }

    @Test fun `the first event of a new day deletes days and counts that are too old`() {
        val d = LevelDriver()
        val today: LocalDate = START.toLocalDate()
        d.store.putDay(MoodDay(today.minusDays(40), lowConfirmed = true))
        d.store.putDay(MoodDay(today.minusDays(5), lowConfirmed = true))
        d.store.putCounts(FaceCounts(today.minusDays(20), 50, 10))
        d.store.putCounts(FaceCounts(today.minusDays(3), 50, 10))
        d.wait(10.seconds)
        assertTrue("nothing is deleted on the same day", d.store.days().size == 2)
        assertTrue("nor the counts", d.store.allCounts().size == 2)
        d.nextDay()
        assertEquals(listOf(today.minusDays(5)), d.store.days().map { it.date })
        assertEquals(listOf(today.minusDays(3)), d.store.allCounts().map { it.date })
    }

    @Test fun `a day with no events does not stop the rules`() {
        val d = LevelDriver()
        d.nextDay()
        d.nextDay()
        d.nextDay()
        assertTrue(d.commands.isEmpty())
        assertEquals(LevelState.Idle, d.rules.state)
    }
}
