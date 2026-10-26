package ally.emotion.levels

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.time.LocalDate
import java.time.LocalDateTime
import kotlin.time.Duration.Companion.hours
import kotlin.time.Duration.Companion.minutes
import kotlin.time.Duration.Companion.seconds

class FaceTriggerTest {
    private val checkIn: List<Line> = listOf(Line.CHECK_IN)

    @Test fun `19 valid readings never trigger`() {
        val d = LevelDriver()
        assertTrue(d.readings(Expression.SADNESS, 19).lines().isEmpty())
    }

    @Test fun `20 readings at 60 percent negative trigger`() {
        val d = LevelDriver()
        assertEquals(checkIn, d.lowFace().lines())
        assertEquals(LevelState.CheckIn(d.now), d.rules.state)
    }

    @Test fun `59 percent does not trigger`() {
        val d = LevelDriver()
        val out = d.readings(Expression.NEUTRAL, 41, every = 5.seconds) + d.readings(Expression.SADNESS, 59, every = 5.seconds)
        assertTrue(out.lines().isEmpty())
    }

    @Test fun `no reading never counts toward 20`() {
        val d = LevelDriver()
        val out = d.readings(Expression.SADNESS, 19) + d.readings(null, 10)
        assertTrue(out.lines().isEmpty())
        assertEquals(19, d.store.counts(d.today).valid)
    }

    @Test fun `readings older than the window stop counting`() {
        val d = LevelDriver()
        d.readings(Expression.SADNESS, 15)
        d.wait(11.minutes)
        assertTrue("15 old + 10 new would be 25, but the old ones are gone", d.readings(Expression.SADNESS, 10).lines().isEmpty())
    }

    @Test fun `a second face check-in waits 2 h`() {
        val d = LevelDriver()
        d.lowFace()
        val first = d.now
        d.reply(soundsLow = false)
        val early = d.readings(Expression.SADNESS, 700, every = 10.seconds)
        assertTrue("no check-in before 2 h", early.lines().isEmpty())
        var out: List<LevelCommand> = emptyList()
        repeat(100) { if (out.lines().isEmpty()) out = d.readings(Expression.SADNESS, 1) }
        assertEquals(checkIn, out.lines())
        val twoHours: LocalDateTime = first.plusHours(2)
        assertTrue("not before 2 h", !d.now.isBefore(twoHours))
        assertTrue("and then at the next reading", !d.now.isAfter(twoHours.plusSeconds(10)))
    }

    @Test fun `a companion greeting also starts the 2 h gap`() {
        val d = LevelDriver()
        d.greeting()
        assertTrue(d.lowFace().lines().isEmpty())
        d.wait(2.hours)
        assertEquals(checkIn, d.lowFace().lines())
    }

    @Test fun `no check-in during a conversation, it fires when the conversation ends`() {
        val d = LevelDriver()
        d.busy(BusyFlow.CONVERSATION, true)
        assertTrue(d.lowFace().lines().isEmpty())
        assertEquals(checkIn, d.busy(BusyFlow.CONVERSATION, false).lines())
    }

    @Test fun `no check-in during a higher flow, it fires when the flow ends`() {
        val d = LevelDriver()
        d.busy(BusyFlow.HIGHER, true)
        assertTrue(d.lowFace().lines().isEmpty())
        assertEquals(checkIn, d.busy(BusyFlow.HIGHER, false).lines())
    }

    @Test fun `two people in view - no check-in`() {
        val d = LevelDriver(people = 2)
        assertTrue(d.lowFace().lines().isEmpty())
        assertEquals(checkIn, d.people(1).lines())
    }

    @Test fun `nobody in view - no check-in`() {
        val d = LevelDriver(people = 0)
        assertTrue(d.lowFace().lines().isEmpty())
    }

    @Test fun `with the baseline on, the bar rises`() {
        val today: LocalDate = START.toLocalDate()

        fun seeded(): LevelDriver {
            val d = LevelDriver()
            for (back in 1L..7L) d.store.putCounts(FaceCounts(today.minusDays(back), 100, 45))
            return d
        }
        val low = seeded()
        val tooLow = low.readings(Expression.NEUTRAL, 38, every = 5.seconds) + low.readings(Expression.SADNESS, 62, every = 5.seconds)
        assertTrue("62% is under 45% + 20 points", tooLow.lines().isEmpty())
        val enough = seeded()
        val out = enough.readings(Expression.NEUTRAL, 35, every = 5.seconds) + enough.readings(Expression.SADNESS, 65, every = 5.seconds)
        assertEquals("65% reaches it", checkIn, out.lines())
        val plain = LevelDriver()
        val noBaseline =
            plain.readings(Expression.NEUTRAL, 38, every = 5.seconds) + plain.readings(Expression.SADNESS, 62, every = 5.seconds)
        assertEquals("without a baseline 60% is enough", Line.CHECK_IN, noBaseline.lines().first())
    }

    @Test fun `readings are counted per day for the baseline`() {
        val d = LevelDriver()
        d.readings(Expression.SADNESS, 5)
        d.readings(Expression.NEUTRAL, 3)
        d.readings(null, 4)
        assertEquals(FaceCounts(d.today, 8, 5), d.store.counts(d.today))
    }
}
