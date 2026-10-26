package ally.emotion.levels

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.time.Duration.Companion.seconds

class CheckInTest {
    @Test fun `a reply is logged as answered`() {
        val d = LevelDriver()
        d.lowFace()
        assertEquals(listOf(Line.CHECK_IN_FINE), d.reply(soundsLow = false).lines())
        assertEquals(MoodDay(d.today, checkIns = 1, answered = 1), d.store.day(d.today))
        assertEquals(LevelState.Idle, d.rules.state)
    }

    @Test fun `no reply for 15 s - one follow-up, logged unanswered`() {
        val d = LevelDriver()
        d.lowFace()
        assertTrue("not before 15 s", d.wait(10.seconds).isEmpty())
        assertEquals(listOf(Line.CHECK_IN_NO_ANSWER), d.wait(30.seconds).lines())
        assertEquals(MoodDay(d.today, checkIns = 1, answered = 0), d.store.day(d.today))
        assertEquals(LevelState.Idle, d.rules.state)
    }

    @Test fun `a higher flow drops a check-in unlogged`() {
        val d = LevelDriver()
        d.lowFace()
        assertTrue(d.busy(BusyFlow.HIGHER, true).isEmpty())
        assertEquals(0, d.store.day(d.today).checkIns)
        assertEquals(LevelState.Idle, d.rules.state)
    }

    @Test fun `a second face stops a check-in`() {
        val d = LevelDriver()
        d.lowFace()
        assertEquals(listOf(Line.TALK_LATER), d.people(2).lines())
        assertEquals(0, d.store.day(d.today).checkIns)
        assertEquals(LevelState.Idle, d.rules.state)
    }

    @Test fun `answers that do not fit the state are ignored`() {
        val d = LevelDriver()
        assertTrue(d.reply(true).isEmpty())
        assertTrue(d.yesNo(YesNoAnswer.YES).isEmpty())
        assertTrue(d.phq(PhqAnswer.SEVERAL_DAYS).isEmpty())
        assertTrue(d.store.days().isEmpty())
    }

    @Test fun `a yes or no during a check-in is ignored`() {
        val d = LevelDriver()
        d.lowFace()
        assertTrue(d.yesNo(YesNoAnswer.YES).isEmpty())
        assertTrue(d.store.day(d.today).checkIns == 0 && !d.store.day(d.today).lowConfirmed)
        assertTrue(d.rules.state is LevelState.CheckIn)
    }
}
