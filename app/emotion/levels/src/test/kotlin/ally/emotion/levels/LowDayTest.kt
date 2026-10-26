package ally.emotion.levels

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.time.Duration.Companion.seconds

class LowDayTest {
    private val ask: List<Line> = listOf(Line.LOW_DAY_QUESTION)

    private fun asked(): LevelDriver {
        val d = LevelDriver()
        d.lowFace()
        d.reply(soundsLow = true)
        return d
    }

    @Test fun `a low reply in a check-in asks the low-day question`() {
        val d = LevelDriver()
        d.lowFace()
        assertEquals(ask, d.reply(soundsLow = true).lines())
        assertEquals(MoodDay(d.today, checkIns = 1, answered = 1), d.store.day(d.today))
        assertEquals(LevelState.LowDayQuestion(d.now, askedAgain = false, fromCheckIn = true), d.rules.state)
    }

    @Test fun `yes records a low day`() {
        val d = asked()
        assertTrue(d.yesNo(YesNoAnswer.YES).isEmpty())
        assertTrue(d.store.day(d.today).lowConfirmed)
        assertEquals(LevelState.Idle, d.rules.state)
    }

    @Test fun `no records nothing and is not asked again today`() {
        val d = asked()
        assertTrue(d.yesNo(YesNoAnswer.NO).isEmpty())
        assertFalse(d.store.day(d.today).lowConfirmed)
        assertEquals(LevelState.Idle, d.rules.state)
        assertTrue(d.suggest().isEmpty())
    }

    @Test fun `unclear is asked once more, then nothing`() {
        val d = asked()
        assertEquals(ask, d.yesNo(YesNoAnswer.UNCLEAR).lines())
        assertTrue(d.yesNo(YesNoAnswer.UNCLEAR).isEmpty())
        assertEquals(LevelState.Idle, d.rules.state)
        assertFalse(d.store.day(d.today).lowConfirmed)
        assertTrue("not asked again the same day", d.suggest().isEmpty())
    }

    @Test fun `silence is asked once more, then nothing`() {
        val d = asked()
        assertTrue(d.wait(10.seconds).isEmpty())
        assertEquals(ask, d.wait(10.seconds).lines())
        assertTrue(d.wait(10.seconds).isEmpty())
        assertTrue(d.wait(10.seconds).isEmpty())
        assertEquals(LevelState.Idle, d.rules.state)
    }

    @Test fun `not asked again on a day already low`() {
        val d = LevelDriver()
        d.store.putDay(MoodDay(d.today, lowConfirmed = true))
        d.lowFace()
        assertTrue(d.reply(soundsLow = true).isEmpty())
        assertEquals(1, d.store.day(d.today).answered)
        assertTrue(d.suggest().isEmpty())
    }

    @Test fun `a suggestion in a conversation asks the question`() {
        val d = LevelDriver()
        d.busy(BusyFlow.CONVERSATION, true)
        assertEquals(ask, d.suggest().lines())
        assertEquals(LevelState.LowDayQuestion(d.now, askedAgain = false, fromCheckIn = false), d.rules.state)
    }

    @Test fun `a suggestion during a higher flow or with two people is not asked`() {
        val d = LevelDriver()
        d.busy(BusyFlow.HIGHER, true)
        assertTrue(d.suggest().isEmpty())
        d.busy(BusyFlow.HIGHER, false)
        d.people(2)
        assertTrue(d.suggest().isEmpty())
    }

    @Test fun `a second suggestion while the question is open is ignored`() {
        val d = LevelDriver()
        d.suggest()
        assertTrue(d.suggest().isEmpty())
    }

    @Test fun `a second face stops the question`() {
        val d = asked()
        assertEquals(listOf(Line.TALK_LATER), d.people(2).lines())
        assertEquals(LevelState.Idle, d.rules.state)
    }

    @Test fun `a higher flow drops the question`() {
        val d = asked()
        assertTrue(d.busy(BusyFlow.HIGHER, true).isEmpty())
        assertEquals(LevelState.Idle, d.rules.state)
        assertFalse(d.store.day(d.today).lowConfirmed)
    }

    @Test fun `the question is asked again the next day`() {
        val d = asked()
        d.yesNo(YesNoAnswer.NO)
        d.nextDay()
        assertEquals(ask, d.suggest().lines())
    }
}
