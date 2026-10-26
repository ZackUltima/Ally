package ally.emotion.levels

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.time.Duration.Companion.seconds

class NoteTest {
    private fun noteSent(d: LevelDriver): Boolean = d.commands.any { it is SendNote }

    /** Two low days stand; the third is confirmed, the lead-in is answered, and item 1 is being asked. */
    private fun atItem1(d: LevelDriver = LevelDriver()): LevelDriver {
        d.lowDays(1, 2)
        d.suggest()
        d.yesNo(YesNoAnswer.YES)
        d.yesNo(YesNoAnswer.YES)
        return d
    }

    private fun answer(
        d: LevelDriver,
        first: PhqAnswer,
        second: PhqAnswer,
    ): List<LevelCommand> {
        d.phq(first)
        return d.phq(second)
    }

    @Test fun `a total of 3 or more sends the low-mood note to the first app guardian`() {
        val d = atItem1()
        val out = answer(d, PhqAnswer.SEVERAL_DAYS, PhqAnswer.MORE_THAN_HALF)
        assertEquals(
            listOf<LevelCommand>(Say(Line.PHQ_CLOSING), Say(Line.NOTE_BEING_SENT, G2), SendNote(NoteKind.LOW_MOOD, G2)),
            out,
        )
        assertTrue(d.store.day(d.today).noteSent)
    }

    @Test fun `a total of 2 sends no note`() {
        val d = atItem1()
        answer(d, PhqAnswer.SEVERAL_DAYS, PhqAnswer.SEVERAL_DAYS)
        assertFalse(noteSent(d))
    }

    @Test fun `no reply at all sends the no-answer note`() {
        val d = LevelDriver()
        d.lowDays(1, 2)
        d.suggest()
        d.yesNo(YesNoAnswer.YES)
        assertEquals(listOf(Line.PHQ_LEAD_IN), d.wait(15.seconds).lines())
        val out = d.wait(15.seconds)
        assertEquals(listOf<LevelCommand>(Say(Line.NOTE_BEING_SENT, G2), SendNote(NoteKind.LOW_MOOD_NO_ANSWER, G2)), out)
        val day: MoodDay = d.store.day(d.today)
        assertTrue(day.phqOffered && day.noteSent)
        assertEquals(LevelState.Idle, d.rules.state)
    }

    @Test fun `one note a week at most`() {
        val tooSoon = LevelDriver()
        tooSoon.store.putDay(MoodDay(tooSoon.today.minusDays(6), noteSent = true))
        answer(atItem1(tooSoon), PhqAnswer.NEARLY_EVERY_DAY, PhqAnswer.NEARLY_EVERY_DAY)
        assertFalse("6 days ago is too soon", noteSent(tooSoon))

        val later = LevelDriver()
        later.store.putDay(MoodDay(later.today.minusDays(7), noteSent = true))
        answer(atItem1(later), PhqAnswer.NEARLY_EVERY_DAY, PhqAnswer.NEARLY_EVERY_DAY)
        assertTrue("7 days ago is enough", noteSent(later))
    }

    @Test fun `switch off - no note, for a high total or for no reply`() {
        val high = LevelDriver(noteOn = false)
        answer(atItem1(high), PhqAnswer.NEARLY_EVERY_DAY, PhqAnswer.NEARLY_EVERY_DAY)
        assertFalse(noteSent(high))

        val silent = LevelDriver()
        silent.send(NoteSwitchChanged(silent.now.plusSeconds(1), on = false))
        silent.lowDays(1, 2)
        silent.suggest()
        silent.yesNo(YesNoAnswer.YES)
        silent.wait(30.seconds)
        assertFalse(noteSent(silent))
        assertTrue(silent.store.day(silent.today).phqOffered)
    }

    @Test fun `the switch turned back on lets the next note go`() {
        val d = LevelDriver(noteOn = false)
        d.send(NoteSwitchChanged(d.now.plusSeconds(1), on = true))
        answer(atItem1(d), PhqAnswer.NEARLY_EVERY_DAY, PhqAnswer.NEARLY_EVERY_DAY)
        assertTrue(noteSent(d))
    }
}
