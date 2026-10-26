package ally.emotion.levels

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.time.Duration.Companion.seconds

/** Whole timelines over several fake days, checked command by command. */
class ScenarioTest {
    @Test fun `three low days in a week lead to the PHQ-2 and a note`() {
        val d = LevelDriver()
        d.lowFace()
        d.reply(soundsLow = true)
        d.yesNo(YesNoAnswer.YES)
        d.nextDay()
        d.lowFace()
        d.reply(soundsLow = true)
        d.yesNo(YesNoAnswer.YES)
        d.nextDay()
        d.lowFace()
        d.reply(soundsLow = true)
        d.yesNo(YesNoAnswer.YES)
        d.yesNo(YesNoAnswer.YES)
        d.phq(PhqAnswer.MORE_THAN_HALF)
        d.phq(PhqAnswer.SEVERAL_DAYS)

        val day: List<LevelCommand> =
            listOf(Say(Line.CHECK_IN), Say(Line.LOW_DAY_QUESTION))
        val expected: List<LevelCommand> =
            day + day + day +
                listOf(
                    Say(Line.PHQ_LEAD_IN),
                    Say(Line.PHQ_ITEM_1),
                    Say(Line.PHQ_ITEM_2),
                    Say(Line.PHQ_CLOSING),
                    Say(Line.NOTE_BEING_SENT, G2),
                    SendNote(NoteKind.LOW_MOOD, G2),
                )
        assertEquals(expected, d.commands)
        assertEquals(Phq2Answers(PhqAnswer.MORE_THAN_HALF, PhqAnswer.SEVERAL_DAYS), d.store.day(d.today).phq)
        assertEquals(LevelState.Idle, d.rules.state)
    }

    @Test fun `a quiet person - unanswered check-ins on three days, then the no-answer note`() {
        val d = LevelDriver()
        repeat(3) {
            d.lowFace()
            d.wait(30.seconds)
            d.nextDay()
        }
        d.wait(5.seconds)
        d.wait(30.seconds)

        val unanswered: List<LevelCommand> = listOf(Say(Line.CHECK_IN), Say(Line.CHECK_IN_NO_ANSWER))
        val expected: List<LevelCommand> =
            unanswered + unanswered + unanswered +
                listOf(
                    Say(Line.PHQ_LEAD_IN),
                    Say(Line.PHQ_LEAD_IN),
                    Say(Line.NOTE_BEING_SENT, G2),
                    SendNote(NoteKind.LOW_MOOD_NO_ANSWER, G2),
                )
        assertEquals(expected, d.commands)
    }

    @Test fun `a person who says no to everything is left alone`() {
        val d = LevelDriver()
        d.lowFace()
        d.reply(soundsLow = true)
        d.yesNo(YesNoAnswer.NO)
        d.nextDay()
        d.lowFace()
        d.reply(soundsLow = false)
        d.nextDay()
        assertTrue(d.commands.none { it is SendNote })
        assertEquals(0, d.commands.count { it == Say(Line.PHQ_LEAD_IN) })
    }
}
