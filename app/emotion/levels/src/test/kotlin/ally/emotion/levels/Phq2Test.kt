package ally.emotion.levels

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.time.LocalDate
import kotlin.time.Duration.Companion.seconds

class Phq2Test {
    private val leadIn: List<Line> = listOf(Line.PHQ_LEAD_IN)

    /** Two low days already stand; a suggestion and a "yes" make the third. */
    private fun atLeadIn(): LevelDriver {
        val d = LevelDriver()
        d.lowDays(1, 2)
        d.suggest()
        assertEquals(leadIn, d.yesNo(YesNoAnswer.YES).lines())
        return d
    }

    private fun atItem1(): LevelDriver {
        val d = atLeadIn()
        assertEquals(listOf(Line.PHQ_ITEM_1), d.yesNo(YesNoAnswer.YES).lines())
        return d
    }

    @Test fun `offered after a confirmed low day completes 3 of 7`() {
        val d = atLeadIn()
        assertTrue(d.rules.state is LevelState.PhqLeadIn)
    }

    @Test fun `2 of 7 is not enough, and a low day eight days back does not count`() {
        val d = LevelDriver()
        d.lowDays(1, 7)
        d.suggest()
        assertTrue(d.yesNo(YesNoAnswer.YES).isEmpty())
        assertEquals(LevelState.Idle, d.rules.state)
    }

    @Test fun `offered at the next answered check-in`() {
        val d = LevelDriver()
        d.lowDays(1, 2, 3)
        d.lowFace()
        assertEquals(listOf(Line.CHECK_IN_FINE, Line.PHQ_LEAD_IN), d.reply(soundsLow = false).lines())
    }

    @Test fun `an unanswered day that completes the count waits until the person is next in view`() {
        val d = LevelDriver(people = 0)
        d.lowDays(1, 2)
        d.store.putDay(MoodDay(d.today, checkIns = 1, answered = 0))
        assertTrue("no offer at midnight", d.nextDay().isEmpty())
        assertEquals(leadIn, d.people(1).lines())
    }

    @Test fun `never on the midnight event itself`() {
        val d = LevelDriver()
        d.lowDays(1, 2, 3)
        assertTrue(d.nextDay().isEmpty())
        assertEquals(leadIn, d.wait(5.seconds).lines())
    }

    @Test fun `it waits while a conversation runs or two people are in view`() {
        val d = LevelDriver(people = 2)
        d.lowDays(1, 2, 3)
        assertTrue(d.nextDay().isEmpty())
        assertTrue(d.wait(5.seconds).isEmpty())
        d.busy(BusyFlow.CONVERSATION, true)
        assertTrue(d.people(1).isEmpty())
        assertEquals(leadIn, d.busy(BusyFlow.CONVERSATION, false).lines())
    }

    @Test fun `not now counts as offered and waits 14 days`() {
        val d = atLeadIn()
        assertTrue(d.yesNo(YesNoAnswer.NO).isEmpty())
        assertEquals(LevelState.Idle, d.rules.state)
        assertTrue(d.store.day(d.today).phqOffered)
    }

    @Test fun `a new offer comes 14 days after the last, not 13`() {
        fun offeredAgo(days: Long): List<Line> {
            val d = LevelDriver()
            val tomorrow: LocalDate = START.toLocalDate().plusDays(1)
            d.store.putDay(MoodDay(tomorrow.minusDays(days), phqOffered = true))
            d.lowDays(1, 2, 3)
            d.nextDay()
            return d.wait(5.seconds).lines()
        }
        assertTrue(offeredAgo(13).isEmpty())
        assertEquals(leadIn, offeredAgo(14))
    }

    @Test fun `an unclear lead-in is asked once more, then treated as not now`() {
        val d = atLeadIn()
        assertEquals(leadIn, d.yesNo(YesNoAnswer.UNCLEAR).lines())
        assertTrue(d.yesNo(YesNoAnswer.UNCLEAR).isEmpty())
        assertEquals(LevelState.Idle, d.rules.state)
        assertTrue(d.store.day(d.today).phqOffered)
        assertFalse(d.commands.any { it is SendNote })
    }

    @Test fun `silence on the lead-in once asks again`() {
        val d = atLeadIn()
        assertTrue(d.wait(10.seconds).isEmpty())
        assertEquals(leadIn, d.wait(10.seconds).lines())
        assertTrue(d.rules.state is LevelState.PhqLeadIn)
    }

    @Test fun `silence then an unclear reply is not no reply at all`() {
        val d = atLeadIn()
        d.wait(15.seconds)
        d.yesNo(YesNoAnswer.UNCLEAR)
        assertEquals(LevelState.Idle, d.rules.state)
        assertFalse(d.commands.any { it is SendNote })
    }

    @Test fun `an unclear reply then silence is not no reply at all`() {
        val d = atLeadIn()
        d.yesNo(YesNoAnswer.UNCLEAR)
        d.wait(30.seconds)
        assertEquals(LevelState.Idle, d.rules.state)
        assertFalse(d.commands.any { it is SendNote })
    }

    @Test fun `both items answered are scored and logged`() {
        val d = atItem1()
        assertEquals(listOf(Line.PHQ_ITEM_2), d.phq(PhqAnswer.SEVERAL_DAYS).lines())
        assertEquals(listOf(Line.PHQ_CLOSING), d.phq(PhqAnswer.SEVERAL_DAYS).lines())
        val day: MoodDay = d.store.day(d.today)
        assertEquals(Phq2Answers(PhqAnswer.SEVERAL_DAYS, PhqAnswer.SEVERAL_DAYS), day.phq)
        assertTrue(day.phqOffered)
        assertEquals(LevelState.Idle, d.rules.state)
    }

    @Test fun `an item unclear twice is not scored and sends no note`() {
        val d = atItem1()
        assertEquals(listOf(Line.PHQ_OPTIONS_AGAIN), d.phq(null).lines())
        assertEquals(listOf(Line.PHQ_ITEM_2), d.phq(null).lines())
        assertEquals(listOf(Line.PHQ_CLOSING), d.phq(PhqAnswer.NEARLY_EVERY_DAY).lines())
        val answers: Phq2Answers? = d.store.day(d.today).phq
        assertEquals(Phq2Answers(null, PhqAnswer.NEARLY_EVERY_DAY), answers)
        assertEquals(null, answers?.total)
        assertFalse(d.commands.any { it is SendNote })
    }

    @Test fun `the second item unclear twice ends with nothing scored`() {
        val d = atItem1()
        d.phq(PhqAnswer.MORE_THAN_HALF)
        d.phq(null)
        assertEquals(listOf(Line.PHQ_CLOSING), d.phq(null).lines())
        assertEquals(Phq2Answers(PhqAnswer.MORE_THAN_HALF, null), d.store.day(d.today).phq)
        assertFalse(d.commands.any { it is SendNote })
    }

    @Test fun `silence on an item counts as unclear`() {
        val d = atItem1()
        assertTrue(d.wait(10.seconds).isEmpty())
        assertEquals(listOf(Line.PHQ_OPTIONS_AGAIN), d.wait(10.seconds).lines())
    }

    @Test fun `stopped by a second person is not counted as offered and comes again`() {
        val d = atItem1()
        d.phq(PhqAnswer.SEVERAL_DAYS)
        assertEquals(listOf(Line.TALK_LATER), d.people(2).lines())
        assertFalse(d.store.day(d.today).phqOffered)
        assertNullAnswers(d)
        assertEquals(leadIn, d.people(1).lines())
    }

    @Test fun `stopped by a fall - nothing said, not counted as offered`() {
        val d = atItem1()
        assertTrue(d.busy(BusyFlow.HIGHER, true).isEmpty())
        assertFalse(d.store.day(d.today).phqOffered)
        assertEquals(LevelState.Idle, d.rules.state)
        assertEquals(leadIn, d.busy(BusyFlow.HIGHER, false).lines())
    }

    @Test fun `withdrawing consent mid questions stops them and keeps nothing`() {
        val d = atItem1()
        assertTrue(d.consent(false).isEmpty())
        assertEquals(LevelState.Idle, d.rules.state)
        assertTrue(d.store.days().isEmpty())
        assertTrue(d.phq(PhqAnswer.SEVERAL_DAYS).isEmpty())
        d.consent(true)
        assertTrue("nothing waits after consent comes back", d.wait(5.seconds).isEmpty())
    }

    private fun assertNullAnswers(d: LevelDriver) {
        assertEquals(null, d.store.day(d.today).phq)
    }
}
