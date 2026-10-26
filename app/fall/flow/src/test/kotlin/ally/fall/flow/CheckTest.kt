package ally.fall.flow

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.time.Duration
import kotlin.time.Duration.Companion.milliseconds
import kotlin.time.Duration.Companion.seconds

class CheckTest {
    private val d = FlowDriver()

    private fun trigger(): Duration {
        d.poses(P1, Posture.FALLEN, 100.milliseconds)
        return d.now
    }

    @Test
    fun `fall or fallen starts the check`() {
        for (posture in listOf(Posture.FALL, Posture.FALLEN)) {
            val driver = FlowDriver()
            val cmds = driver.poses(P1, posture, 100.milliseconds)
            assertTrue(Say(Line.CHECK_FIRST) in cmds)
            assertTrue(Log(FlowStage.CHECK, driver.now) in cmds)
        }
    }

    @Test
    fun `standing and sitting do not start the check`() {
        d.poses(P1, Posture.UPRIGHT, 2.seconds)
        d.poses(P1, Posture.OTHER, 2.seconds)
        assertTrue(d.commands.isEmpty())
    }

    @Test
    fun `a second ask at 15 s`() {
        val start = trigger()
        d.wait(25.seconds)
        assertEquals(listOf(start + 15.seconds), d.timesOf(Say(Line.CHECK_SECOND)))
    }

    @Test
    fun `silence for 30 s escalates`() {
        val start = trigger()
        d.wait(29.5.seconds)
        assertTrue(d.commands.none { it is StartAlert })
        val cmds = d.wait(0.5.seconds)
        assertEquals(start + 30.seconds, d.now)
        assertTrue(StartAlert(EventCode.FALL) in cmds)
        assertTrue(PlaceCall(G1) in cmds)
        assertTrue(Say(Line.CALLING, G1) in cmds)
        assertTrue(Log(FlowStage.ESCALATED, d.now) in cmds)
        assertTrue(Log(FlowStage.CALLING, d.now) in cmds)
    }

    @Test
    fun `an unclear answer does not cancel and still escalates at 30 s`() {
        val start = trigger()
        d.wait(5.seconds)
        assertTrue(d.say(Reply.UNCLEAR).isEmpty())
        d.wait(25.seconds)
        assertEquals(listOf(start + 30.seconds), d.timesOf(StartAlert(EventCode.FALL)))
    }

    @Test
    fun `help escalates at once`() {
        trigger()
        d.wait(4.seconds)
        val cmds = d.say(Reply.HELP)
        assertTrue(StartAlert(EventCode.FALL) in cmds)
        assertEquals(listOf(d.now), d.timesOf(StartAlert(EventCode.FALL)))
    }

    @Test
    fun `help by tap escalates at once`() {
        trigger()
        assertTrue(StartAlert(EventCode.FALL) in d.say(Reply.HELP, Via.TAP))
    }

    @Test
    fun `I'm okay cancels and calls nobody`() {
        trigger()
        d.wait(3.seconds)
        val cmds = d.say(Reply.OKAY)
        assertTrue(Say(Line.CANCELLED) in cmds)
        assertTrue(Log(FlowStage.CANCELLED, d.now) in cmds)
        d.wait(60.seconds)
        assertTrue(d.commands.none { it is StartAlert || it is PlaceCall })
    }

    @Test
    fun `the quiet note goes to app guardians only when on`() {
        val on = FlowDriver(FlowSettings(quietNoteOn = true))
        on.poses(P1, Posture.FALLEN, 100.milliseconds)
        assertTrue(Notify(QuietNote, listOf(G2, G3)) in on.say(Reply.OKAY))

        trigger()
        assertTrue(d.say(Reply.OKAY).none { it is Notify })
    }

    @Test
    fun `standing for 3 s cancels`() {
        trigger()
        d.poses(P1, Posture.UPRIGHT, 3.1.seconds) // 31 ticks span 3.0 s
        val stages = d.commands.filterIsInstance<Log>().map { it.stage }
        assertEquals(listOf(FlowStage.CHECK, FlowStage.CANCELLED, FlowStage.WATCHING), stages)
        assertTrue(Say(Line.CANCELLED) in d.commands)
        assertEquals(FlowState.Watching, d.flow.state)
    }

    @Test
    fun `movement on the floor does not cancel`() {
        trigger()
        repeat(29) {
            d.poses(P1, Posture.OTHER, 500.milliseconds)
            d.poses(P1, Posture.FALLEN, 500.milliseconds)
        }
        assertFalse(Log(FlowStage.CANCELLED, d.now) in d.commands)
        assertTrue(d.commands.none { it is Log && it.stage == FlowStage.CANCELLED })
    }

    @Test
    fun `the 30 s escalation fires without pose ticks`() { // Review Focus 1: the camera has stopped
        val start = trigger()
        d.wait(31.seconds)
        assertEquals(listOf(start + 30.seconds), d.timesOf(StartAlert(EventCode.FALL)))
    }

    @Test(expected = IllegalArgumentException::class)
    fun `an event earlier than the last is refused`() {
        d.send(ClockTick(5.seconds))
        d.send(ClockTick(4.seconds))
    }
}
