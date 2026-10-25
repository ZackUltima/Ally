package ally.fall.flow

import org.junit.Assert.assertEquals
import org.junit.Test
import kotlin.time.Duration
import kotlin.time.Duration.Companion.milliseconds
import kotlin.time.Duration.Companion.minutes
import kotlin.time.Duration.Companion.seconds

/** Whole timelines, checked command by command with their times. */
class ScenarioTest {
    private val d = FlowDriver()

    private fun ms(n: Int): Duration = n.milliseconds

    @Test
    fun `fall, silence, guardian 2 acknowledges, person stands up`() {
        d.poses(P1, Posture.FALLEN, 100.milliseconds)
        d.wait(30.seconds) // guardian 1 never reports back
        d.wait(90.seconds)
        d.connected(G2)
        d.poses(P1, Posture.FALLEN, 5.seconds)
        d.acknowledged(G2)
        d.poses(P1, Posture.UPRIGHT, 3.1.seconds)
        val expected =
            listOf(
                ms(100) to Log(FlowStage.CHECK, ms(100)),
                ms(100) to Say(Line.CHECK_FIRST),
                ms(15_100) to Say(Line.CHECK_SECOND),
                ms(30_100) to Log(FlowStage.ESCALATED, ms(30_100)),
                ms(30_100) to StartAlert(EventCode.FALL),
                ms(30_100) to Log(FlowStage.CALLING, ms(30_100)),
                ms(30_100) to PlaceCall(G1),
                ms(30_100) to Say(Line.CALLING, G1),
                ms(120_100) to PlaceCall(G2),
                ms(120_100) to Say(Line.STILL_CALLING, G2),
                ms(120_100) to PauseAlarm,
                ms(125_100) to StopAlert,
                ms(125_100) to Log(FlowStage.HANDLED, ms(125_100), by = G2),
                ms(125_100) to Say(Line.ACKNOWLEDGED, G2),
                ms(125_100) to Notify(Handled(G2), listOf(G3)),
                ms(125_100) to Log(FlowStage.DOWN_WATCH, ms(125_100)),
                ms(128_200) to Log(FlowStage.WATCHING, ms(128_200)),
            )
        assertEquals(expected, d.timed)
    }

    @Test
    fun `fall, I'm okay, still down, check-in unanswered, calls`() {
        d.poses(P1, Posture.FALLEN, 100.milliseconds)
        d.say(Reply.OKAY)
        d.poses(P1, Posture.FALLEN, 2.minutes + 30.seconds)
        val expected =
            listOf(
                ms(100) to Log(FlowStage.CHECK, ms(100)),
                ms(100) to Say(Line.CHECK_FIRST),
                ms(100) to Log(FlowStage.CANCELLED, ms(100)),
                ms(100) to Say(Line.CANCELLED),
                ms(100) to Log(FlowStage.DOWN_WATCH, ms(100)),
                ms(120_100) to Log(FlowStage.CHECK_IN, ms(120_100)),
                ms(120_100) to Say(Line.CHECK_IN_FIRST),
                ms(135_100) to Say(Line.CHECK_IN_SECOND),
                ms(150_100) to Log(FlowStage.ESCALATED, ms(150_100)),
                ms(150_100) to StartAlert(EventCode.FALL),
                ms(150_100) to Log(FlowStage.CALLING, ms(150_100)),
                ms(150_100) to PlaceCall(G1),
                ms(150_100) to Say(Line.CALLING, G1),
            )
        assertEquals(expected, d.timed)
    }

    @Test
    fun `every stage change is logged once`() {
        val transient = setOf(FlowStage.ESCALATED, FlowStage.HANDLED, FlowStage.CANCELLED)
        var changes = 0
        d.observer = { before, after, out ->
            val logged = out.filterIsInstance<Log>().map { it.stage }.filter { it !in transient }
            val expected = if (stageOf(after) != stageOf(before)) listOf(stageOf(after)) else emptyList()
            assertEquals("from $before to $after", expected, logged)
            if (expected.isNotEmpty()) changes += 1
            if (out.any { it is Log && it.stage in transient }) assertEquals(1, logged.size)
        }
        // a fall cancelled by standing up
        d.poses(P1, Posture.FALL, 1.seconds)
        d.poses(P1, Posture.UPRIGHT, 3.1.seconds)
        // a fall, "I'm okay" while down, an answered check-in, an unanswered one, calling, Help is here
        d.poses(P1, Posture.FALLEN, 1.seconds)
        d.say(Reply.OKAY)
        d.poses(P1, Posture.FALLEN, 2.minutes)
        d.say(Reply.OKAY, Via.TAP)
        d.noPose(2.minutes + 30.seconds)
        d.ended(G1)
        d.wait(100.seconds)
        d.helpIsHere()
        // still down: help in Down-watch, an all clear by tap, then up and out of Down-watch
        d.poses(P1, Posture.FALLEN, 10.seconds)
        d.say(Reply.HELP, Via.TAP)
        d.say(Reply.OKAY, Via.TAP)
        d.poses(P1, Posture.UPRIGHT, 3.1.seconds)
        // a help request from Watching, acknowledged
        d.say(Reply.HELP)
        d.acknowledged(G3)
        assertEquals(FlowState.Watching, d.flow.state)
        // Check, Watching; Check, Down-watch, Check-in, Down-watch, Check-in, Calling, Down-watch;
        // Calling, Down-watch, Watching; Calling, Watching.
        assertEquals(14, changes)
    }

    private fun stageOf(s: FlowState): FlowStage =
        when (s) {
            FlowState.Watching -> FlowStage.WATCHING
            is FlowState.Check -> FlowStage.CHECK
            is FlowState.Calling -> FlowStage.CALLING
            is FlowState.DownWatch -> FlowStage.DOWN_WATCH
            is FlowState.CheckIn -> FlowStage.CHECK_IN
        }
}
