package ally.fall.flow

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.time.Duration
import kotlin.time.Duration.Companion.milliseconds
import kotlin.time.Duration.Companion.minutes
import kotlin.time.Duration.Companion.seconds

class DownWatchTest {
    private val d = FlowDriver()

    /** A fall and "I'm okay" while still down: Down-watch starts. Returns the time it started. */
    private fun downWatch(person: PersonId = P1): Duration {
        d.poses(person, Posture.FALLEN, 100.milliseconds)
        d.say(Reply.OKAY)
        return d.now
    }

    private fun stages(): List<FlowStage> = d.commands.filterIsInstance<Log>().map { it.stage }

    @Test
    fun `I'm okay while still down starts Down-watch`() {
        val start = downWatch()
        assertTrue(Log(FlowStage.DOWN_WATCH, start) in d.commands)
        assertEquals(FlowState.DownWatch(P1, start + 2.minutes), d.flow.state)
    }

    @Test
    fun `one alert per fall`() { // design choice 11; each check-in is answered
        downWatch()
        repeat(5) {
            d.poses(P1, Posture.FALLEN, 2.minutes)
            d.say(Reply.OKAY)
        }
        assertEquals(1, d.commands.count { it == Say(Line.CHECK_FIRST) })
        assertTrue(d.commands.none { it is StartAlert })
    }

    @Test
    fun `a check-in every 2 min while down`() {
        val start = downWatch()
        d.poses(P1, Posture.FALLEN, 2.minutes)
        d.say(Reply.OKAY)
        d.poses(P1, Posture.FALLEN, 2.minutes)
        assertEquals(listOf(start + 2.minutes, start + 4.minutes), d.timesOf(Say(Line.CHECK_IN_FIRST)))
        assertTrue(Log(FlowStage.CHECK_IN, start + 2.minutes) in d.commands)
    }

    @Test
    fun `an unanswered check-in is asked again, then escalates`() { // D-64 decision 1
        val start = downWatch()
        d.poses(P1, Posture.FALLEN, 2.minutes + 31.seconds)
        val asked = start + 2.minutes
        assertEquals(listOf(asked + 15.seconds), d.timesOf(Say(Line.CHECK_IN_SECOND)))
        assertEquals(listOf(asked + 30.seconds), d.timesOf(StartAlert(EventCode.FALL)))
        assertEquals(P1, (d.flow.state as FlowState.Calling).person)
    }

    @Test
    fun `an unclear answer to a check-in still escalates`() {
        val start = downWatch()
        d.wait(2.minutes + 5.seconds)
        d.say(Reply.UNCLEAR)
        d.wait(26.seconds)
        assertEquals(listOf(start + 2.minutes + 30.seconds), d.timesOf(StartAlert(EventCode.FALL)))
    }

    @Test
    fun `help in Down-watch escalates`() {
        downWatch()
        d.wait(30.seconds)
        assertTrue(StartAlert(EventCode.FALL) in d.say(Reply.HELP, Via.TAP))
    }

    @Test
    fun `help in a check-in escalates`() {
        downWatch()
        d.wait(2.minutes)
        assertTrue(StartAlert(EventCode.FALL) in d.say(Reply.HELP))
    }

    @Test
    fun `I'm okay outside a check-in changes nothing`() {
        downWatch()
        d.wait(30.seconds)
        assertTrue(d.say(Reply.OKAY).isEmpty())
    }

    @Test
    fun `off the ground ends Down-watch`() {
        downWatch()
        d.poses(P1, Posture.UPRIGHT, 3.1.seconds)
        assertEquals(FlowStage.WATCHING, stages().last())
        assertEquals(FlowState.Watching, d.flow.state)
        val cmds = d.poses(P1, Posture.FALL, 100.milliseconds)
        assertTrue(Say(Line.CHECK_FIRST) in cmds)
    }

    @Test
    fun `sitting does not count as off the ground`() {
        downWatch()
        d.poses(P1, Posture.OTHER, 10.seconds)
        assertTrue(d.flow.state is FlowState.DownWatch)
    }

    @Test
    fun `standing for 3 s during a check-in returns to Watching`() { // T7
        downWatch()
        d.poses(P1, Posture.FALLEN, 2.minutes)
        assertTrue(d.flow.state is FlowState.CheckIn)
        d.poses(P1, Posture.UPRIGHT, 3.1.seconds)
        assertEquals(FlowState.Watching, d.flow.state)
        assertTrue(d.commands.none { it is StartAlert })
    }

    @Test
    fun `last seen upright, then out of view`() { // Review Focus 5, D-64 decision 5: the known false-alarm path
        val start = downWatch()
        d.poses(P1, Posture.UPRIGHT, 2.seconds)
        d.noPose(2.minutes + 30.seconds)
        assertTrue(Say(Line.CHECK_IN_FIRST) in d.commands)
        assertEquals(listOf(start + 2.minutes + 30.seconds), d.timesOf(StartAlert(EventCode.FALL)))
    }

    @Test
    fun `after an acknowledgement, a person still down is watched`() {
        d.poses(P1, Posture.FALLEN, 100.milliseconds)
        d.say(Reply.HELP)
        d.poses(P1, Posture.FALLEN, 20.seconds)
        val cmds = d.acknowledged(G2)
        assertTrue(Log(FlowStage.DOWN_WATCH, d.now) in cmds)
        assertEquals(FlowState.DownWatch(P1, d.now + 2.minutes), d.flow.state)
    }

    @Test
    fun `after an acknowledgement, a person already up returns to Watching`() {
        d.poses(P1, Posture.FALLEN, 100.milliseconds)
        d.say(Reply.HELP)
        d.poses(P1, Posture.UPRIGHT, 3.1.seconds)
        val cmds = d.acknowledged(G2)
        assertTrue(Log(FlowStage.WATCHING, d.now) in cmds)
        assertEquals(FlowState.Watching, d.flow.state)
    }

    @Test
    fun `an all clear while still down starts Down-watch`() {
        d.poses(P1, Posture.FALLEN, 100.milliseconds)
        d.say(Reply.HELP)
        val cmds = d.say(Reply.OKAY, Via.TAP)
        assertTrue(Log(FlowStage.DOWN_WATCH, d.now) in cmds)
    }

    @Test
    fun `help in Watching starts a help request`() { // D-64 decision 4
        val cmds = d.say(Reply.HELP)
        assertTrue(StartAlert(EventCode.HELP_REQUEST) in cmds)
        assertTrue(PlaceCall(G1) in cmds)
        val handled = d.acknowledged(G1)
        assertTrue(Log(FlowStage.WATCHING, d.now) in handled)
        assertEquals(FlowState.Watching, d.flow.state)
    }

    @Test
    fun `I'm okay or unclear in Watching changes nothing`() {
        assertTrue(d.say(Reply.OKAY).isEmpty())
        assertTrue(d.say(Reply.UNCLEAR, Via.TAP).isEmpty())
    }
}
