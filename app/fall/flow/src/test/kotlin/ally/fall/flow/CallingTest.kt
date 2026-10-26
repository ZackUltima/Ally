package ally.fall.flow

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.time.Duration
import kotlin.time.Duration.Companion.milliseconds
import kotlin.time.Duration.Companion.minutes
import kotlin.time.Duration.Companion.seconds

class CallingTest {
    private val d = FlowDriver()

    /** A fall, then "help": calling starts with guardian 1. Returns the time of the escalation. */
    private fun escalate(): Duration {
        d.poses(P1, Posture.FALLEN, 100.milliseconds)
        d.say(Reply.HELP)
        return d.now
    }

    private fun placed(): List<GuardianId> = d.commands.filterIsInstance<PlaceCall>().map { it.guardian }

    @Test
    fun `calling order and rounds follow D-22`() {
        escalate()
        for (g in listOf(G1, G2, G3, G1, G2, G3, G1, G2)) {
            d.wait(1.seconds)
            val cmds = d.ended(g)
            val next = cmds.filterIsInstance<PlaceCall>().single().guardian
            assertTrue(Say(Line.STILL_CALLING, next) in cmds)
        }
        assertEquals(listOf(G1, G2, G3, G1, G2, G3, G1, G2, G3), placed())
    }

    @Test
    fun `a later round starts after the wait`() {
        escalate()
        for (g in listOf(G1, G2, G3, G1, G2, G3, G1, G2, G3)) d.ended(g)
        val endOfRound3 = d.now
        d.wait(10.minutes - 500.milliseconds)
        assertEquals(9, placed().size)
        d.wait(500.milliseconds)
        assertEquals(listOf(endOfRound3 + 10.minutes), d.timesOf(PlaceCall(G1)).drop(3))
        assertTrue(Say(Line.STILL_CALLING, G1) in d.commands)
    }

    @Test
    fun `a connected call pauses the alarm`() {
        escalate()
        assertEquals(listOf(PauseAlarm), d.connected(G1))
        val cmds = d.ended(G1)
        assertTrue(ResumeAlarm in cmds)
        assertTrue(PlaceCall(G2) in cmds)
    }

    @Test
    fun `a call that never reports back moves on after 90 s`() { // D-64 decision 3
        val start = escalate()
        d.wait(200.seconds)
        assertEquals(listOf(start + 90.seconds), d.timesOf(PlaceCall(G2)))
        assertEquals(listOf(start + 180.seconds), d.timesOf(PlaceCall(G3)))
    }

    @Test
    fun `a connected call has no safety timeout`() {
        escalate()
        d.connected(G1)
        d.wait(5.minutes)
        assertEquals(listOf(G1), placed())
    }

    @Test
    fun `a late call-ended from an earlier call is ignored`() { // Review Focus 2
        escalate()
        d.ended(G1)
        assertTrue(d.ended(G1).isEmpty())
        assertTrue(d.connected(G1).isEmpty())
        assertEquals(listOf(G1, G2), placed())
        assertTrue(PlaceCall(G3) in d.ended(G2))
    }

    @Test
    fun `the app acknowledgement stops everything`() {
        escalate()
        d.ended(G1)
        val cmds = d.acknowledged(G2)
        assertTrue(StopAlert in cmds)
        assertTrue(Say(Line.ACKNOWLEDGED, G2) in cmds)
        assertTrue(Notify(Handled(G2), listOf(G3)) in cmds)
        assertTrue(Log(FlowStage.HANDLED, d.now, by = G2) in cmds)
        d.wait(2.minutes + 29.seconds) // past the 90 s safety timeout; the person is still down (Down-watch)
        assertEquals(listOf(G1, G2), placed())
    }

    @Test
    fun `Help is here stops the calls`() {
        escalate()
        val cmds = d.helpIsHere()
        assertTrue(StopAlert in cmds)
        assertTrue(Say(Line.HELP_IS_HERE) in cmds)
        assertTrue(Notify(Handled(null), listOf(G2, G3)) in cmds)
        assertTrue(Log(FlowStage.HANDLED, d.now) in cmds)
        d.wait(2.minutes + 29.seconds) // past the 90 s safety timeout; the person is still down (Down-watch)
        assertEquals(listOf(G1), placed())
    }

    @Test
    fun `I'm okay between calls cancels with an all clear`() {
        escalate()
        for (g in listOf(G1, G2, G3, G1, G2, G3, G1, G2, G3)) d.ended(g)
        d.wait(1.minutes) // waiting for round 4
        val cmds = d.say(Reply.OKAY)
        assertTrue(StopAlert in cmds)
        assertTrue(Notify(AllClear, listOf(G2, G3)) in cmds)
        assertTrue(Say(Line.ALL_CLEAR) in cmds)
        assertTrue(Log(FlowStage.CANCELLED, d.now) in cmds)
        repeat(6) {
            // past the time round 4 was due; each Down-watch check-in is answered
            d.wait(2.minutes)
            d.say(Reply.OKAY)
        }
        assertEquals(9, placed().size)
    }

    @Test
    fun `I'm okay under the alarm, between calls, cancels`() {
        escalate()
        d.ended(G1) // guardian 2 is dialling, no call connected
        assertTrue(Log(FlowStage.CANCELLED, d.now) in d.say(Reply.OKAY))
    }

    @Test
    fun `I'm okay by voice is not heard while a call is connected`() { // Review Focus 3
        escalate()
        d.connected(G1)
        assertTrue(d.say(Reply.OKAY).isEmpty())
        val cmds = d.say(Reply.OKAY, Via.TAP)
        assertTrue(StopAlert in cmds)
        assertTrue(Log(FlowStage.CANCELLED, d.now) in cmds)
    }

    @Test
    fun `help and new falls are ignored while calling`() {
        escalate()
        assertTrue(d.say(Reply.HELP).isEmpty())
        assertTrue(d.say(Reply.HELP, Via.TAP).isEmpty())
        assertTrue(d.poses(P2, Posture.FALL, 1.seconds).isEmpty())
        assertEquals(1, d.commands.count { it is StartAlert })
    }

    @Test
    fun `standing up does not cancel once calling has started`() {
        escalate()
        d.poses(P1, Posture.UPRIGHT, 10.seconds)
        assertTrue(d.commands.none { it is Log && it.stage == FlowStage.CANCELLED })
        assertTrue(d.flow.state is FlowState.Calling)
    }

    @Test
    fun `a guardian without the app is still called`() {
        escalate()
        assertEquals(listOf(G1), placed()) // guardian 1 has no app
        d.ended(G1)
        assertEquals(listOf(G1, G2), placed())
    }
}
