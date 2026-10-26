package ally.fall.flow

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.time.Duration.Companion.milliseconds
import kotlin.time.Duration.Companion.seconds

class SeveralPeopleTest {
    private val d = FlowDriver()

    @Test
    fun `a fall by anyone starts the flow`() {
        d.posesOf(mapOf(P1 to Posture.UPRIGHT, P2 to Posture.FALLEN), 100.milliseconds)
        assertEquals(P2, (d.flow.state as FlowState.Check).person)
    }

    @Test
    fun `two falls on one tick pick the lowest id`() {
        d.posesOf(mapOf(P2 to Posture.FALL, P1 to Posture.FALLEN), 100.milliseconds)
        assertEquals(P1, (d.flow.state as FlowState.Check).person)
    }

    @Test
    fun `another person standing does not cancel the check`() {
        d.posesOf(mapOf(P1 to Posture.UPRIGHT, P2 to Posture.FALLEN), 100.milliseconds)
        d.posesOf(mapOf(P1 to Posture.UPRIGHT, P2 to Posture.FALLEN), 10.seconds)
        assertTrue(d.flow.state is FlowState.Check)
    }

    @Test
    fun `Down-watch follows the person who fell`() { // Review Focus 4
        d.posesOf(mapOf(P1 to Posture.UPRIGHT, P2 to Posture.FALLEN), 100.milliseconds)
        d.say(Reply.OKAY)
        d.posesOf(mapOf(P1 to Posture.UPRIGHT, P2 to Posture.FALLEN), 10.seconds)
        assertEquals(P2, (d.flow.state as FlowState.DownWatch).person)
    }

    @Test
    fun `the person who fell standing up ends it`() {
        d.posesOf(mapOf(P1 to Posture.UPRIGHT, P2 to Posture.FALLEN), 100.milliseconds)
        d.say(Reply.OKAY)
        d.posesOf(mapOf(P1 to Posture.OTHER, P2 to Posture.UPRIGHT), 3.1.seconds)
        assertEquals(FlowState.Watching, d.flow.state)
    }
}
