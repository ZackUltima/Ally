package ally.fall.flow

import org.junit.Assert.assertEquals
import org.junit.Test
import kotlin.time.Duration.Companion.minutes
import kotlin.time.Duration.Companion.seconds

class ScreenTest {
    private val settings = FlowSettings(quietNoteOn = false)

    @Test
    fun `watching shows no buttons`() {
        assertEquals(FlowScreen(Face.WATCHING, emptySet(), null, null), screenFor(FlowState.Watching, settings))
    }

    @Test
    fun `the check shows the countdown and two buttons`() {
        val screen = screenFor(FlowState.Check(P1, 10.seconds, askedAgain = false), settings)
        assertEquals(FlowScreen(Face.ALERT, setOf(Button.OKAY, Button.HELP), 40.seconds, null), screen)
    }

    @Test
    fun `a check-in shows the same`() {
        val screen = screenFor(FlowState.CheckIn(P1, 2.minutes, askedAgain = true), settings)
        assertEquals(FlowScreen(Face.ALERT, setOf(Button.OKAY, Button.HELP), 2.minutes + 30.seconds, null), screen)
    }

    @Test
    fun `calling shows who is called, Help is here and I'm okay`() { // PO 2026-10-24: a tap can cancel at any time
        val calling = FlowState.Calling(EventCode.FALL, P1, G2, connected = true, placedAt = 5.seconds, waitUntil = null)
        val buttons = setOf(Button.OKAY, Button.HELP_IS_HERE)
        assertEquals(FlowScreen(Face.CALLING, buttons, null, G2), screenFor(calling, settings))
    }

    @Test
    fun `calling shows nobody while a later round waits`() {
        val waiting =
            FlowState.Calling(EventCode.FALL, P1, G1, connected = false, placedAt = null, waitUntil = 15.minutes)
        val buttons = setOf(Button.OKAY, Button.HELP_IS_HERE)
        assertEquals(FlowScreen(Face.CALLING, buttons, null, null), screenFor(waiting, settings))
    }

    @Test
    fun `Down-watch keeps a Help button`() { // D-64 decision 6
        val screen = screenFor(FlowState.DownWatch(P1, 2.minutes), settings)
        assertEquals(FlowScreen(Face.WATCHING, setOf(Button.HELP), null, null), screen)
    }
}
