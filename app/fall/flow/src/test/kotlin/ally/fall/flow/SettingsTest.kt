package ally.fall.flow

import org.junit.Assert.assertEquals
import org.junit.Test
import kotlin.time.Duration.Companion.minutes
import kotlin.time.Duration.Companion.seconds

class SettingsTest {
    private fun g(
        id: Int,
        app: Boolean,
    ): Guardian = Guardian(GuardianId(id), "Guardian $id", app)

    @Test
    fun `starting values match D-22, D-55 and D-64`() {
        val s = FlowSettings(quietNoteOn = true)
        assertEquals(15.seconds, s.secondAsk)
        assertEquals(30.seconds, s.checkTimeout)
        assertEquals(3.seconds, s.offGroundHold)
        assertEquals(2.minutes, s.checkInEvery)
        assertEquals(15.seconds, s.checkInSecondAsk)
        assertEquals(30.seconds, s.checkInTimeout)
        assertEquals(3, s.backToBackRounds)
        assertEquals(10.minutes, s.laterRoundGap)
        assertEquals(90.seconds, s.callSafetyTimeout)
    }

    @Test(expected = IllegalArgumentException::class)
    fun `no guardians is refused`() {
        Guardians(emptyList())
    }

    @Test(expected = IllegalArgumentException::class)
    fun `four guardians are refused`() {
        Guardians(List(4) { g(it + 1, app = true) })
    }

    @Test(expected = IllegalArgumentException::class)
    fun `guardians without any app are refused`() {
        Guardians(listOf(g(1, app = false)))
    }

    @Test(expected = IllegalArgumentException::class)
    fun `a repeated guardian id is refused`() {
        Guardians(listOf(g(1, app = true), g(1, app = false)))
    }

    @Test
    fun `calling order is kept and app guardians are listed`() {
        val gs = Guardians(listOf(g(1, app = false), g(2, app = true), g(3, app = true)))
        assertEquals(listOf(1, 2, 3), gs.inOrder.map { it.id.value })
        assertEquals(listOf(2, 3), gs.withApp.map { it.id.value })
    }
}
