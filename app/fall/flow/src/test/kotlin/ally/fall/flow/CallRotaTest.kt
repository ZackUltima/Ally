package ally.fall.flow

import org.junit.Assert.assertEquals
import org.junit.Test
import kotlin.time.Duration.Companion.minutes
import kotlin.time.Duration.Companion.seconds

class CallRotaTest {
    private val t = 100.seconds

    @Test
    fun `three rounds back to back, in order`() {
        val rota = CallRota(THREE_GUARDIANS, backToBackRounds = 3, laterRoundGap = 10.minutes)
        val calls = listOf(rota.first()) + List(8) { (rota.afterEnded(t) as CallRota.Next.Call).guardian }
        assertEquals(listOf(G1, G2, G3, G1, G2, G3, G1, G2, G3), calls)
    }

    @Test
    fun `after round 3, the next round waits 10 min from its end`() { // D-64 decision 2
        val rota = CallRota(THREE_GUARDIANS, backToBackRounds = 3, laterRoundGap = 10.minutes)
        rota.first()
        repeat(8) { rota.afterEnded(t) }
        assertEquals(CallRota.Next.WaitUntil(t + 10.minutes, G1), rota.afterEnded(t))
        assertEquals(CallRota.Next.Call(G2), rota.afterEnded(t))
        assertEquals(CallRota.Next.Call(G3), rota.afterEnded(t))
        assertEquals(CallRota.Next.WaitUntil(t + 30.minutes, G1), rota.afterEnded(t + 20.minutes))
    }

    @Test
    fun `one guardian is called again each round`() {
        val one = Guardians(listOf(Guardian(G1, "Mei", hasApp = true)))
        val rota = CallRota(one, backToBackRounds = 3, laterRoundGap = 10.minutes)
        assertEquals(G1, rota.first())
        assertEquals(CallRota.Next.Call(G1), rota.afterEnded(t))
        assertEquals(CallRota.Next.Call(G1), rota.afterEnded(t))
        assertEquals(CallRota.Next.WaitUntil(t + 10.minutes, G1), rota.afterEnded(t))
    }
}
