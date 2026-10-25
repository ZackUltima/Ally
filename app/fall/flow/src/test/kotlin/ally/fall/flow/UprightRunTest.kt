package ally.fall.flow

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.time.Duration
import kotlin.time.Duration.Companion.milliseconds
import kotlin.time.Duration.Companion.seconds

class UprightRunTest {
    private val run = UprightRun(3.seconds)
    private var t: Duration = Duration.ZERO

    /** [n] ticks 0.1 s apart, the first at the current time. */
    private fun feed(
        n: Int,
        posture: Posture?,
    ) {
        repeat(n) {
            run.update(t, posture)
            t += 100.milliseconds
        }
    }

    @Test
    fun `2,9 s of standing is not enough`() {
        feed(30, Posture.UPRIGHT) // 0.0 .. 2.9 s
        assertFalse(run.isOffGround)
    }

    @Test
    fun `3,0 s of standing or walking is enough`() {
        feed(31, Posture.UPRIGHT) // 0.0 .. 3.0 s
        assertTrue(run.isOffGround)
    }

    @Test
    fun `sitting breaks the run`() {
        feed(20, Posture.UPRIGHT)
        feed(1, Posture.OTHER)
        feed(20, Posture.UPRIGHT)
        assertFalse(run.isOffGround)
    }

    @Test
    fun `a tick with no pose breaks the run`() { // D-64 decision 5
        feed(20, Posture.UPRIGHT)
        feed(1, null)
        feed(20, Posture.UPRIGHT)
        assertFalse(run.isOffGround)
    }
}
