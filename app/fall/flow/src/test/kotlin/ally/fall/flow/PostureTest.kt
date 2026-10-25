package ally.fall.flow

import org.junit.Assert.assertEquals
import org.junit.Test

class PostureTest {
    @Test
    fun `labels map by name`() {
        assertEquals(Posture.FALL, postureOf("fall"))
        assertEquals(Posture.FALLEN, postureOf("fallen"))
        assertEquals(Posture.UPRIGHT, postureOf("standing"))
        assertEquals(Posture.UPRIGHT, postureOf("walk"))
        assertEquals(Posture.UPRIGHT, postureOf("upright")) // groups5 (T10)
        assertEquals(Posture.OTHER, postureOf("sitting"))
        assertEquals(Posture.OTHER, postureOf("lying"))
        assertEquals(Posture.OTHER, postureOf("rest"))
    }

    @Test(expected = IllegalArgumentException::class)
    fun `an empty label is refused`() {
        postureOf("")
    }
}
