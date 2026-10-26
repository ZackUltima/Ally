package ally.speech.phrases

import org.junit.Assert.assertEquals
import org.junit.Test

class NormaliseTest {
    @Test fun `punctuation, apostrophes and case go`() {
        assertEquals(listOf("help", "i", "cant", "breathe"), tokens("Help! I can’t BREATHE."))
        assertEquals(listOf("im", "okay"), tokens("I'm okay"))
    }

    @Test fun `cannot and i am are mapped`() {
        assertEquals(listOf("im", "bleeding"), tokens("I am bleeding"))
        assertEquals(listOf("i", "cant", "get", "up"), tokens("I cannot get up"))
        assertEquals(listOf("i", "cant", "breathe"), tokens("I can not breathe"))
    }

    @Test fun `digits stay and other symbols become spaces`() {
        assertEquals(listOf("room", "2", "now"), tokens("room-2, now"))
    }

    @Test fun `empty and symbol-only text give no tokens`() {
        assertEquals(emptyList<String>(), tokens(""))
        assertEquals(emptyList<String>(), tokens(" ?! "))
    }
}
