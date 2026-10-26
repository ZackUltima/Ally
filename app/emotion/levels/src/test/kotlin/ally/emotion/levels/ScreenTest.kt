package ally.emotion.levels

import org.junit.Assert.assertEquals
import org.junit.Test
import java.time.LocalDateTime

class ScreenTest {
    private val at: LocalDateTime = START

    @Test fun `idle shows the watching face and no buttons`() {
        assertEquals(LevelScreen(Face.WATCHING, Buttons.NONE), levelScreen(LevelState.Idle))
    }

    @Test fun `a check-in shows the caring face`() {
        assertEquals(LevelScreen(Face.CARING, Buttons.NONE), levelScreen(LevelState.CheckIn(at)))
    }

    @Test fun `the low-day question shows Yes and No`() {
        val state = LevelState.LowDayQuestion(at, askedAgain = false, fromCheckIn = true)
        assertEquals(LevelScreen(Face.CARING, Buttons.YES_NO), levelScreen(state))
    }

    @Test fun `the lead-in shows Yes and Not now`() {
        val state = LevelState.PhqLeadIn(at, askedAgain = false, silentOnce = false)
        assertEquals(LevelScreen(Face.CARING, Buttons.YES_NOT_NOW), levelScreen(state))
    }

    @Test fun `the items show the four answers`() {
        val state = LevelState.PhqItem(1, at, repeated = false, first = null)
        assertEquals(LevelScreen(Face.CARING, Buttons.PHQ_FOUR), levelScreen(state))
    }
}
