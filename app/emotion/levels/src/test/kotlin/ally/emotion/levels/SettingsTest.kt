package ally.emotion.levels

import org.junit.Assert.assertEquals
import org.junit.Test
import kotlin.time.Duration.Companion.hours
import kotlin.time.Duration.Companion.minutes
import kotlin.time.Duration.Companion.seconds

class SettingsTest {
    @Test fun `starting values match D-23 and the design plan`() {
        val s = LevelSettings()
        assertEquals(10.minutes, s.window)
        assertEquals(20, s.minValid)
        assertEquals(0.60, s.negativeShare, 1e-9)
        assertEquals(2.hours, s.faceGap)
        assertEquals(15.seconds, s.replyWait)
        assertEquals(14, s.baselineDays)
        assertEquals(7, s.baselineMinDays)
        assertEquals(0.20, s.baselineMargin, 1e-9)
        assertEquals(3, s.phqLowDays)
        assertEquals(7, s.phqWindowDays)
        assertEquals(14, s.phqGapDays)
        assertEquals(7, s.noteGapDays)
        assertEquals(30, s.keepDays)
    }

    @Test(expected = IllegalArgumentException::class)
    fun `a minimum of zero valid readings is refused`() {
        LevelSettings(minValid = 0)
    }

    @Test(expected = IllegalArgumentException::class)
    fun `the baseline cannot need more days than it looks back over`() {
        LevelSettings(baselineDays = 5, baselineMinDays = 6)
    }
}
