package ally.emotion.levels

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class ConsentTest {
    @Test fun `consent off - no face trigger and no low-day question`() {
        val d = LevelDriver(consent = false)
        assertTrue(d.lowFace().lines().isEmpty())
        assertTrue(d.suggest().isEmpty())
        assertTrue(d.store.days().isEmpty())
        assertTrue(d.store.allCounts().isEmpty())
    }

    @Test fun `consent off - a higher flow is still accepted`() {
        val d = LevelDriver(consent = false)
        assertTrue(d.busy(BusyFlow.HIGHER, true).isEmpty())
        assertTrue(d.busy(BusyFlow.HIGHER, false).isEmpty())
        assertTrue(d.commands.isEmpty())
    }

    @Test fun `withdrawing consent mid check-in stops it and deletes everything`() {
        val d = LevelDriver()
        d.store.putDay(MoodDay(START.toLocalDate().minusDays(2), lowConfirmed = true))
        d.lowFace()
        assertTrue(d.rules.state is LevelState.CheckIn)
        assertTrue(d.consent(false).isEmpty())
        assertEquals(LevelState.Idle, d.rules.state)
        assertTrue(d.store.days().isEmpty())
        assertTrue(d.store.allCounts().isEmpty())
        assertTrue(d.reply(soundsLow = false).isEmpty())
        assertTrue(d.readings(Expression.SADNESS, 30).isEmpty())
        assertTrue("nothing written afterwards", d.store.days().isEmpty() && d.store.allCounts().isEmpty())
    }

    @Test fun `consent given again starts from nothing`() {
        val d = LevelDriver()
        d.lowFace()
        d.consent(false)
        d.consent(true)
        assertEquals(listOf(Line.CHECK_IN), d.lowFace().lines())
    }

    @Test fun `forget my mood diary deletes days and counts`() {
        val d = LevelDriver()
        d.store.putDay(MoodDay(START.toLocalDate().minusDays(2), lowConfirmed = true))
        d.readings(Expression.SADNESS, 5)
        assertTrue(d.forget().isEmpty())
        assertTrue(d.store.days().isEmpty())
        assertTrue(d.store.allCounts().isEmpty())
    }

    @Test fun `the note switch can change without a command`() {
        val d = LevelDriver()
        assertTrue(d.send(NoteSwitchChanged(d.now.plusSeconds(1), on = false)).isEmpty())
    }
}
