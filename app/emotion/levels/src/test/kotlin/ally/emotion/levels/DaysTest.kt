package ally.emotion.levels

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.time.LocalDate

class DaysTest {
    private val today: LocalDate = LocalDate.of(2026, 11, 15)

    private fun ago(days: Long): LocalDate = today.minusDays(days)

    @Test fun `a day with check-ins and none answered is unanswered`() {
        assertTrue(MoodDay(today, checkIns = 2, answered = 0).isUnanswered())
    }

    @Test fun `one answered check-in makes the day not unanswered`() {
        assertFalse(MoodDay(today, checkIns = 3, answered = 1).isUnanswered())
    }

    @Test fun `a day with no check-in is not unanswered`() {
        assertFalse(MoodDay(today).isUnanswered())
    }

    @Test fun `low = confirmed or unanswered`() {
        assertTrue(MoodDay(today, lowConfirmed = true).isLow())
        assertTrue(MoodDay(today, checkIns = 1).isLow())
        assertFalse(MoodDay(today, checkIns = 1, answered = 1).isLow())
        assertFalse(MoodDay(today).isLow())
    }

    @Test fun `three low days of the last seven, today included`() {
        val store = InMemoryMoodStore()
        store.putDay(MoodDay(today, lowConfirmed = true))
        store.putDay(MoodDay(ago(3), checkIns = 1))
        store.putDay(MoodDay(ago(6), lowConfirmed = true))
        store.putDay(MoodDay(ago(7), lowConfirmed = true)) // the eighth day back is outside the seven
        store.putDay(MoodDay(ago(2), checkIns = 2, answered = 1)) // answered, so not low
        assertEquals(3, lowDayCount(store, today, 7))
        assertEquals(1, lowDayCount(store, today, 1))
    }

    @Test fun `the last date picks the newest matching day`() {
        val store = InMemoryMoodStore()
        store.putDay(MoodDay(ago(20), phqOffered = true))
        store.putDay(MoodDay(ago(5), phqOffered = true))
        store.putDay(MoodDay(ago(1)))
        assertEquals(ago(5), lastDate(store) { it.phqOffered })
        assertNull(lastDate(store) { it.noteSent })
    }

    @Test fun `days older than 30 and counts older than 14 are deleted`() {
        val store = InMemoryMoodStore()
        listOf(40L, 31L, 30L, 29L).forEach { store.putDay(MoodDay(ago(it), lowConfirmed = true)) }
        listOf(16L, 15L, 14L, 13L).forEach { store.putCounts(FaceCounts(ago(it), 40, 4)) }
        deleteOld(store, today, LevelSettings())
        assertEquals(listOf(ago(30), ago(29)), store.days().map { it.date })
        assertEquals(listOf(ago(14), ago(13)), store.allCounts().map { it.date })
    }

    @Test fun `after a 40-day gap the first deletion clears the old log`() {
        val store = InMemoryMoodStore()
        store.putDay(MoodDay(ago(40), lowConfirmed = true, checkIns = 1))
        store.putCounts(FaceCounts(ago(40), 100, 50))
        deleteOld(store, today, LevelSettings())
        assertEquals(emptyList<MoodDay>(), store.days())
        assertEquals(emptyList<FaceCounts>(), store.allCounts())
    }
}
