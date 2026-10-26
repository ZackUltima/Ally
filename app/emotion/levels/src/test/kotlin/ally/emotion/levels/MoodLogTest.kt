package ally.emotion.levels

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test
import java.time.LocalDate

class MoodLogTest {
    private val d: LocalDate = LocalDate.of(2026, 11, 15)

    @Test fun `an absent day reads as empty`() {
        val store = InMemoryMoodStore()
        assertEquals(MoodDay(d), store.day(d))
        assertEquals(FaceCounts(d, 0, 0), store.counts(d))
    }

    @Test fun `PHQ-2 total only when both items are scored`() {
        assertEquals(3, Phq2Answers(PhqAnswer.SEVERAL_DAYS, PhqAnswer.MORE_THAN_HALF).total)
        assertNull(Phq2Answers(null, PhqAnswer.NEARLY_EVERY_DAY).total)
        assertNull(Phq2Answers(PhqAnswer.NOT_AT_ALL, null).total)
    }

    @Test fun `the four answers score 0 to 3`() {
        assertEquals(listOf(0, 1, 2, 3), PhqAnswer.entries.map { it.score })
    }

    @Test fun `a stored day and its counts read back`() {
        val store = InMemoryMoodStore()
        store.putDay(MoodDay(d, checkIns = 2, answered = 1, lowConfirmed = true))
        store.putCounts(FaceCounts(d, 120, 30))
        assertEquals(2, store.day(d).checkIns)
        assertEquals(FaceCounts(d, 120, 30), store.counts(d))
    }

    @Test fun `putting a day again replaces it`() {
        val store = InMemoryMoodStore()
        store.putDay(MoodDay(d, checkIns = 1))
        store.putDay(MoodDay(d, checkIns = 2))
        assertEquals(1, store.days().size)
        assertEquals(2, store.day(d).checkIns)
    }

    @Test fun `days and counts come back oldest first`() {
        val store = InMemoryMoodStore()
        store.putDay(MoodDay(d))
        store.putDay(MoodDay(d.minusDays(2)))
        store.putDay(MoodDay(d.minusDays(1)))
        store.putCounts(FaceCounts(d, 1, 0))
        store.putCounts(FaceCounts(d.minusDays(3), 1, 0))
        assertEquals(listOf(d.minusDays(2), d.minusDays(1), d), store.days().map { it.date })
        assertEquals(listOf(d.minusDays(3), d), store.allCounts().map { it.date })
    }

    @Test fun `clear deletes days and counts`() {
        val store = InMemoryMoodStore()
        store.putDay(MoodDay(d, checkIns = 1))
        store.putCounts(FaceCounts(d, 50, 10))
        store.clear()
        assertEquals(emptyList<MoodDay>(), store.days())
        assertEquals(emptyList<FaceCounts>(), store.allCounts())
    }
}
