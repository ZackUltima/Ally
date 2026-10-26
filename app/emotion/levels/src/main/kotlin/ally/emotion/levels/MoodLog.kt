package ally.emotion.levels

import java.time.LocalDate
import java.util.TreeMap

/** The four PHQ-2 answers and their scores (D-23). */
enum class PhqAnswer(
    val score: Int,
) {
    NOT_AT_ALL(0),
    SEVERAL_DAYS(1),
    MORE_THAN_HALF(2),
    NEARLY_EVERY_DAY(3),
}

/** The two PHQ-2 items; a null answer was not scored, and the total exists only when both were. */
data class Phq2Answers(
    val first: PhqAnswer?,
    val second: PhqAnswer?,
) {
    val total: Int? = if (first != null && second != null) first.score + second.score else null
}

/** One day of the mood log: check-ins asked and answered, and what was offered or sent (design plan E13). */
data class MoodDay(
    val date: LocalDate,
    val checkIns: Int = 0,
    val answered: Int = 0,
    val lowConfirmed: Boolean = false,
    val phqOffered: Boolean = false,
    val phq: Phq2Answers? = null,
    val noteSent: Boolean = false,
)

/** Where the mood log lives. The Room version is built in EM3 behind this same interface (EM2a decision 7). */
interface MoodStore {
    fun day(date: LocalDate): MoodDay

    fun putDay(day: MoodDay)

    fun days(): List<MoodDay>

    fun counts(date: LocalDate): FaceCounts

    fun putCounts(counts: FaceCounts)

    fun allCounts(): List<FaceCounts>

    fun deleteDaysBefore(date: LocalDate)

    fun deleteCountsBefore(date: LocalDate)

    fun clear()
}

/** The store used by the tests and until EM3: two maps by date. */
class InMemoryMoodStore : MoodStore {
    private val moodDays: TreeMap<LocalDate, MoodDay> = TreeMap()
    private val faceCounts: TreeMap<LocalDate, FaceCounts> = TreeMap()

    override fun day(date: LocalDate): MoodDay = moodDays[date] ?: MoodDay(date)

    override fun putDay(day: MoodDay) {
        moodDays[day.date] = day
    }

    override fun days(): List<MoodDay> = moodDays.values.toList()

    override fun counts(date: LocalDate): FaceCounts = faceCounts[date] ?: FaceCounts(date, 0, 0)

    override fun putCounts(counts: FaceCounts) {
        faceCounts[counts.date] = counts
    }

    override fun allCounts(): List<FaceCounts> = faceCounts.values.toList()

    override fun deleteDaysBefore(date: LocalDate) {
        moodDays.headMap(date).clear()
    }

    override fun deleteCountsBefore(date: LocalDate) {
        faceCounts.headMap(date).clear()
    }

    override fun clear() {
        moodDays.clear()
        faceCounts.clear()
    }
}
