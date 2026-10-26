package ally.emotion.levels

import java.time.LocalDate

/** Check-ins were asked and none was answered (design plan E13); one answer makes the day not unanswered. */
fun MoodDay.isUnanswered(): Boolean = checkIns >= 1 && answered == 0

/** A low day: the person said so, or every check-in that day went unanswered. */
fun MoodDay.isLow(): Boolean = lowConfirmed || isUnanswered()

/** Low days among the last `days` days, today included. */
fun lowDayCount(
    store: MoodStore,
    today: LocalDate,
    days: Int,
): Int {
    val first: LocalDate = today.minusDays(days.toLong() - 1)
    return store.days().count { !it.date.isBefore(first) && !it.date.isAfter(today) && it.isLow() }
}

/** The newest date whose day satisfies `pick`, or null. */
fun lastDate(
    store: MoodStore,
    pick: (MoodDay) -> Boolean,
): LocalDate? = store.days().filter(pick).maxOfOrNull { it.date }

/** Deletes days older than `keepDays` and face counts older than `baselineDays` (design plan E13). */
fun deleteOld(
    store: MoodStore,
    today: LocalDate,
    s: LevelSettings,
) {
    store.deleteDaysBefore(today.minusDays(s.keepDays.toLong()))
    store.deleteCountsBefore(today.minusDays(s.baselineDays.toLong()))
}
