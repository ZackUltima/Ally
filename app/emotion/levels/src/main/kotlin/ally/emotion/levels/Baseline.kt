package ally.emotion.levels

import java.time.LocalDate

/** One day's face counts: valid readings and how many of them were negative. */
data class FaceCounts(
    val date: LocalDate,
    val valid: Int,
    val negative: Int,
)

/**
 * The person's own usual share of negative readings (EM2a decision 3), pooled over the qualifying days of the
 * `baselineDays` before `today`: not low, at least `minValid` readings, today left out. Null while fewer than
 * `baselineMinDays` days qualify, so the plain 60% rule applies.
 */
fun baselineShare(
    counts: List<FaceCounts>,
    lowDays: Set<LocalDate>,
    today: LocalDate,
    s: LevelSettings,
): Double? {
    require(counts.map { it.date }.toSet().size == counts.size) { "two rows for one day" }
    val first = today.minusDays(s.baselineDays.toLong())
    val days =
        counts.filter {
            !it.date.isBefore(first) && it.date.isBefore(today) && it.date !in lowDays && it.valid >= s.minValid
        }
    if (days.size < s.baselineMinDays) return null
    return days.sumOf { it.negative }.toDouble() / days.sumOf { it.valid }
}

/** The share the 10-minute window must reach: 60%, or the person's baseline plus 20 points if that is higher. */
fun triggerShare(
    baseline: Double?,
    s: LevelSettings,
): Double = if (baseline == null) s.negativeShare else maxOf(s.negativeShare, baseline + s.baselineMargin)
