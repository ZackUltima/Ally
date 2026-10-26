package ally.emotion.levels

import kotlin.time.Duration
import kotlin.time.Duration.Companion.hours
import kotlin.time.Duration.Companion.minutes
import kotlin.time.Duration.Companion.seconds

/** Windows, shares and day counts of the emotion level rules; starting values from D-23 and EM2a decisions 3 and 4. */
data class LevelSettings(
    val window: Duration = 10.minutes,
    val minValid: Int = 20,
    val negativeShare: Double = 0.60,
    val faceGap: Duration = 2.hours,
    val replyWait: Duration = 15.seconds,
    val baselineDays: Int = 14,
    val baselineMinDays: Int = 7,
    val baselineMargin: Double = 0.20,
    val phqLowDays: Int = 3,
    val phqWindowDays: Int = 7,
    val phqGapDays: Int = 14,
    val noteGapDays: Int = 7,
    val keepDays: Int = 30,
) {
    init {
        require(minValid > 0) { "minValid must be above 0, got $minValid" }
        require(baselineMinDays <= baselineDays) {
            "the baseline needs $baselineMinDays days but looks back over only $baselineDays"
        }
    }
}
