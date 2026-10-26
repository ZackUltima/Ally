package ally.emotion.levels

import java.time.LocalDateTime

/** The stages that wait for the person. */
sealed interface LevelState {
    data object Idle : LevelState

    /** "You seem a bit quiet today. How are you feeling?", asked at [since]. */
    data class CheckIn(
        val since: LocalDateTime,
    ) : LevelState

    /** "It sounds like you're feeling low today. Is that right?", asked at [since]; [fromCheckIn] if a check-in led here. */
    data class LowDayQuestion(
        val since: LocalDateTime,
        val askedAgain: Boolean,
        val fromCheckIn: Boolean,
    ) : LevelState

    /**
     * "Can I ask you two quick questions?", asked at [since]; [askedAgain] once it was repeated, and [silentOnce] if
     * the first asking got no reply at all (EM2a decision 4).
     */
    data class PhqLeadIn(
        val since: LocalDateTime,
        val askedAgain: Boolean,
        val silentOnce: Boolean,
    ) : LevelState

    /** PHQ-2 item [number] (1 or 2), asked at [since]; [repeated] once the four answers were read again. */
    data class PhqItem(
        val number: Int,
        val since: LocalDateTime,
        val repeated: Boolean,
        val first: PhqAnswer?,
    ) : LevelState
}
