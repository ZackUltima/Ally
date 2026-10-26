package ally.emotion.levels

import java.time.LocalDateTime

/** Answers to a yes or no question; NO also means "Not now" (the screen's button). */
enum class YesNoAnswer { YES, NO, UNCLEAR }

/** Which busier flow started or ended: a fall, help request or crisis (HIGHER), or a free conversation. */
enum class BusyFlow { HIGHER, CONVERSATION }

/** What the rules are told. Every event carries the phone's local date and time (EM2a decision 2). */
sealed interface LevelEvent {
    val at: LocalDateTime
}

/** One expression reading; a null expression is "no reading" and never counts (D-23 rule 3). */
data class Reading(
    override val at: LocalDateTime,
    val expression: Expression?,
) : LevelEvent

/** How many people the camera sees now. */
data class PeopleInView(
    override val at: LocalDateTime,
    val count: Int,
) : LevelEvent

data class Tick(
    override val at: LocalDateTime,
) : LevelEvent

/** The reply to a check-in; the phrase checker or the language model decides [soundsLow]. */
data class CheckInReply(
    override val at: LocalDateTime,
    val soundsLow: Boolean,
) : LevelEvent

/** Something said in a conversation sounds like a low day, outside a check-in. */
data class LowDaySuggested(
    override val at: LocalDateTime,
) : LevelEvent

data class YesNo(
    override val at: LocalDateTime,
    val answer: YesNoAnswer,
) : LevelEvent

/** The answer to a PHQ-2 item; null is unclear. */
data class PhqReply(
    override val at: LocalDateTime,
    val answer: PhqAnswer?,
) : LevelEvent

/** Ally greeted the person on its own (it starts the same 2 h gap as a face check-in). */
data class Greeting(
    override val at: LocalDateTime,
) : LevelEvent

/** A busier flow started (`running`) or ended. */
data class Busy(
    override val at: LocalDateTime,
    val flow: BusyFlow,
    val running: Boolean,
) : LevelEvent

data class ConsentChanged(
    override val at: LocalDateTime,
    val on: Boolean,
) : LevelEvent

data class NoteSwitchChanged(
    override val at: LocalDateTime,
    val on: Boolean,
) : LevelEvent

/** "Forget my mood diary". */
data class ForgetDiary(
    override val at: LocalDateTime,
) : LevelEvent
