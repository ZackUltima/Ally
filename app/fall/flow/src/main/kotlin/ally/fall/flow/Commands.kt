package ally.fall.flow

import kotlin.time.Duration

/** Every stage the flow passes through; Escalated, Handled and Cancelled never wait (technical plan section 6). */
enum class FlowStage { WATCHING, CHECK, ESCALATED, CALLING, HANDLED, CANCELLED, DOWN_WATCH, CHECK_IN }

/** The event code an alert carries (part 2 section E7). */
enum class EventCode { FALL, HELP_REQUEST }

/** Ids of design plan section 8's exact lines; the words live in F4's wording table. */
enum class Line {
    CHECK_FIRST,
    CHECK_SECOND,
    CANCELLED,
    CALLING,
    STILL_CALLING,
    ACKNOWLEDGED,
    HELP_IS_HERE,
    ALL_CLEAR,
    CHECK_IN_FIRST,
    CHECK_IN_SECOND,
}

sealed interface GuardianMessage

/** Someone is handling the alert; [by] is null after Help is here. */
data class Handled(
    val by: GuardianId?,
) : GuardianMessage

data object AllClear : GuardianMessage

/** A low-priority note that a possible fall was detected and cancelled (D-22). */
data object QuietNote : GuardianMessage

/** What the flow asks the adaptor to do; the flow never talks to a channel itself (T8). */
sealed interface FlowCommand

/** One line of the event log for every stage entered. */
data class Log(
    val stage: FlowStage,
    val at: Duration,
    val by: GuardianId? = null,
) : FlowCommand

data class Say(
    val line: Line,
    val guardian: GuardianId? = null,
) : FlowCommand

/** The alarm, the push and the snapshot, at the same moment (section 7). */
data class StartAlert(
    val code: EventCode,
) : FlowCommand

data class PlaceCall(
    val guardian: GuardianId,
) : FlowCommand

data object PauseAlarm : FlowCommand

data object ResumeAlarm : FlowCommand

/** Calls and alarm stop. */
data object StopAlert : FlowCommand

data class Notify(
    val message: GuardianMessage,
    val to: List<GuardianId>,
) : FlowCommand
