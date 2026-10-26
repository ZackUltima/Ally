package ally.fall.flow

import kotlin.time.Duration

/** A tracked person's ID from the pose tracker (D-33). */
@JvmInline
value class PersonId(
    val value: Int,
)

/** What the phrase checker or a button made of the person's answer (D-22 rule 2). */
enum class Reply { OKAY, HELP, UNCLEAR }

enum class Via { VOICE, TAP }

/** One input to the flow. [at] is time since an arbitrary start, never wall-clock; it must not go back. */
sealed interface FlowEvent {
    val at: Duration
}

/** One 0.1 s tick of the fall model; a person missing from [postures] has no pose on this tick. */
data class PoseTick(
    override val at: Duration,
    val postures: Map<PersonId, Posture>,
) : FlowEvent

/** The service's own timer, so timers fire even when no pose ticks arrive (the camera has stopped). */
data class ClockTick(
    override val at: Duration,
) : FlowEvent

data class Answer(
    override val at: Duration,
    val reply: Reply,
    val via: Via,
) : FlowEvent

data class CallConnected(
    override val at: Duration,
    val guardian: GuardianId,
) : FlowEvent

data class CallEnded(
    override val at: Duration,
    val guardian: GuardianId,
) : FlowEvent

/** A tap on "I'm handling it" in the guardian app. */
data class AppAcknowledged(
    override val at: Duration,
    val guardian: GuardianId,
) : FlowEvent

/** A tap on Help is here on Ally's screen by someone who has arrived. */
data class HelpIsHere(
    override val at: Duration,
) : FlowEvent
