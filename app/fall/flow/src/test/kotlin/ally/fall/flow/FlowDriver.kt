package ally.fall.flow

import kotlin.time.Duration
import kotlin.time.Duration.Companion.milliseconds

val P1 = PersonId(1)
val P2 = PersonId(2)
val G1 = GuardianId(1)
val G2 = GuardianId(2)
val G3 = GuardianId(3)

/** Guardian 1 has no guardian app; 2 and 3 have it. */
val THREE_GUARDIANS =
    Guardians(
        listOf(
            Guardian(G1, "Mei", hasApp = false),
            Guardian(G2, "Ahmad", hasApp = true),
            Guardian(G3, "Siti", hasApp = true),
        ),
    )

/** Test helper: drives a [FallFlow] with a fake clock and keeps every command with its time. */
class FlowDriver(
    val settings: FlowSettings = FlowSettings(quietNoteOn = false),
    guardians: Guardians = THREE_GUARDIANS,
) {
    val flow = FallFlow(settings, guardians)
    var now: Duration = Duration.ZERO
        private set
    val commands = mutableListOf<FlowCommand>()
    val timed = mutableListOf<Pair<Duration, FlowCommand>>()

    /** Called after every event with the state before, the state after and the commands. */
    var observer: ((FlowState, FlowState, List<FlowCommand>) -> Unit)? = null

    fun send(event: FlowEvent): List<FlowCommand> {
        now = event.at
        val before = flow.state
        val out = flow.handle(event)
        commands += out
        timed += out.map { event.at to it }
        observer?.invoke(before, flow.state, out)
        return out
    }

    /** One pose tick every 0.1 s for [duration], the first 0.1 s after now. */
    fun poses(
        person: PersonId,
        posture: Posture,
        duration: Duration,
    ): List<FlowCommand> = posesOf(mapOf(person to posture), duration)

    fun posesOf(
        postures: Map<PersonId, Posture>,
        duration: Duration,
    ): List<FlowCommand> = ticks(duration, TICK) { PoseTick(it, postures) }

    /** Pose ticks in which nobody has a pose. */
    fun noPose(duration: Duration): List<FlowCommand> = posesOf(emptyMap(), duration)

    /** Clock ticks every 0.5 s and no pose ticks, as when the camera has stopped. */
    fun wait(duration: Duration): List<FlowCommand> = ticks(duration, CLOCK) { ClockTick(it) }

    fun say(
        reply: Reply,
        via: Via = Via.VOICE,
    ): List<FlowCommand> = send(Answer(now, reply, via))

    fun connected(guardian: GuardianId): List<FlowCommand> = send(CallConnected(now, guardian))

    fun ended(guardian: GuardianId): List<FlowCommand> = send(CallEnded(now, guardian))

    fun acknowledged(guardian: GuardianId): List<FlowCommand> = send(AppAcknowledged(now, guardian))

    fun helpIsHere(): List<FlowCommand> = send(HelpIsHere(now))

    /** Times at which [command] was given. */
    fun timesOf(command: FlowCommand): List<Duration> = timed.filter { it.second == command }.map { it.first }

    private fun ticks(
        duration: Duration,
        step: Duration,
        make: (Duration) -> FlowEvent,
    ): List<FlowCommand> {
        val out = mutableListOf<FlowCommand>()
        val end = now + duration
        var t = now
        while (t < end) {
            t = minOf(t + step, end)
            out += send(make(t))
        }
        return out
    }

    companion object {
        val TICK: Duration = 100.milliseconds
        val CLOCK: Duration = 500.milliseconds
    }
}
