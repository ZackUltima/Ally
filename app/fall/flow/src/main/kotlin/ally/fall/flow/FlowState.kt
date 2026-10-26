package ally.fall.flow

import kotlin.time.Duration

/** The stages that wait for something (technical plan section 6). */
sealed interface FlowState {
    data object Watching : FlowState

    /** "Did you fall? Are you okay?", asked of [person] at [since]. */
    data class Check(
        val person: PersonId,
        val since: Duration,
        val askedAgain: Boolean,
    ) : FlowState

    /**
     * Guardians are being called for an alert with [code]; [person] is null for a help request from Watching.
     * [current] is the guardian being called, placed at [placedAt]; while a later round waits, [placedAt] is null
     * and [current] is the guardian who is called at [waitUntil].
     */
    data class Calling(
        val code: EventCode,
        val person: PersonId?,
        val current: GuardianId?,
        val connected: Boolean,
        val placedAt: Duration?,
        val waitUntil: Duration?,
    ) : FlowState

    /** [person] is still down after a cancel or an acknowledgement; the next check-in is at [nextCheckInAt]. */
    data class DownWatch(
        val person: PersonId,
        val nextCheckInAt: Duration,
    ) : FlowState

    /** "Are you still okay down there?", asked of [person] at [since]. */
    data class CheckIn(
        val person: PersonId,
        val since: Duration,
        val askedAgain: Boolean,
    ) : FlowState
}
