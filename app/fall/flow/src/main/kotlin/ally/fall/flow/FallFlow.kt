package ally.fall.flow

import kotlin.time.Duration

/**
 * The fall flow as a pure reducer (technical plan section 6): each event, carrying its own time, goes in;
 * the commands to carry out come back. Timers are deadlines checked on every event, whatever its type.
 */
class FallFlow(
    private val settings: FlowSettings,
    private val guardians: Guardians,
) {
    var state: FlowState = FlowState.Watching
        private set

    private var lastAt: Duration = Duration.ZERO
    private var run: UprightRun? = null
    private var rota: CallRota? = null

    fun handle(event: FlowEvent): List<FlowCommand> {
        require(event.at >= lastAt) { "event at ${event.at} is earlier than the last one at $lastAt" }
        lastAt = event.at
        val out = mutableListOf<FlowCommand>()
        fireTimers(event.at, out)
        when (event) {
            is PoseTick -> onPoses(event, out)
            is Answer -> onAnswer(event, out)
            is CallConnected -> onConnected(event)?.let { out += it }
            is CallEnded -> onEnded(event, out)
            is AppAcknowledged -> onHandled(event.guardian, event.at, out)
            is HelpIsHere -> onHandled(null, event.at, out)
            is ClockTick -> Unit
        }
        return out
    }

    private fun fireTimers(
        at: Duration,
        out: MutableList<FlowCommand>,
    ) {
        when (val s = state) {
            is FlowState.Check -> {
                if (at >= s.since + settings.checkTimeout) {
                    escalate(EventCode.FALL, s.person, at, out)
                } else if (!s.askedAgain && at >= s.since + settings.secondAsk) {
                    state = s.copy(askedAgain = true)
                    out += Say(Line.CHECK_SECOND)
                }
            }

            is FlowState.Calling -> {
                val waited = s.waitUntil
                val placedAt = s.placedAt
                if (waited != null && at >= waited) {
                    s.current?.let { call(s, it, at, out) }
                } else if (placedAt != null && !s.connected && at >= placedAt + settings.callSafetyTimeout) {
                    nextCall(s, at, out) // D-64 decision 3: a call that never reports back
                }
            }

            is FlowState.DownWatch -> {
                if (at >= s.nextCheckInAt) {
                    state = FlowState.CheckIn(s.person, at, askedAgain = false)
                    out += Log(FlowStage.CHECK_IN, at)
                    out += Say(Line.CHECK_IN_FIRST)
                }
            }

            // D-64 decision 1: asked again at 15 s, escalated at 30 s, like the fall check.
            is FlowState.CheckIn -> {
                if (at >= s.since + settings.checkInTimeout) {
                    escalate(EventCode.FALL, s.person, at, out)
                } else if (!s.askedAgain && at >= s.since + settings.checkInSecondAsk) {
                    state = s.copy(askedAgain = true)
                    out += Say(Line.CHECK_IN_SECOND)
                }
            }

            FlowState.Watching -> {}
        }
    }

    private fun onPoses(
        tick: PoseTick,
        out: MutableList<FlowCommand>,
    ) {
        when (val s = state) {
            FlowState.Watching -> {
                val fallen =
                    tick.postures
                        .filterValues { it == Posture.FALL || it == Posture.FALLEN }
                        .keys
                        .minByOrNull { it.value }
                if (fallen != null) startCheck(fallen, tick.at, out)
            }

            is FlowState.Check -> {
                feedRun(s.person, tick)
                if (run?.isOffGround == true) cancel(tick.at, out)
            }

            // Standing up does not cancel once calling has started; the run only decides where to go after.
            is FlowState.Calling -> {
                s.person?.let { feedRun(it, tick) }
            }

            // New fall or fallen outputs are ignored here (one alert per fall); only the person who fell counts.
            is FlowState.DownWatch -> {
                feedRun(s.person, tick)
                if (run?.isOffGround == true) enterWatching(tick.at, out)
            }

            is FlowState.CheckIn -> {
                feedRun(s.person, tick)
                if (run?.isOffGround == true) enterWatching(tick.at, out)
            }
        }
    }

    private fun onAnswer(
        answer: Answer,
        out: MutableList<FlowCommand>,
    ) {
        when (val s = state) {
            is FlowState.Check -> {
                when (answer.reply) {
                    Reply.OKAY -> cancel(answer.at, out)
                    Reply.HELP -> escalate(EventCode.FALL, s.person, answer.at, out)
                    Reply.UNCLEAR -> Unit
                }
            }

            // A clear "I'm okay" cancels, except by voice while a guardian is on the line (section 6 change F).
            is FlowState.Calling -> {
                val heard = !(s.connected && answer.via == Via.VOICE)
                if (answer.reply == Reply.OKAY && heard) allClear(answer.at, out)
            }

            is FlowState.DownWatch -> {
                if (answer.reply == Reply.HELP) escalate(EventCode.FALL, s.person, answer.at, out)
            }

            is FlowState.CheckIn -> {
                when (answer.reply) {
                    Reply.OKAY -> enterDownWatch(s.person, answer.at, out)
                    Reply.HELP -> escalate(EventCode.FALL, s.person, answer.at, out)
                    Reply.UNCLEAR -> Unit
                }
            }

            // D-64 decision 4: only the phrase checker's level-3 match reaches here as HELP (T20).
            FlowState.Watching -> {
                if (answer.reply == Reply.HELP) escalate(EventCode.HELP_REQUEST, null, answer.at, out)
            }
        }
    }

    private fun onConnected(event: CallConnected): FlowCommand? {
        val s = state
        if (s !is FlowState.Calling || s.current != event.guardian || s.connected || s.placedAt == null) return null
        state = s.copy(connected = true)
        return PauseAlarm
    }

    private fun onEnded(
        event: CallEnded,
        out: MutableList<FlowCommand>,
    ) {
        val s = state
        if (s !is FlowState.Calling || s.current != event.guardian || s.placedAt == null) return // a late event
        if (s.connected) out += ResumeAlarm
        nextCall(s, event.at, out)
    }

    /** An app acknowledgement by [by], or Help is here when [by] is null. */
    private fun onHandled(
        by: GuardianId?,
        at: Duration,
        out: MutableList<FlowCommand>,
    ) {
        if (state !is FlowState.Calling) return
        out += StopAlert
        out += Log(FlowStage.HANDLED, at, by)
        if (by == null) out += Say(Line.HELP_IS_HERE) else out += Say(Line.ACKNOWLEDGED, by)
        val others = guardians.withApp.map { it.id }.filter { it != by }
        if (others.isNotEmpty()) out += Notify(Handled(by), others)
        afterAlert(at, out)
    }

    private fun feedRun(
        person: PersonId,
        tick: PoseTick,
    ) {
        run?.update(tick.at, tick.postures[person])
    }

    private fun startCheck(
        person: PersonId,
        at: Duration,
        out: MutableList<FlowCommand>,
    ) {
        run = UprightRun(settings.offGroundHold)
        state = FlowState.Check(person, at, askedAgain = false)
        out += Log(FlowStage.CHECK, at)
        out += Say(Line.CHECK_FIRST)
    }

    /** "I'm okay" or off the ground before any alert went out. */
    private fun cancel(
        at: Duration,
        out: MutableList<FlowCommand>,
    ) {
        out += Log(FlowStage.CANCELLED, at)
        out += Say(Line.CANCELLED)
        if (settings.quietNoteOn) out += Notify(QuietNote, guardians.withApp.map { it.id })
        afterAlert(at, out)
    }

    /** "I'm okay" after the alert went out: calls and alarm stop, every app guardian gets an all clear. */
    private fun allClear(
        at: Duration,
        out: MutableList<FlowCommand>,
    ) {
        out += StopAlert
        out += Log(FlowStage.CANCELLED, at)
        out += Say(Line.ALL_CLEAR)
        out += Notify(AllClear, guardians.withApp.map { it.id })
        afterAlert(at, out)
    }

    private fun escalate(
        code: EventCode,
        person: PersonId?,
        at: Duration,
        out: MutableList<FlowCommand>,
    ) {
        val newRota = CallRota(guardians, settings.backToBackRounds, settings.laterRoundGap)
        rota = newRota
        val first = newRota.first()
        out += Log(FlowStage.ESCALATED, at)
        out += StartAlert(code)
        state = FlowState.Calling(code, person, first, connected = false, placedAt = at, waitUntil = null)
        out += Log(FlowStage.CALLING, at)
        out += PlaceCall(first)
        out += Say(Line.CALLING, first)
    }

    private fun nextCall(
        s: FlowState.Calling,
        at: Duration,
        out: MutableList<FlowCommand>,
    ) {
        when (val next = checkNotNull(rota) { "calling without a rota" }.afterEnded(at)) {
            is CallRota.Next.Call -> {
                call(s, next.guardian, at, out)
            }

            is CallRota.Next.WaitUntil -> {
                state = s.copy(current = next.guardian, connected = false, placedAt = null, waitUntil = next.at)
            }
        }
    }

    private fun call(
        s: FlowState.Calling,
        guardian: GuardianId,
        at: Duration,
        out: MutableList<FlowCommand>,
    ) {
        state = s.copy(current = guardian, connected = false, placedAt = at, waitUntil = null)
        out += PlaceCall(guardian)
        out += Say(Line.STILL_CALLING, guardian)
    }

    /** After Handled or Cancelled: Down-watch while the person who fell is still down, otherwise Watching. */
    private fun afterAlert(
        at: Duration,
        out: MutableList<FlowCommand>,
    ) {
        rota = null
        val person = personOf(state)
        if (person != null && run?.isOffGround != true) enterDownWatch(person, at, out) else enterWatching(at, out)
    }

    private fun personOf(s: FlowState): PersonId? =
        when (s) {
            is FlowState.Check -> s.person
            is FlowState.Calling -> s.person
            is FlowState.DownWatch -> s.person
            is FlowState.CheckIn -> s.person
            FlowState.Watching -> null
        }

    private fun enterDownWatch(
        person: PersonId,
        at: Duration,
        out: MutableList<FlowCommand>,
    ) {
        state = FlowState.DownWatch(person, at + settings.checkInEvery)
        out += Log(FlowStage.DOWN_WATCH, at)
    }

    private fun enterWatching(
        at: Duration,
        out: MutableList<FlowCommand>,
    ) {
        run = null
        state = FlowState.Watching
        out += Log(FlowStage.WATCHING, at)
    }
}
