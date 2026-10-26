package ally.emotion.levels

import ally.fall.flow.GuardianId
import ally.fall.flow.Guardians
import java.time.LocalDate
import java.time.LocalDateTime
import java.time.temporal.ChronoUnit
import kotlin.time.toJavaDuration

/** Shares are fractions of at most a few hundred readings; this keeps 13/20 reaching 0.45 + 0.20. */
private const val SHARE_TOLERANCE: Double = 1e-9

/** The PHQ-2 total from which the level 2 note goes to the guardian (D-23). */
private const val NOTE_TOTAL: Int = 3

/**
 * The emotion level rules as a pure reducer (technical plan section E6, T13): one event in, commands out. It never
 * talks to the screen, the voice or the backend itself. Level 3 never passes through here; it goes to the fall flow.
 */
class LevelRules(
    private val settings: LevelSettings,
    private val guardians: Guardians,
    private val store: MoodStore,
    consent: Boolean,
    noteOn: Boolean,
    start: LocalDateTime,
) {
    var state: LevelState = LevelState.Idle
        private set

    private var consentOn: Boolean = consent
    private var noteSwitchOn: Boolean = noteOn
    private var lastAt: LocalDateTime = start
    private var today: LocalDate = start.toLocalDate()
    private var people: Int = 0
    private var higherBusy: Boolean = false
    private var inConversation: Boolean = false
    private var lastContact: LocalDateTime? = null
    private var lowDayDeclined: LocalDate? = null

    /** The PHQ-2 is due (a day started eligible, or one was stopped part-way) and waits for a free moment. */
    private var phqPending: Boolean = false
    private val window: ReadingWindow = ReadingWindow(settings.window)

    fun handle(event: LevelEvent): List<LevelCommand> {
        val out: MutableList<LevelCommand> = mutableListOf()
        val at: LocalDateTime = timeOf(event, out)
        val dayStarted: Boolean = startNewDay(at)
        expire(at, out)
        when (event) {
            is Reading -> onReading(event.expression, at)
            is PeopleInView -> onPeople(event.count, out)
            is Tick -> Unit
            is CheckInReply -> onCheckInReply(event.soundsLow, at, out)
            is LowDaySuggested -> onSuggested(at, out)
            is YesNo -> onYesNo(event.answer, at, out)
            is PhqReply -> onPhqReply(event.answer, at, out)
            is Greeting -> lastContact = at
            is Busy -> onBusy(event)
            is ConsentChanged -> onConsent(event.on)
            is NoteSwitchChanged -> noteSwitchOn = event.on
            is ForgetDiary -> store.clear()
        }
        if (!dayStarted) offerWaitingPhq(at, out)
        faceTrigger(at, out)
        return out
    }

    /** An event earlier than the last one is treated as the last time and reported, never swallowed (Review Focus 1). */
    private fun timeOf(
        event: LevelEvent,
        out: MutableList<LevelCommand>,
    ): LocalDateTime {
        if (event.at.isBefore(lastAt)) {
            out += ClockWentBack(lastAt, event.at)
            return lastAt
        }
        lastAt = event.at
        return event.at
    }

    /**
     * The first event of a new day deletes what is too old, whatever happened in between (Review Focus 2), and marks
     * the PHQ-2 as due if the day starts eligible. Returns true for that event, which never makes the offer itself.
     */
    private fun startNewDay(at: LocalDateTime): Boolean {
        val date: LocalDate = at.toLocalDate()
        if (date == today) return false
        today = date
        deleteOld(store, date, settings)
        phqPending = consentOn && phqEligible()
        return true
    }

    private fun expire(
        at: LocalDateTime,
        out: MutableList<LevelCommand>,
    ) {
        val since: LocalDateTime = state.waitingSince() ?: return
        if (at.isBefore(since.plusWait())) return
        when (val s = state) {
            is LevelState.CheckIn -> {
                logCheckIn(answered = false)
                state = LevelState.Idle
                out += Say(Line.CHECK_IN_NO_ANSWER)
            }

            is LevelState.LowDayQuestion -> {
                askAgainOrStop(s, at, out)
            }

            is LevelState.PhqLeadIn -> {
                leadInUnclear(s, silent = true, at, out)
            }

            is LevelState.PhqItem -> {
                itemUnclear(s, at, out)
            }

            LevelState.Idle -> {
                // nothing is waiting, and waitingSince() already returned
            }
        }
    }

    private fun LevelState.waitingSince(): LocalDateTime? =
        when (this) {
            is LevelState.CheckIn -> since
            is LevelState.LowDayQuestion -> since
            is LevelState.PhqLeadIn -> since
            is LevelState.PhqItem -> since
            LevelState.Idle -> null
        }

    private fun LocalDateTime.plusWait(): LocalDateTime = plus(settings.replyWait.toJavaDuration())

    private fun onReading(
        expression: Expression?,
        at: LocalDateTime,
    ) {
        if (!consentOn) return
        window.add(at, expression)
        if (expression == null) return
        val counts: FaceCounts = store.counts(today)
        store.putCounts(
            counts.copy(
                valid = counts.valid + 1,
                negative = counts.negative + if (expression.isNegative) 1 else 0,
            ),
        )
    }

    private fun onPeople(
        count: Int,
        out: MutableList<LevelCommand>,
    ) {
        people = count
        if (count > 1 && state != LevelState.Idle) {
            stop()
            out += Say(Line.TALK_LATER)
        }
    }

    private fun onBusy(event: Busy) {
        when (event.flow) {
            BusyFlow.HIGHER -> {
                higherBusy = event.running
                if (event.running) stop()
            }

            BusyFlow.CONVERSATION -> {
                inConversation = event.running
            }
        }
    }

    /** Drops whatever is being asked. A PHQ-2 stopped part-way is not counted and is offered again (decision 6). */
    private fun stop() {
        if (state is LevelState.PhqLeadIn || state is LevelState.PhqItem) phqPending = true
        state = LevelState.Idle
    }

    private fun onConsent(on: Boolean) {
        consentOn = on
        if (on) return
        state = LevelState.Idle
        window.clear()
        store.clear()
        lastContact = null
        lowDayDeclined = null
        phqPending = false
    }

    private fun onCheckInReply(
        soundsLow: Boolean,
        at: LocalDateTime,
        out: MutableList<LevelCommand>,
    ) {
        if (state !is LevelState.CheckIn) return
        logCheckIn(answered = true)
        state = LevelState.Idle
        if (!soundsLow) {
            out += Say(Line.CHECK_IN_FINE)
            offerPhqNow(at, out)
        } else if (lowDayAllowed()) {
            askLowDay(at, fromCheckIn = true, out)
        }
    }

    private fun onSuggested(
        at: LocalDateTime,
        out: MutableList<LevelCommand>,
    ) {
        val free: Boolean = consentOn && state == LevelState.Idle && !higherBusy && people <= 1
        if (free && lowDayAllowed()) askLowDay(at, fromCheckIn = false, out)
    }

    private fun onYesNo(
        answer: YesNoAnswer,
        at: LocalDateTime,
        out: MutableList<LevelCommand>,
    ) {
        when (val s = state) {
            is LevelState.LowDayQuestion -> onLowDayAnswer(s, answer, at, out)
            is LevelState.PhqLeadIn -> onLeadInAnswer(s, answer, at, out)
            is LevelState.CheckIn, is LevelState.PhqItem, LevelState.Idle -> Unit
        }
    }

    private fun onLowDayAnswer(
        s: LevelState.LowDayQuestion,
        answer: YesNoAnswer,
        at: LocalDateTime,
        out: MutableList<LevelCommand>,
    ) {
        when (answer) {
            YesNoAnswer.YES -> {
                store.putDay(store.day(today).copy(lowConfirmed = true))
                state = LevelState.Idle
                offerPhqNow(at, out)
            }

            YesNoAnswer.NO -> {
                lowDayDeclined = today
                state = LevelState.Idle
            }

            YesNoAnswer.UNCLEAR -> {
                askAgainOrStop(s, at, out)
            }
        }
    }

    /** Unclear or silent: ask once more, then nothing; the question is not raised again that day (choice 28). */
    private fun askAgainOrStop(
        s: LevelState.LowDayQuestion,
        at: LocalDateTime,
        out: MutableList<LevelCommand>,
    ) {
        if (s.askedAgain) {
            state = LevelState.Idle
            lowDayDeclined = today
        } else {
            state = s.copy(since = at, askedAgain = true)
            out += Say(Line.LOW_DAY_QUESTION)
        }
    }

    private fun lowDayAllowed(): Boolean = !store.day(today).lowConfirmed && lowDayDeclined != today

    private fun askLowDay(
        at: LocalDateTime,
        fromCheckIn: Boolean,
        out: MutableList<LevelCommand>,
    ) {
        state = LevelState.LowDayQuestion(at, askedAgain = false, fromCheckIn = fromCheckIn)
        out += Say(Line.LOW_DAY_QUESTION)
    }

    private fun logCheckIn(answered: Boolean) {
        val day: MoodDay = store.day(today)
        store.putDay(day.copy(checkIns = day.checkIns + 1, answered = day.answered + if (answered) 1 else 0))
    }

    /** Eligible: at least `phqLowDays` low days in the last `phqWindowDays` (today included), none offered lately. */
    private fun phqEligible(): Boolean {
        if (lowDayCount(store, today, settings.phqWindowDays) < settings.phqLowDays) return false
        val last: LocalDate = lastDate(store) { it.phqOffered } ?: return true
        return ChronoUnit.DAYS.between(last, today) >= settings.phqGapDays
    }

    /** Right after a confirmed low day (a) or an answered check-in (b): offered if eligible and the person is alone. */
    private fun offerPhqNow(
        at: LocalDateTime,
        out: MutableList<LevelCommand>,
    ) {
        if (consentOn && !higherBusy && people <= 1 && phqEligible()) startLeadIn(at, out)
    }

    /** (c) A PHQ-2 that is due waits for exactly one person in view, nothing busy and nothing being asked. */
    private fun offerWaitingPhq(
        at: LocalDateTime,
        out: MutableList<LevelCommand>,
    ) {
        if (!phqPending || !consentOn || state != LevelState.Idle || higherBusy || inConversation || people != 1) return
        if (phqEligible()) startLeadIn(at, out) else phqPending = false
    }

    private fun startLeadIn(
        at: LocalDateTime,
        out: MutableList<LevelCommand>,
    ) {
        state = LevelState.PhqLeadIn(at, askedAgain = false, silentOnce = false)
        out += Say(Line.PHQ_LEAD_IN)
    }

    private fun onLeadInAnswer(
        s: LevelState.PhqLeadIn,
        answer: YesNoAnswer,
        at: LocalDateTime,
        out: MutableList<LevelCommand>,
    ) {
        when (answer) {
            YesNoAnswer.YES -> {
                state = LevelState.PhqItem(1, at, repeated = false, first = null)
                out += Say(Line.PHQ_ITEM_1)
            }

            YesNoAnswer.NO -> {
                notNow()
            }

            YesNoAnswer.UNCLEAR -> {
                leadInUnclear(s, silent = false, at, out)
            }
        }
    }

    /**
     * Unclear or silent on the lead-in: asked once more. Then two silences are "no reply at all" (the no-answer note);
     * anything else is treated as Not now (EM2a decision 4).
     */
    private fun leadInUnclear(
        s: LevelState.PhqLeadIn,
        silent: Boolean,
        at: LocalDateTime,
        out: MutableList<LevelCommand>,
    ) {
        if (!s.askedAgain) {
            state = LevelState.PhqLeadIn(at, askedAgain = true, silentOnce = silent)
            out += Say(Line.PHQ_LEAD_IN)
        } else if (silent && s.silentOnce) {
            countOffer()
            state = LevelState.Idle
            sendNote(NoteKind.LOW_MOOD_NO_ANSWER, out)
        } else {
            notNow()
        }
    }

    /** The person chose not to: the offer counts, and the next one waits `phqGapDays`. */
    private fun notNow() {
        countOffer()
        state = LevelState.Idle
    }

    private fun countOffer() {
        store.putDay(store.day(today).copy(phqOffered = true))
        phqPending = false
    }

    private fun onPhqReply(
        answer: PhqAnswer?,
        at: LocalDateTime,
        out: MutableList<LevelCommand>,
    ) {
        val s: LevelState.PhqItem = state as? LevelState.PhqItem ?: return
        if (answer == null) itemUnclear(s, at, out) else itemScored(s, answer, at, out)
    }

    private fun itemScored(
        s: LevelState.PhqItem,
        answer: PhqAnswer,
        at: LocalDateTime,
        out: MutableList<LevelCommand>,
    ) {
        if (s.number == 1) nextItem(answer, at, out) else finish(Phq2Answers(s.first, answer), out)
    }

    /** An unclear item has its answers read once more, then it is left unscored (D-23 rule 6). */
    private fun itemUnclear(
        s: LevelState.PhqItem,
        at: LocalDateTime,
        out: MutableList<LevelCommand>,
    ) {
        if (!s.repeated) {
            state = s.copy(since = at, repeated = true)
            out += Say(Line.PHQ_OPTIONS_AGAIN)
        } else if (s.number == 1) {
            nextItem(null, at, out)
        } else {
            finish(Phq2Answers(s.first, null), out)
        }
    }

    private fun nextItem(
        first: PhqAnswer?,
        at: LocalDateTime,
        out: MutableList<LevelCommand>,
    ) {
        state = LevelState.PhqItem(2, at, repeated = false, first = first)
        out += Say(Line.PHQ_ITEM_2)
    }

    private fun finish(
        answers: Phq2Answers,
        out: MutableList<LevelCommand>,
    ) {
        store.putDay(store.day(today).copy(phq = answers))
        countOffer()
        state = LevelState.Idle
        out += Say(Line.PHQ_CLOSING)
        val total: Int? = answers.total
        if (total != null && total >= NOTE_TOTAL) sendNote(NoteKind.LOW_MOOD, out)
    }

    /** At most one note a week, only with the switch on, to the first guardian who has the app. */
    private fun sendNote(
        kind: NoteKind,
        out: MutableList<LevelCommand>,
    ) {
        if (!noteSwitchOn) return
        val last: LocalDate? = lastDate(store) { it.noteSent }
        if (last != null && ChronoUnit.DAYS.between(last, today) < settings.noteGapDays) return
        val to: GuardianId = guardians.withApp.first().id
        store.putDay(store.day(today).copy(noteSent = true))
        out += Say(Line.NOTE_BEING_SENT, to)
        out += SendNote(kind, to)
    }

    /** Checked after every event: 20 readings in 10 minutes, a high enough negative share, one person, 2 h since the last. */
    private fun faceTrigger(
        at: LocalDateTime,
        out: MutableList<LevelCommand>,
    ) {
        if (!consentOn || state != LevelState.Idle || higherBusy || inConversation || people != 1) return
        if (window.validCount(at) < settings.minValid) return
        if (lastContact?.let { at.isBefore(it.plus(settings.faceGap.toJavaDuration())) } == true) return
        val lowDays: Set<LocalDate> =
            store
                .days()
                .filter { it.isLow() }
                .map { it.date }
                .toSet()
        val baseline: Double? = baselineShare(store.allCounts(), lowDays, today, settings)
        if (window.negativeShare(at) + SHARE_TOLERANCE < triggerShare(baseline, settings)) return
        state = LevelState.CheckIn(at)
        lastContact = at
        out += Say(Line.CHECK_IN)
    }
}
