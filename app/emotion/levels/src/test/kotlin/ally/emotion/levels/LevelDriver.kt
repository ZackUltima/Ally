package ally.emotion.levels

import ally.fall.flow.Guardian
import ally.fall.flow.GuardianId
import ally.fall.flow.Guardians
import java.time.LocalDateTime
import kotlin.time.Duration
import kotlin.time.Duration.Companion.hours
import kotlin.time.Duration.Companion.seconds
import kotlin.time.toJavaDuration

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

val START: LocalDateTime = LocalDateTime.of(2026, 11, 15, 9, 0)

/** The line ids said in these commands, in order. */
fun List<LevelCommand>.lines(): List<Line> = filterIsInstance<Say>().map { it.line }

/** Test helper: drives [LevelRules] with a fake clock. Every call moves time on and returns the commands it caused. */
class LevelDriver(
    val settings: LevelSettings = LevelSettings(),
    guardians: Guardians = THREE_GUARDIANS,
    val store: InMemoryMoodStore = InMemoryMoodStore(),
    consent: Boolean = true,
    noteOn: Boolean = true,
    people: Int = 1,
) {
    val rules = LevelRules(settings, guardians, store, consent, noteOn, START)
    var now: LocalDateTime = START
        private set
    val commands = mutableListOf<LevelCommand>()

    init {
        if (people != 0) people(people)
    }

    val today get() = now.toLocalDate()

    fun send(event: LevelEvent): List<LevelCommand> {
        now = maxOf(now, event.at)
        val out = rules.handle(event)
        commands += out
        return out
    }

    /** [count] readings, the first [every] after now; a null expression is "no reading". */
    fun readings(
        expression: Expression?,
        count: Int,
        every: Duration = 10.seconds,
    ): List<LevelCommand> {
        val out = mutableListOf<LevelCommand>()
        repeat(count) {
            out += send(Reading(advance(every), expression))
        }
        return out
    }

    fun people(n: Int): List<LevelCommand> = send(PeopleInView(advance(1.seconds), n))

    /** A tick every 5 s for [duration]. */
    fun wait(duration: Duration): List<LevelCommand> {
        val out = mutableListOf<LevelCommand>()
        var left: Duration = duration
        while (left > Duration.ZERO) {
            val step: Duration = minOf(left, 5.seconds)
            out += send(Tick(advance(step)))
            left -= step
        }
        return out
    }

    /** One tick on the next date at [at] o'clock. */
    fun nextDay(at: Duration = 9.hours): List<LevelCommand> {
        val midnight: LocalDateTime = now.toLocalDate().plusDays(1).atStartOfDay()
        return send(Tick(midnight.plus(at.toJavaDuration())))
    }

    fun reply(soundsLow: Boolean): List<LevelCommand> = send(CheckInReply(advance(1.seconds), soundsLow))

    fun yesNo(answer: YesNoAnswer): List<LevelCommand> = send(YesNo(advance(1.seconds), answer))

    fun phq(answer: PhqAnswer?): List<LevelCommand> = send(PhqReply(advance(1.seconds), answer))

    fun busy(
        flow: BusyFlow,
        running: Boolean,
    ): List<LevelCommand> = send(Busy(advance(1.seconds), flow, running))

    /** Marks the days [back] days before today as confirmed low days. */
    fun lowDays(vararg back: Long) {
        back.forEach { store.putDay(MoodDay(today.minusDays(it), lowConfirmed = true)) }
    }

    fun suggest(): List<LevelCommand> = send(LowDaySuggested(advance(1.seconds)))

    fun greeting(): List<LevelCommand> = send(Greeting(advance(1.seconds)))

    fun consent(on: Boolean): List<LevelCommand> = send(ConsentChanged(advance(1.seconds), on))

    fun forget(): List<LevelCommand> = send(ForgetDiary(advance(1.seconds)))

    /** Enough negative readings in the window for the face trigger (12 negative, 8 neutral). */
    fun lowFace(): List<LevelCommand> = readings(Expression.NEUTRAL, 8) + readings(Expression.SADNESS, 12)

    private fun advance(by: Duration): LocalDateTime {
        now = now.plus(by.toJavaDuration())
        return now
    }
}
