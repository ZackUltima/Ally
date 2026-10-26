package ally.fall.flow

import kotlin.time.Duration
import kotlin.time.Duration.Companion.minutes
import kotlin.time.Duration.Companion.seconds

/** Timers and switches of the fall flow; starting values from D-22, D-55 and D-64. */
data class FlowSettings(
    val secondAsk: Duration = 15.seconds,
    val checkTimeout: Duration = 30.seconds,
    val offGroundHold: Duration = 3.seconds,
    val checkInEvery: Duration = 2.minutes,
    val checkInSecondAsk: Duration = 15.seconds,
    val checkInTimeout: Duration = 30.seconds,
    val backToBackRounds: Int = 3,
    val laterRoundGap: Duration = 10.minutes,
    val callSafetyTimeout: Duration = 90.seconds,
    val quietNoteOn: Boolean,
)

@JvmInline
value class GuardianId(
    val value: Int,
)

data class Guardian(
    val id: GuardianId,
    val name: String,
    val hasApp: Boolean,
)

/** One to three guardians in calling order, at least one with the guardian app (design choice 13). */
class Guardians(
    list: List<Guardian>,
) {
    val inOrder: List<Guardian> = list.toList()
    val withApp: List<Guardian> = inOrder.filter { it.hasApp }

    init {
        require(inOrder.size in 1..3) { "1 to 3 guardians are needed, got ${inOrder.size}" }
        require(inOrder.map { it.id }.toSet().size == inOrder.size) { "guardian ids must be distinct" }
        require(withApp.isNotEmpty()) { "at least one guardian must have the guardian app" }
    }
}
