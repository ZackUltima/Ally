package ally.fall.flow

import kotlin.time.Duration

/**
 * Who is called next, and when (D-22 calling pattern): guardians in order, [backToBackRounds] rounds back to
 * back, then each later round starts [laterRoundGap] after the end of the previous one (D-64 decision 2).
 */
class CallRota(
    guardians: Guardians,
    private val backToBackRounds: Int,
    private val laterRoundGap: Duration,
) {
    sealed interface Next {
        data class Call(
            val guardian: GuardianId,
        ) : Next

        data class WaitUntil(
            val at: Duration,
            val guardian: GuardianId,
        ) : Next
    }

    private val order: List<GuardianId> = guardians.inOrder.map { it.id }
    private var placed = 0

    fun first(): GuardianId {
        placed = 1
        return order.first()
    }

    /** The call before ended at [at]; says who is next and whether the next round must wait. */
    fun afterEnded(at: Duration): Next {
        val round = placed / order.size
        val position = placed % order.size
        placed += 1
        val guardian = order[position]
        val startsLaterRound = position == 0 && round >= backToBackRounds
        return if (startsLaterRound) Next.WaitUntil(at + laterRoundGap, guardian) else Next.Call(guardian)
    }
}
