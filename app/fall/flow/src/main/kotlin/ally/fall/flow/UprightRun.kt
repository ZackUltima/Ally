package ally.fall.flow

import kotlin.time.Duration

/**
 * The off-the-ground test for one person: standing or walking for [hold] without a break (D-55 item 7).
 * Any other posture, or a tick with no pose, breaks the run (D-64 decision 5).
 */
class UprightRun(
    private val hold: Duration,
) {
    private var since: Duration? = null

    var isOffGround: Boolean = false
        private set

    fun update(
        at: Duration,
        posture: Posture?,
    ) {
        if (posture == Posture.UPRIGHT) {
            val start = since ?: at
            since = start
            isOffGround = at - start >= hold
        } else {
            since = null
            isOffGround = false
        }
    }
}
