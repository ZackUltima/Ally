package ally.fall.flow

import kotlin.time.Duration

/** The face the screen layer draws (design plan section 4). The relieved face is drawn in F4 from the Log commands. */
enum class Face { WATCHING, ALERT, CALLING }

enum class Button { OKAY, HELP, HELP_IS_HERE }

/** One screen state for the face and screen layer (technical plan section 8). */
data class FlowScreen(
    val face: Face,
    val buttons: Set<Button>,
    val countdownEndsAt: Duration?,
    val calling: GuardianId?,
)

fun screenFor(
    state: FlowState,
    settings: FlowSettings,
): FlowScreen =
    when (state) {
        FlowState.Watching -> {
            FlowScreen(Face.WATCHING, emptySet(), null, null)
        }

        is FlowState.Check -> {
            FlowScreen(Face.ALERT, setOf(Button.OKAY, Button.HELP), state.since + settings.checkTimeout, null)
        }

        is FlowState.CheckIn -> {
            FlowScreen(Face.ALERT, setOf(Button.OKAY, Button.HELP), state.since + settings.checkInTimeout, null)
        }

        // Who is being called; nobody while a later round waits. I'm okay: a tap cancels at any time (PO 2026-10-24).
        is FlowState.Calling -> {
            val calling = if (state.placedAt != null) state.current else null
            FlowScreen(Face.CALLING, setOf(Button.OKAY, Button.HELP_IS_HERE), null, calling)
        }

        // D-64 decision 6: the Help button stays on screen while the person is down.
        is FlowState.DownWatch -> {
            FlowScreen(Face.WATCHING, setOf(Button.HELP), null, null)
        }
    }
