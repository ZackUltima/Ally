package ally.emotion.levels

/** The pixel face: watching while nothing is asked, caring while Ally asks something (design plan section E3). */
enum class Face { WATCHING, CARING }

/** The buttons under a question; PHQ_FOUR is the four answers of a PHQ-2 item. */
enum class Buttons { NONE, YES_NO, YES_NOT_NOW, PHQ_FOUR }

data class LevelScreen(
    val face: Face,
    val buttons: Buttons,
)

/** What the screen shows for each state; the words come from EM3's wording table. */
fun levelScreen(state: LevelState): LevelScreen =
    when (state) {
        LevelState.Idle -> LevelScreen(Face.WATCHING, Buttons.NONE)
        is LevelState.CheckIn -> LevelScreen(Face.CARING, Buttons.NONE)
        is LevelState.LowDayQuestion -> LevelScreen(Face.CARING, Buttons.YES_NO)
        is LevelState.PhqLeadIn -> LevelScreen(Face.CARING, Buttons.YES_NOT_NOW)
        is LevelState.PhqItem -> LevelScreen(Face.CARING, Buttons.PHQ_FOUR)
    }
