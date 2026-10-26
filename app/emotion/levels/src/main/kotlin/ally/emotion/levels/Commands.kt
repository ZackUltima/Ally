package ally.emotion.levels

import ally.fall.flow.GuardianId
import java.time.LocalDateTime

/** Ids of design plan section E10's exact lines; the words live in EM3's wording table. */
enum class Line {
    CHECK_IN,
    CHECK_IN_FINE,
    CHECK_IN_NO_ANSWER,
    TALK_LATER,
    LOW_DAY_QUESTION,
    PHQ_LEAD_IN,
    PHQ_ITEM_1,
    PHQ_ITEM_2,
    PHQ_OPTIONS_AGAIN,
    PHQ_CLOSING,
    NOTE_BEING_SENT,
}

enum class NoteKind { LOW_MOOD, LOW_MOOD_NO_ANSWER }

/** What the rules ask the adaptor to do; they never talk to the screen, the voice or the backend (T8). */
sealed interface LevelCommand

/** Say this line; [guardian] names the guardian a line is about. */
data class Say(
    val line: Line,
    val guardian: GuardianId? = null,
) : LevelCommand

/** Send the level 2 note to [to]. */
data class SendNote(
    val kind: NoteKind,
    val to: GuardianId,
) : LevelCommand

/** An event came with a time earlier than the last one; the rules went on from the last time, and the adaptor logs it. */
data class ClockWentBack(
    val from: LocalDateTime,
    val to: LocalDateTime,
) : LevelCommand
