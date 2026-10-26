package ally.speech.phrases

/** What the phone is doing while the text was heard. */
data class Heard(
    val inConversation: Boolean,
    val allySpeaking: Boolean,
    val callConnected: Boolean,
)

/** Booleans only: the checker keeps no text (T21). */
data class PhraseResult(
    val level3: Boolean,
    val lowDay: Boolean,
)

/**
 * Checks one recogniser text for the level 3 help and danger phrases and for the low-day words (T20).
 * Phrases are tokenised once here; a guardian is called as "call <name>".
 */
class PhraseChecker(
    guardianNames: List<String>,
) {
    private val level3: List<List<String>> =
        (LEVEL3_PHRASES + guardianNames.map { "call $it" }).map { tokens(it) }
    private val lowDay: List<List<String>> = LOW_DAY_PHRASES.map { tokens(it) }

    fun check(
        text: String,
        heard: Heard,
    ): PhraseResult {
        if (heard.allySpeaking || heard.callConnected) return PhraseResult(level3 = false, lowDay = false)
        val words: List<String> = tokens(text)
        return PhraseResult(
            level3 = isLevel3(words),
            lowDay = heard.inConversation && lowDay.any { containsRun(words, it) },
        )
    }

    /** A phrase in a row with at most two other words, or "help" and nothing else. */
    private fun isLevel3(words: List<String>): Boolean {
        if (words.isNotEmpty() && words.all { it == "help" }) return true
        return level3.any { words.size - it.size <= MAX_OTHER_WORDS && containsRun(words, it) }
    }

    private fun containsRun(
        words: List<String>,
        run: List<String>,
    ): Boolean = run.isNotEmpty() && words.windowed(run.size).any { it == run }

    private companion object {
        const val MAX_OTHER_WORDS: Int = 2
    }
}
