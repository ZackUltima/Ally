package ally.speech.phrases

/**
 * Turns one piece of recogniser text into lower-case word tokens: apostrophes go, any other symbol becomes a space,
 * "cannot" and "can not" become "cant", and "i am" becomes "im" (so "I'm" and "I am" match the same phrase).
 */
fun tokens(text: String): List<String> {
    val letters: String =
        text
            .lowercase()
            .replace('’', '\'')
            .replace("'", "")
            .map { if (it.isLetterOrDigit()) it else ' ' }
            .joinToString("")
    val words: List<String> = letters.split(' ').filter { it.isNotEmpty() }
    val out: MutableList<String> = mutableListOf()
    var i = 0
    while (i < words.size) {
        val word: String = words[i]
        val next: String? = words.getOrNull(i + 1)
        val canNot: Boolean = word == "can" && next == "not"
        val iAm: Boolean = word == "i" && next == "am"
        out.add(
            when {
                word == "cannot" || canNot -> "cant"
                iAm -> "im"
                else -> word
            },
        )
        i += if (canNot || iAm) 2 else 1
    }
    return out
}
