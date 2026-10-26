package ally.emotion.levels

/** The eight FER+ expressions (D-14); `isNegative` is the five that count towards a low mood (D-23 rule 3). */
enum class Expression(
    val isNegative: Boolean,
) {
    NEUTRAL(false),
    HAPPINESS(false),
    SURPRISE(false),
    SADNESS(true),
    ANGER(true),
    DISGUST(true),
    FEAR(true),
    CONTEMPT(true),
}

/** Maps a FER+ class name to its expression by name, never by index; an unknown name is a bug, not a guess. */
fun expressionOf(name: String): Expression =
    when (name) {
        "neutral" -> Expression.NEUTRAL
        "happiness" -> Expression.HAPPINESS
        "surprise" -> Expression.SURPRISE
        "sadness" -> Expression.SADNESS
        "anger" -> Expression.ANGER
        "disgust" -> Expression.DISGUST
        "fear" -> Expression.FEAR
        "contempt" -> Expression.CONTEMPT
        else -> throw IllegalArgumentException("unknown FER+ expression name: $name")
    }
