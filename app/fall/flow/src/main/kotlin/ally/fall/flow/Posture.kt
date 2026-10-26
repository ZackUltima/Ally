package ally.fall.flow

/** What the flow needs to know about one person's label on one tick (technical plan section 1, T10). */
enum class Posture { FALL, FALLEN, UPRIGHT, OTHER }

/** Maps the fall model's label name, never its index, to a posture; `upright` is the groups5 label. */
fun postureOf(label: String): Posture {
    require(label.isNotBlank()) { "label must not be blank" }
    return when (label) {
        "fall" -> Posture.FALL
        "fallen" -> Posture.FALLEN
        "standing", "walk", "upright" -> Posture.UPRIGHT
        else -> Posture.OTHER
    }
}
