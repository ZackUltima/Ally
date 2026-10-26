package ally.emotion.levels

import org.junit.Assert.assertEquals
import org.junit.Test

class ExpressionTest {
    private val names =
        listOf("neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear", "contempt")

    @Test fun `the eight FER+ names map by name`() {
        assertEquals(Expression.NEUTRAL, expressionOf("neutral"))
        assertEquals(Expression.HAPPINESS, expressionOf("happiness"))
        assertEquals(Expression.SURPRISE, expressionOf("surprise"))
        assertEquals(Expression.SADNESS, expressionOf("sadness"))
        assertEquals(Expression.ANGER, expressionOf("anger"))
        assertEquals(Expression.DISGUST, expressionOf("disgust"))
        assertEquals(Expression.FEAR, expressionOf("fear"))
        assertEquals(Expression.CONTEMPT, expressionOf("contempt"))
        assertEquals(8, names.map { expressionOf(it) }.toSet().size)
    }

    @Test fun `exactly five expressions are negative`() {
        val negative = names.map { expressionOf(it) }.filter { it.isNegative }.toSet()
        assertEquals(
            setOf(Expression.SADNESS, Expression.ANGER, Expression.FEAR, Expression.DISGUST, Expression.CONTEMPT),
            negative,
        )
    }

    @Test(expected = IllegalArgumentException::class)
    fun `an unknown name is refused`() {
        expressionOf("sad")
    }
}
