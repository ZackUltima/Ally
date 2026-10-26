package ally.speech.phrases

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class PhraseCheckerTest {
    private val checker = PhraseChecker(listOf("Mei Ling", "Ahmad"))
    private val talking = Heard(inConversation = true, allySpeaking = false, callConnected = false)
    private val idle = Heard(inConversation = false, allySpeaking = false, callConnected = false)

    private fun level3(
        text: String,
        heard: Heard = idle,
    ): Boolean = checker.check(text, heard).level3

    private fun lowDay(
        text: String,
        heard: Heard = talking,
    ): Boolean = checker.check(text, heard).lowDay

    @Test fun `every starting help and danger phrase is caught`() {
        val phrases =
            listOf(
                "help me",
                "I need help",
                "call my son",
                "call my daughter",
                "call Mei Ling",
                "call Ahmad",
                "I can't breathe",
                "chest pain",
                "I can't get up",
                "I'm bleeding",
            )
        phrases.forEach { assertTrue("expected level 3: $it", level3(it)) }
    }

    @Test fun `a phrase with two other words still counts`() {
        assertTrue(level3("please help me now"))
        assertFalse(level3("please could you help me now"))
    }

    @Test fun `bare help counts only on its own`() {
        assertTrue(level3("help"))
        assertTrue(level3("help help"))
        assertTrue(level3("Help!"))
        assertFalse(level3("help yourself"))
    }

    @Test fun `recogniser spellings of cannot breathe are level 3`() {
        assertTrue(level3("I cannot breathe"))
        assertTrue(level3("I cant breathe"))
        assertTrue(level3("I can not breathe"))
    }

    @Test fun `can you help me find my glasses is not level 3`() {
        assertFalse(level3("can you help me find my glasses"))
    }

    @Test fun `I'll call my daughter about dinner later is not level 3`() {
        assertFalse(level3("I'll call my daughter about dinner later"))
    }

    @Test fun `a guardian is called by name`() {
        assertTrue(level3("call Mei Ling"))
        assertTrue(level3("call Ahmad please"))
        assertFalse(level3("call Wei"))
    }

    @Test fun `Ally's own check line never triggers`() {
        assertFalse(level3("Are you okay? Say 'I'm okay', or say 'help'."))
        assertFalse(level3("Are you okay? Say 'I'm okay', or say"))
        assertFalse(level3("or say 'help'."))
    }

    @Test fun `nothing is matched while Ally speaks or a call is connected`() {
        val speaking = talking.copy(allySpeaking = true)
        val onCall = talking.copy(callConnected = true)
        assertEquals(PhraseResult(level3 = false, lowDay = false), checker.check("help me, I feel sad", speaking))
        assertEquals(PhraseResult(level3 = false, lowDay = false), checker.check("help me, I feel sad", onCall))
    }

    @Test fun `the low-day words are found in a conversation`() {
        assertTrue(lowDay("I've been feeling down all week"))
        listOf(
            "I am sad",
            "so lonely",
            "feel low",
            "depressed",
            "miserable",
            "nobody cares",
            "fed up",
            "I can't be bothered",
            "I don't enjoy anything",
        ).forEach { assertTrue("expected a low-day match: $it", lowDay(it)) }
    }

    @Test fun `low-day words are ignored outside a conversation`() {
        assertFalse(lowDay("I am so lonely", idle))
    }

    @Test fun `a word inside another word is not a low-day word`() {
        assertFalse(lowDay("the sadder story"))
    }

    @Test fun `a negated low-day word still suggests a low day`() {
        assertTrue(lowDay("I'm not sad"))
    }

    @Test fun `ordinary talk gives neither`() {
        assertEquals(PhraseResult(level3 = false, lowDay = false), checker.check("what is for lunch today", talking))
    }

    @Test fun `the result keeps no text`() {
        val fields = PhraseResult::class.java.declaredFields.filter { !it.isSynthetic }
        assertEquals(2, fields.size)
        assertTrue(fields.all { it.type == Boolean::class.javaPrimitiveType })
    }
}
