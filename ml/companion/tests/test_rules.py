from ally_companion.roomnote import RoomNote, SeenObject
from ally_companion.rules import check_reply, curator_flags, split_sentences

NOTE = RoomNote(
    time="Friday 23 October, 7:40 pm",
    people="one",
    objects=(SeenObject("cup", "table", small=False), SeenObject("glasses", "sofa", small=True)),
    talk="comforting",
    notes=("granddaughter Mei visits every Sunday after church",),
)


def test_three_sentences_pass_four_fail() -> None:
    assert check_reply("That sounds lovely. I am glad. Tell me more.", None).ok
    four = check_reply("That sounds lovely. I am glad. Really glad. Tell me more.", None)
    assert not four.ok and any("sentences" in p for p in four.problems)


def test_an_empty_reply_fails() -> None:
    assert not check_reply("   ", None).ok
    assert not check_reply("<note>likes tea</note>", None).ok


def test_two_questions_fail() -> None:
    assert not check_reply("How are you? Did you sleep?", None).ok
    assert check_reply("I am glad you slept well. How are you feeling now?", None).ok


def test_abbreviations_and_decimals_are_not_sentence_ends() -> None:  # Review Focus 2
    assert len(split_sentences("Mr. Tan came at 3.5 pm, e.g. after lunch. Nice.")) == 2
    assert split_sentences("Dr. Lee is kind! Is he?") == ["Dr. Lee is kind!", "Is he?"]


def test_a_tag_or_key_line_fails() -> None:
    for reply in ("<room> Hello there.", "objects: cup", "Hello. </room>", "talk: comforting"):
        assert not check_reply(reply, NOTE).ok, reply


def test_an_exact_note_entry_fails() -> None:
    assert not check_reply("You have cup (table) nearby.", NOTE).ok
    assert not check_reply("Your glasses (sofa, small) are there.", NOTE).ok


def test_using_the_note_in_own_words_passes() -> None:  # Review Focus 1, design choice 47
    assert check_reply("Is that your tea cup on the table?", NOTE).ok
    assert check_reply("I can see a cup on the table.", NOTE).ok


def test_five_words_copied_from_a_memory_note_fail() -> None:
    copied = check_reply("I know Mei visits every Sunday after church.", NOTE)
    assert not copied.ok and any("copied" in p for p in copied.problems)
    # no five words in a row from the memory note
    assert check_reply("Is Mei coming to see you on Sunday?", NOTE).ok


def test_a_memory_note_only_at_the_end() -> None:
    assert check_reply("That is nice. <note>likes orchids</note>", None).ok
    mid = check_reply("That is nice. <note>likes orchids</note> Tell me more.", None)
    assert not mid.ok and any("note" in p for p in mid.problems)
    one_more = check_reply("A. B. C. <note>likes orchids</note>", None)
    assert one_more.ok  # the note is not counted as a sentence


def test_flags_never_fail_a_reply() -> None:  # D-66 decision 4
    r = check_reply("Take 5 mg of it.", None)
    assert r.ok and r.flags


def test_curator_flags_name_their_group() -> None:
    assert curator_flags("You should take two tablets.") == ("medical advice",)
    assert curator_flags("This is our secret.") == ("secrecy",)
    assert curator_flags("Breaking news today!") == ("invented news",)
    assert curator_flags("I’m a real person, you know.") == ("claims to be human",)
    assert curator_flags("What a lovely afternoon.") == ()
    assert curator_flags("I am a doctor? No, but your dose matters.") == ("medical advice",)
