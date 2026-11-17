from fractions import Fraction

from sheet_model.measures import measures, meter, repair, repaired, total


def line(text: str, sure: float | list[float] = 0.99) -> dict:
    tokens = text.split()
    return {"tokens": tokens, "sure": [sure] * len(tokens) if isinstance(sure, float) else sure}


# The time signature is printed on a page's first line only. Measured line by line, every later line had no measure
# length, and the unmarked triplets on a real page's last line, the ones the rule is for, were never looked at.
def test_the_time_signature_is_carried_down_the_page():
    page = [["clef-G2", "time-C|", "note-L2_1", "barline"], ["clef-G2", "note-L2_2", "note-L2_2", "barline", "note-L1_1"]]
    assert measures(page) == [(0, [2], Fraction(4)), (1, [1, 2], Fraction(4)), (1, [4], Fraction(4))]
    assert meter("time-6/8") == 3 and meter("time-3/2") == 6 and meter("time-?") is None


# A chord's three notes were added as three notes in a row, so every measure with a chord failed the sum; and a
# whole-measure rest, which has no length of its own, made every empty measure fail too.
def test_a_chord_counts_once_and_a_whole_measure_rest_is_not_summed():
    tokens = "note-L4_4. chord note-L2_4. chord note-S0_4. rest-8 grace-L3_16 note-L1_2".split()
    assert total(tokens, list(range(len(tokens)))) == Fraction(4)
    assert total(["rest-measure"], [0]) is None
    assert total(["multirest", "count-1", "count-2"], [0, 1, 2]) is None


# Real page q2: plain eighths in common time, two of them read as triplet eighths at 0.6 sure. The measure comes
# up a third of a beat short, and taking the two marks off is the one change that makes it add up.
def test_marks_on_plain_notes_are_taken_off():
    page = [line("clef-G2 time-C note-L1_8 note-L2_8:3 note-L3_8:3 note-L1_8 note-L2_4 note-L2_4 barline", [0.99, 0.99, 0.99, 0.6, 0.6, 0.99, 0.99, 0.99, 0.99])]
    lines, failed, changed = repaired(page)
    assert (failed, changed) == (1, 1)
    assert lines[0][2:6] == ["note-L1_8", "note-L2_8", "note-L3_8", "note-L1_8"]


# Real page p2: six triplet eighths on the last line with no mark, read as plain. The measure is a beat too long, and
# of the runs of three eighths it could turn into triplets the rule takes the ones the reader was least sure of.
def test_unmarked_triplets_are_marked_where_the_reader_doubted():
    sure = [0.99, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.99, 0.99, 0.99, 0.99, 0.99]
    page = [line("time-2/4 note-L1_8 note-L2_8 note-L3_8 note-L1_8 note-L2_8 note-L3_8 barline note-L1_8 note-L1_8 note-L1_8 note-L1_8", sure)]
    lines, failed, changed = repaired(page)
    assert (failed, changed) == (1, 1)
    assert lines[0][1:7] == ["note-L1_8:3", "note-L2_8:3", "note-L3_8:3", "note-L1_8:3", "note-L2_8:3", "note-L3_8:3"]
    assert lines[0][8:] == page[0]["tokens"][8:]          # the measure that adds up is left as read


# Made-up page e1: a sixteenth read as an eighth leaves the measure a quarter of a beat long. The first rule turned
# three other sixteenths into a triplet to take the quarter back, a note the reader has no token for, and doubled
# the page's mistakes, 60 to 120. A wrong length that is not a triplet's is not this rule's to mend.
def test_sixteenths_are_never_made_into_triplets():
    page = [line("time-2/4 note-L1_8 note-L2_16 note-L3_16 note-L1_16 note-L2_16 note-L3_16 note-L1_16 note-L2_16 barline", 0.5)]
    lines, failed, changed = repaired(page)
    assert (failed, changed) == (1, 0)
    assert lines[0] == page[0]["tokens"]


# A measure of nine equally doubted eighths in 4/4 can lose its extra half beat three ways. Picking the first found
# changed right notes as often as wrong ones; with nothing to choose by, the measure is left as read.
def test_a_measure_with_two_equal_repairs_is_left_alone():
    tokens = "note-L1_8 note-L1_8 note-L1_8 note-L1_8 note-L1_8 note-L1_8 note-L1_8 note-L1_8 note-L1_8".split()
    assert repair(tokens, [0.5] * 9, list(range(9)), Fraction(4)) is None


# The notes stacked on a chord kept their plain length when the chord's first note became a triplet's, and the page
# then scored a wrong length for each of them.
def test_a_chords_stacked_notes_change_with_it():
    tokens = "note-L1_8 chord note-L3_8 note-L2_8 note-L3_8 note-L1_4".split()
    plan = repair(tokens, [0.5] * 6, list(range(6)), Fraction(2))
    assert plan == {0: "note-L1_8:3", 2: "note-L3_8:3", 3: "note-L2_8:3", 4: "note-L3_8:3"}


# The limit keeps the rule from changing what the reader was sure of: on real pages it never saw, changes to sure
# tokens broke as many measures as they mended.
def test_a_token_the_reader_was_sure_of_is_not_changed():
    page = [line("time-2/4 note-L1_8 note-L2_8 note-L3_8 note-L1_4 barline", 0.97)]
    assert repaired(page, limit=0.9)[2] == 0
    assert repaired(page, limit=1.0)[2] == 1


# A tremolo between two notes prints both at the length of the pair. Added up as two notes, every measure with one
# came out too long, was counted as failing, and the triplet rule was let loose on notes it was never measured on.
def test_a_tremolos_pair_counts_once_and_is_left_alone():
    tokens = "note-L2_2 tremolo note-L3_2 note-L4_2 chord note-L2_2 tremolo note-L3_2 chord note-L1_2".split()
    assert total(tokens, list(range(len(tokens)))) == Fraction(4)
    page = [line("clef-G2 time-2/4 note-L2_8 note-L2_8 note-L2_8 note-L2_4 tremolo note-L3_4 barline", 0.5)]
    assert total(page[0]["tokens"], [2, 3, 4, 5, 6, 7]) == Fraction(5, 2)   # half a beat too long
    assert repair(page[0]["tokens"], page[0]["sure"], [2, 3, 4, 5, 6, 7], Fraction(2)) is None
    assert repaired(page) == ([page[0]["tokens"]], 1, 0)
