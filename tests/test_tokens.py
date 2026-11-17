from fractions import Fraction

from sheet_model.tokens import events, length, split_counts, tie_to_no_note, to_pitches, to_positions

# Two measures in D major, then the bass clef and the treble clef an octave down.
LINE = ["clef-G2", "key-2#", "time-4/4", "note-F#4_4", "note-C#5_4", "note-C5_4", "note-C4_4", "note-C5_4", "barline",
        "note-G#4_4", "tie", "barline", "note-G#4_4", "note-G#4_4", "note-F4_4", "barline",
        "clef-F4", "note-G2_4", "grace-D3_8", "note-C3_4", "barline", "clef-G2_8vb", "note-E3_2"]


# The first reader, writing pitches, read runs of notes an octave low or in the wrong clef: it carried the clef and key
# along the line itself. Positions name only what is drawn. Turned back into pitches they must give every note's
# pitch exactly, or a reader trained on them is scored wrong for notes it read right.
def test_positions_name_what_is_drawn_and_give_back_every_pitch():
    positions = to_positions(LINE)
    assert positions == [
        "clef-G2", "key-2#", "time-4/4",
        "note-S1_4",                    # F sharp from the key: nothing drawn
        "note-S3_4",                    # C sharp from the key
        "acc-n", "note-S3_4",           # C natural, drawn
        "acc-n", "note-L0_4",           # middle C, an octave down: the natural above does not reach it
        "note-S3_4",                    # C natural again: the natural holds to the barline
        "barline",
        "acc-#", "note-L2_4", "tie", "barline",
        "note-L2_4",                    # tied over the barline: keeps its sharp, undrawn
        "note-L2_4",                    # and so does the next G in that measure, as the generator draws it
        "acc-n", "note-S1_4",
        "barline",
        "clef-F4", "note-L1_4", "grace-L3_8", "acc-n", "note-S2_4",
        "barline",
        "clef-G2_8vb", "note-L1_2",     # the treble clef's bottom line, an octave down
    ]
    assert to_pitches(positions) == LINE


# A reader can write any token of its vocabulary anywhere: an accidental with no note after it, a note before any
# clef, a place it never saw. Reading those must not fail, and a stray accidental must not change a later note.
def test_a_malformed_reading_still_turns_into_pitches():
    assert to_pitches(["note-L2_4", "acc-#", "barline", "note-L2_4", "note-Q9_4", "<unknown>"]) == [
        "note-G4_4", "barline", "note-G4_4", "note-Q9_4", "<unknown>"]


# Real parts have two notes at once and grace notes with accidentals of their own; the first readers had met neither
# and read both wrong. Their tokens must turn into positions and back without losing a note or an accidental.
def test_chords_and_grace_note_accidentals_come_back_from_positions():
    line = ["clef-G2", "key-1#", "time-3/4", "note-F#5_4", "chord", "note-A4_4", "grace-D#5_16", "grace-E5_16", "note-D#5_4", "note-F5_4", "chord", "note-D5_4", "barline"]
    positions = to_positions(line, graces=True)
    assert positions == [
        "clef-G2", "key-1#", "time-3/4",
        "note-L5_4", "chord", "note-S2_4",                 # F sharp from the key, over A
        "acc-#", "grace-L4_16", "grace-S4_16",            # the grace note's own sharp, drawn
        "note-L4_4",                                       # D sharp again: the grace note's sharp holds
        "acc-n", "note-L5_4", "chord", "acc-n", "note-L4_4",
        "barline",
    ]
    assert to_pitches(positions) == line
    assert events(line)[:3] == [("F#5", Fraction(1)), ("A4", Fraction(1)), ("D#5", "grace")]   # a chord scores as its notes, top first


# The first readers knew a rest of several measures only as one token for each count they had met, and read a
# printed 2 as 5, 8 and 12. Written digit by digit, any count can be read, and must score as that many measures.
def test_a_long_rests_count_is_read_digit_by_digit():
    assert split_counts(["clef-F4", "multirest-12", "barline", "multirest-7"]) == ["clef-F4", "multirest", "count-1", "count-2", "barline", "multirest", "count-7"]
    assert events(["multirest", "count-1", "count-2", "barline", "note-C4_4"]) == [("rest", "measure")] * 12 + [("C4", Fraction(1))]
    assert events(["multirest-3"]) == [("rest", "measure")] * 3            # the first readers' token still scores
    assert events(["multirest", "barline", "multirest", "count-9", "count-9", "count-9", "count-9"]) == [("rest", "measure")] * 2   # no count, or an absurd one: one measure


# Six sixteenths in the time of four last a quarter note. Scored as six in the time of two they would last half that,
# and every sextuplet a reader got right would count as wrong.
def test_six_in_the_time_of_four_fill_a_quarter_note():
    assert 6 * length("16:6") == 1 and 3 * length("8:3") == 1 and 3 * length("4:3") == 2


# Catches: the first dataset tied 8,167 notes to a rest, which no real page does, and a reader trained on those lines
# learns a tie where none can be; or the check also dropping a tie that carries over a barline or to the next line.
def test_a_tie_that_leads_to_a_rest_is_found():
    assert tie_to_no_note(["clef-G2", "note-D4_8", "tie", "rest-8", "note-D4_4"])
    assert tie_to_no_note(["note-D4_8", "tie", "barline", "grace-E4_16", "note-D4_4"])
    assert not tie_to_no_note(["note-D4_8", "tie", "barline", "key-2b", "note-D4_4", "rest-4"])
    assert not tie_to_no_note(["note-D4_2", "tie"])


# Real scores have double sharps and double flats (the truth's format 3), and the first position tokens had no sign
# for either: turning such a line into positions stopped with a KeyError, so a dataset of real scores could not be
# trained on at all.
def test_double_sharps_and_flats_are_drawn_and_read_back():
    line = ["clef-G2", "key-4#", "note-F##4_4", "note-F#4_4", "note-Bbb4_4", "note-B4_4", "barline", "note-F##4_2"]
    positions = to_positions(line, graces=True)
    assert positions == [
        "clef-G2", "key-4#",
        "acc-##", "note-S1_4",          # F double sharp
        "acc-#", "note-S1_4",           # back to the key's F sharp: the sharp is drawn again
        "acc-bb", "note-L3_4",
        "acc-n", "note-L3_4",
        "barline",
        "acc-##", "note-S1_2",          # a new measure: drawn again
    ]
    assert to_pitches(positions) == line


# A real chord can hold one note while the others change: each tied note has its own `tie`. The first code kept one
# tie, the last, and forgot it at the next note it met. A chord's lower note tied over a barline was then given an
# accidental the page does not draw, and read back from a page it lost its sharp.
def test_each_tied_note_of_a_chord_keeps_its_accidental_undrawn():
    line = ["clef-G2", "note-E5_2", "chord", "note-C#5_2", "tie", "chord", "note-G#4_2", "tie", "barline",
            "note-F5_2", "chord", "note-C#5_2", "chord", "note-G#4_2", "note-C#5_4", "note-G4_4", "barline"]
    positions = to_positions(line, graces=True)
    assert positions == [
        "clef-G2", "note-S4_2", "chord", "acc-#", "note-S3_2", "tie", "chord", "acc-#", "note-L2_2", "tie", "barline",
        "note-L5_2", "chord", "note-S3_2", "chord", "note-L2_2",   # both sharps tied in: neither is drawn
        "note-S3_4",                                               # C sharp holds to the barline
        "acc-n", "note-L2_4",                                      # and so does the G sharp, so its natural is drawn
        "barline",
    ]
    assert to_pitches(positions) == line
    # A tie leads to the very next chord and no further: the C after the F is not tied, and is the key's C.
    assert to_pitches(["note-S3_4", "acc-#", "tie", "barline", "note-L5_4", "note-S3_4"])[-1] == "note-C5_4"


# A grace note tied to the main note under it: the tie is the grace note's. Taken as a tie from the note before the
# grace note, it hid the natural drawn on a note at that note's place, and the line did not come back from positions.
def test_a_tie_after_a_grace_note_is_the_grace_notes():
    line = ["clef-F4", "note-D#3_2", "grace-D2_8", "tie", "note-D3_4", "chord", "note-D2_4"]
    positions = to_positions(line, graces=True)
    assert positions == ["clef-F4", "acc-#", "note-L3_2", "grace-S-1_8", "tie", "acc-n", "note-L3_4", "chord", "note-S-1_4"]
    assert to_pitches(positions) == line


# Format 3 writes any tuplet: `:5` is five in the time of four, and one that is not a count over the power of two
# below it names both numbers. `:2/3` stopped the scoring with a ValueError; a double dot and a 64th must have their
# lengths too, or every such note a reader gets right counts as wrong.
def test_any_tuplet_and_any_length_has_its_length():
    assert 2 * length("8:2/3") == Fraction(3, 2)                 # two eighths in the time of three
    assert 5 * length("16:5") == 1 and 7 * length("16:7") == 1 and 4 * length("8:4/3") == Fraction(3, 2)
    assert length("4..") == Fraction(7, 4) and 16 * length("64") == 1
    assert events(["note-C4_8:2/3", "rest-8:2/3", "note-C4_8:0", "note-C4_8:2/0", "note-C4_8:x/3"]) == [("C4", Fraction(3, 4)), ("rest", Fraction(3, 4))]


# A tremolo between two notes prints each at the length of both together, and a score's file gives each half of it.
# Scored at the printed length, a reader that read the page right had every such note's length wrong: the survey's
# second largest gap.
def test_a_tremolos_two_notes_each_sound_for_half_their_printed_length():
    assert events(["note-C4_2", "tremolo", "note-E4_2", "note-G4_2"]) == [("C4", Fraction(1)), ("E4", Fraction(1)), ("G4", Fraction(2))]
    chords = ["note-E4_4", "chord", "note-C4_4", "tremolo", "note-G4_4", "tie", "chord", "note-D4_4", "barline", "note-G4_4"]
    assert events(chords) == [("E4", Fraction(1, 2)), ("C4", Fraction(1, 2)), ("G4", Fraction(1, 2)), ("D4", Fraction(1, 2)), ("G4", Fraction(1))]
    # A reader can write the sign anywhere: with no note before it or none after, nothing is halved and nothing fails.
    assert events(["tremolo", "rest-4", "tremolo", "barline", "note-C4_4", "chord"]) == [("rest", Fraction(1)), ("C4", Fraction(1))]
    positions = to_positions(["clef-G2", "note-C5_2", "tremolo", "note-Eb5_2", "chord", "note-C5_2"], graces=True)
    # Catches a crash scoring r54k-tremolo on the survey: a tremolo read after a rest of several measures, which
    # follows a note without ending its chord, halved the rests' length "measure".
    assert events(["note-C4_4", "multirest", "count-2", "tremolo", "note-E4_4"]) == [("C4", Fraction(1, 2)), ("rest", "measure"), ("rest", "measure"), ("E4", Fraction(1, 2))]
    assert positions == ["clef-G2", "note-S3_2", "tremolo", "acc-b", "note-S4_2", "chord", "note-S3_2"]
