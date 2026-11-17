from fractions import Fraction

from sheet_model.mistakes import join_pages, kind, mistakes, openings, peeled, place_tags, rank, tags_of_example, tags_of_musicxml

Q, E, T = Fraction(1), Fraction(1, 2), Fraction(1, 3)


# The survey's first counts put a triplet read as plain eighths under "length", with every other wrong length, and
# the largest single cause of mistakes on real music was invisible in the table.
def test_a_triplet_read_as_plain_notes_is_named_for_what_it_is():
    assert kind(("C4", T), ("C4", E)) == "triplet read plain"
    assert kind(("C4", E), ("C4", T)) == "plain read triplet"
    assert kind(("C4", Q), ("C4", E)) == "length"
    assert kind(("C4", Q), ("D4", Q)) == "pitch"
    assert kind(("C4", Q), ("D4", E)) == "pitch and length"


# An extra note read after the page's last true note pointed one past the end of the truth, and looking up what was
# drawn there stopped the whole count.
def test_an_extra_note_at_the_end_is_counted_at_the_last_true_note():
    truth = [("C4", Q), ("D4", Q)]
    assert mistakes(truth, [("C4", Q), ("D4", Q), ("E4", Q)]) == [(1, "extra")]
    assert mistakes(truth, [("D4", Q)]) == [(0, "missing")]


# The generator's highest note sits in the space above the third ledger line. A count that called that note "higher
# than the generator draws" blamed the generator for notes it does draw, and one that missed the line above it hid
# the gap that made 33 of the reader's mistakes on real music.
def test_a_note_is_beyond_the_generator_only_past_its_last_place():
    assert place_tags(8) == []                                    # the top line
    assert place_tags(15) == ["ledger lines above: 3"]            # F6 in the treble clef: drawn
    assert place_tags(16) == ["ledger lines above: 4", "higher than the generator draws"]
    assert place_tags(-7) == ["ledger lines below: 3"]
    assert place_tags(-8) == ["ledger lines below: 4", "lower than the generator draws"]


# Ranked by share wrong alone, a tag on one note that happened to be misread came above a tag on a hundred notes with
# forty misread. The rank is by how many mistakes a tag adds beyond what its notes would make anyway.
def test_tags_are_ranked_by_the_mistakes_they_add():
    tags = [["rare"]] + [["common"]] * 100 + [[]] * 100
    wrong = [(0, "pitch")] + [(1 + n, "pitch") for n in range(40)] + [(150, "pitch")]
    rows = rank([(tags, wrong)])
    assert [row["tag"] for row in rows] == ["common", "rare"]
    assert rows[0]["notes"] == 100 and rows[0]["wrong"] == 40
    assert round(rows[0]["excess"]) == 38                         # the other 101 notes make 2 mistakes: 2 in 100 expected


# On the survey a part full of two-note tremolos was in 2/4, and "time 2/4" came fourth in the ranking for mistakes
# that were all the tremolos'. A list to build from must count each mistake once, for the tag that explains most.
def test_the_gap_list_counts_a_mistake_for_one_tag_alone():
    tags = [["tremolo", "time 2/4"]] * 50 + [["time 2/4"]] * 50 + [[]] * 100
    wrong = [(n, "length") for n in range(50)] + [(60, "pitch"), (150, "pitch")]
    rows = peeled([(tags, wrong)])
    assert [row["tag"] for row in rows] == ["tremolo"]            # without the tremolos, 2/4 reads as the rest does
    assert rows[0]["left"] == {"notes": 150, "wrong": 2}


def example() -> dict:
    def note(pitch, token, **more):
        return {"type": "note", "step": pitch[0], "octave": int(pitch[-1]), "token": token, **more}
    first = [{"type": "clef", "token": "clef-G2"}, {"type": "key", "sharps": 4, "token": "key-4#"},
             note("G6", "note-G#6_8:3", tuplet=[3, 2], tuplet_shown="none"), {"type": "multirest", "token": "multirest-3"}]
    second = [{"type": "clef", "token": "clef-F4", "cut_off": True}, note("C4", "note-C4_4", tremolo=2)]

    def staff(top, low):   # the staff's top and bottom lines in the photo, each as its points from left to right
        return {"lines_px": [[[100.0, top], [900.0, top]], [[100.0, low], [900.0, low]]]}
    return {"sheet": {"shape": "bow"}, "camera": {"light": "dim"}, "shot": {"tilt_deg": 40.0}, "level": 2,
            "systems": [{"symbols": first, "staff": staff(200.0, 260.0)}, {"symbols": second, "staff": staff(400.0, 430.0)}]}


# A rest of three measures is three events in the score. Tagged as one, every tag after it sat on the wrong note.
def test_an_example_gives_one_list_of_tags_for_each_event():
    events, tags, page = tags_of_example(example())
    assert events == [("G#6", T), ("rest", "measure"), ("rest", "measure"), ("rest", "measure"), ("C4", Q)]
    assert len(tags) == len(events)
    assert tags[0] == ["first line of the page", "clef: G2", "key of 4 sharps or flats", "ledger lines above: 4", "higher than the generator draws", "triplet, no mark"]
    assert tags[3] == ["first line of the page", "clef: G2", "key of 4 sharps or flats", "rest of whole measures"]
    # Middle C under the bass clef, on the second line: its clef is half outside the photo and its staff 30 px tall.
    assert tags[4] == ["clef: F4", "clef or key cut off by the photo's edge", "staff under 46 px tall in the photo", "ledger lines above: 1", "tremolo strokes"]
    assert page == ["sheet: bow", "light: dim", "level 2", "camera tilted over 30 degrees"]


# One alto clef read as a treble clef moves every note on its line, and a page of four lines read 91% wrong for it.
# Counted with every other mistake, that looked like a reader that could not read the notes; counted by line, it is
# one sign.
def test_a_line_with_its_clef_read_wrong_is_counted_apart():
    true = [["clef-C3", "key-1b", "note-C4_4", "note-D4_4", "barline"], ["clef-C3", "key-1b", "note-C4_4", "barline"], ["clef-C3", "note-C4_4"]]
    read = [["clef-G2", "key-1b", "note-L3_4", "note-S3_4", "barline"], ["clef-C3", "note-L3_4", "barline"], ["clef-C3", "note-L3_4"]]
    # Each line: how its start was read, its notes, the notes wrong, and the notes wrong with the start put right.
    assert openings(true, read) == [("clef wrong", 2, 2, 0), ("key wrong", 1, 0, 0), ("clef and key right", 1, 0, 0)]
    assert openings(true, read[:2]) is None   # a line was missed: which one is not known


MUSICXML = """<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="4.0"><part-list><score-part id="P1"><part-name>Violin</part-name></score-part></part-list>
<part id="P1"><measure number="1">
<attributes><divisions>6</divisions><key><fifths>0</fifths></key><time><beats>2</beats><beat-type>4</beat-type></time>
<clef><sign>G</sign><line>2</line></clef></attributes>
{first}{second}
</measure></part></score-partwise>"""


def triplet(marked: bool) -> str:
    notes = ""
    for place in range(3):
        mark = ""
        if marked and place == 0:
            mark = '<notations><tuplet type="start" bracket="yes" show-number="actual"/></notations>'
        if marked and place == 2:
            mark = '<notations><tuplet type="stop"/></notations>'
        notes += ("<note><pitch><step>G</step><octave>6</octave></pitch><duration>2</duration><type>eighth</type>"
                  f"<time-modification><actual-notes>3</actual-notes><normal-notes>2</normal-notes></time-modification>{mark}</note>")
    return notes


# MusicXML marks a triplet on its first and last notes, and an unmarked triplet on none. The first tagging carried the
# last mark seen onto every triplet after it, so the unmarked triplets on the page's last line counted as bracketed,
# and the kind of triplet the reader misreads most looked like one it reads well.
def test_a_triplet_with_no_mark_after_a_marked_one_is_tagged_unmarked(tmp_path):
    path = tmp_path / "page.musicxml"
    path.write_text(MUSICXML.format(first=triplet(True), second=triplet(False)))
    events, tags = tags_of_musicxml(path, {Fraction(1, 3)})
    assert events == [("G6", T)] * 6
    assert ["triplet, bracket and number" in mine for mine in tags] == [True] * 3 + [False] * 3
    assert ["triplet, no mark" in mine for mine in tags] == [False] * 3 + [True] * 3
    assert all("higher than the generator draws" in mine for mine in tags)   # G6 is on the fourth ledger line


def voiced(step: str, voice: int, duration: int = 6, kind: str = "quarter") -> str:
    return f"<note><pitch><step>{step}</step><octave>4</octave></pitch><duration>{duration}</duration><voice>{voice}</voice><type>{kind}</type></note>"


# OpenScore's files keep some notes of a single line in a second voice. Read voice by voice, a measure's last note came
# before its first, and a page the reader read right counted 30 notes wrong, all blamed on "two voices on the staff".
def test_notes_kept_in_a_second_voice_are_read_from_left_to_right(tmp_path):
    path = tmp_path / "page.musicxml"
    # C in voice 1, then D in voice 2 after a rest hidden in it: one line, C then D.
    line = voiced("C", 1) + "<backup><duration>6</duration></backup><forward><duration>6</duration><voice>2</voice></forward>" + voiced("D", 2)
    # Whereas E and G start together and sound together: two voices.
    both = voiced("E", 1) + voiced("F", 1) + "<backup><duration>12</duration></backup>" + voiced("G", 2, 12, "half")
    path.write_text(MUSICXML.format(first=line, second=""))
    events, tags = tags_of_musicxml(path, {Q})
    assert [name for name, _ in events] == ["C4", "D4"]
    assert not any("two voices on the staff" in mine for mine in tags)
    path.write_text(MUSICXML.format(first=both, second=""))
    events, tags = tags_of_musicxml(path, {Q})
    assert [name for name, _ in events] == ["E4", "G4", "F4"]
    assert all("two voices on the staff" in mine for mine in tags)


# Pages joined in the order of their names put page 10 before page 2: the music is then scored out of order, and
# every note of the moved pages counts as wrong.
def test_a_scores_pages_are_joined_in_page_order():
    reads = {"7-1.p10": ["last"], "7-1.p2": ["second"], "7-1.p1": ["first"], "8-0": ["alone"]}
    assert join_pages(reads) == {"7-1": ["first", "second", "last"], "8-0": ["alone"]}


# A page cut from a real score has no composer's level, and its chords and tremolos were not named: such a page was
# tagged "level None", and a reader's mistakes on a tremolo's notes or on a held note of a chord could not be told
# from the rest, so the generator could not be asked for more of what was misread.
def test_a_real_scores_page_is_tagged_for_what_format_3_adds():
    def note(pitch, token, **more):
        return {"type": "note", "step": pitch[0], "octave": int(pitch[-1]), "token": token, **more}
    symbols = [{"type": "clef", "token": "clef-G2"},
               note("E5", "note-E5_2", tie="i"), note("C5", "note-C5_2", under="n1"),
               note("G4", "note-G4_2", rocked=3), note("F4", "note-F##4_2", under="n3", alter=2),
               note("A4", "note-A4_64", duration=64), note("B4", "note-B4_4..", dots=2), {"type": "barline", "token": "barline"}]
    truth = {"sheet": {"shape": "flat"}, "camera": {"light": "bright"}, "shot": {"tilt_deg": 5.0}, "level": None, "source": {"kind": "score"},
             "systems": [{"symbols": symbols, "staff": {"lines_px": [[[100.0, 200.0]], [[100.0, 300.0]]]}}]}
    events, tags, page = tags_of_example(truth)
    # The tremolo's four notes sound for half the half note each is printed as.
    assert events == [("E5", Q), ("C5", Q), ("G4", Q), ("F##4", Q), ("A4", Fraction(1, 16)), ("B4", Fraction(7, 4))]
    assert tags[0] == ["first line of the page", "clef: G2", "tied", "chord of 2", "tie on a chord", "tremolo between two notes"]
    assert tags[1] == ["first line of the page", "clef: G2", "chord of 2", "tie on a chord", "tremolo between two notes"]
    assert tags[2] == ["first line of the page", "clef: G2", "tremolo between two notes", "chord of 2"]
    assert tags[3] == ["first line of the page", "clef: G2", "double sharp or flat", "tremolo between two notes", "chord of 2"]
    assert tags[4] == ["first line of the page", "clef: G2", "64th note"]
    assert tags[5] == ["first line of the page", "clef: G2", "double dot"]
    assert page == ["sheet: flat", "light: bright", "music: a real score"]
