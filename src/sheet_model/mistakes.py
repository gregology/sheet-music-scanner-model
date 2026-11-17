"""Which notes a reading gets wrong, and what was drawn at each: the counts behind README.md, "Three measures".

A reading and its truth are lists of events, as `tokens.events` gives them: (pitch or "rest", length). `mistakes`
lines the two up and names each difference. `tags_of_example` and `tags_of_musicxml` say, for every event of a truth,
what the page shows there (a triplet with no mark, a note four ledger lines up, a bowed sheet), so that `rank` can
say which of those the reader gets wrong more often than the rest.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from fractions import Fraction
from pathlib import Path

from rapidfuzz.distance import Levenshtein

from .tokens import STEPS, bottom_line, events, to_pitches

# The places on the staff the generator draws, in steps from the bottom line: three ledger lines and the space
# beyond them, either way (`compose.py`'s CLEFS and `set_range` in the training data's repository, 2026-11-15).
LOWEST, HIGHEST = -7, 15
# A staff this many pixels tall in a photo is drawn smaller than the line reader's picture of it (`lines.py`: 4 of
# 14 spaces of 160 px, 46 px), so the photo holds less than the reader is given.
STAFF_SMALL = 46


def triplet(length) -> bool:
    return isinstance(length, Fraction) and length.denominator % 3 == 0


def kind(true: tuple, read: tuple) -> str:
    """What is wrong with a note read in place of another."""
    if true[0] == read[0]:
        if triplet(true[1]) and not triplet(read[1]):
            return "triplet read plain"
        if triplet(read[1]) and not triplet(true[1]):
            return "plain read triplet"
        return "length"
    return "pitch" if true[1] == read[1] else "pitch and length"


def mistakes(truth: list[tuple], found: list[tuple]) -> list[tuple[int, str]]:
    """Every difference between a reading and its truth, as (place in the truth, kind). An extra note is counted at
    the true note it was read before, or at the last one. The number of differences is the experiment's count of
    notes wrong."""
    out = []
    for step, at, other in Levenshtein.editops(truth, found):
        if step == "delete":
            out.append((at, "missing"))
        elif step == "insert":
            out.append((min(at, len(truth) - 1), "extra"))
        else:
            out.append((at, kind(truth[at], found[other])))
    return out


def read_events(lines: list[list[str]]) -> list[tuple]:
    """A page's events as read. A position reader's lines go back to pitches one by one: each has its own clef."""
    tokens = []
    for line in lines:
        positions = any(token.startswith(("note-L", "note-S", "grace-L", "grace-S")) for token in line)
        tokens += to_pitches(line) if positions else line
    return events(tokens)


def opening(tokens: list[str]) -> tuple[str | None, str]:
    """A line's clef and key signature, as its first tokens give them. A line with no key signature is in the key
    with none."""
    clef = next((token for token in tokens[:2] if token.startswith("clef-")), None)
    key = next((token for token in tokens[:3] if token.startswith("key-")), "key-0")
    return clef, key


def openings(systems: list[list[str]], lines: list[list[str]]) -> list[tuple[str, int, int, int]] | None:
    """For each line of a page: how its opening was read ("clef wrong", "key wrong" or "clef and key right"), its
    notes, the notes wrong when the line is scored on its own, and the notes wrong had its clef and key been read
    right and the rest as it was. A wrong clef or key moves every note that follows, so these lines show how much of
    a page's count one sign carries. None when the reading has another number of lines than the page: its lines
    cannot be told apart."""
    if len(systems) != len(lines):
        return None
    out = []
    for true, read in zip(systems, lines):
        (clef, key), (read_clef, read_key) = opening(true), opening(read)
        truth = events(true)
        rest = list(read)
        for start in ("clef-", "key-"):
            if rest and rest[0].startswith(start):
                rest.pop(0)
        mended = ([clef] if clef else []) + ([key] if key != "key-0" else []) + rest
        how = "clef wrong" if clef != read_clef else "key wrong" if key != read_key else "clef and key right"
        out.append((how, len(truth), Levenshtein.distance(truth, read_events([read])), Levenshtein.distance(truth, read_events([mended]))))
    return out


def place_tags(place: int) -> list[str]:
    """What a note's height adds to its tags. `place` counts steps up from the staff's bottom line: 0 to 8 are on
    the staff, 10 is the first ledger line above, -2 the first below."""
    tags = []
    if place >= 10:
        tags.append(f"ledger lines above: {(place - 8) // 2}")
    elif place <= -2:
        tags.append(f"ledger lines below: {-place // 2}")
    if place > HIGHEST:
        tags.append("higher than the generator draws")
    elif place < LOWEST:
        tags.append("lower than the generator draws")
    return tags


def tuplet_tag(actual: int, normal: int, shown: str | None) -> str:
    name = {(3, 2): "triplet", (6, 4): "sextuplet"}.get((actual, normal), f"tuplet {actual} in {normal}")
    return f"{name}, {({'bracket': 'bracket and number', 'number': 'number only'}).get(shown, 'no mark')}"


def tags_of_example(truth: dict) -> tuple[list[tuple], list[list[str]], list[str]]:
    """A generated example's events, what is drawn at each, and what holds for the whole photo. A page cut from a
    real score has no composer's level (the truth's format 3): it is tagged as a real score."""
    page = [f"sheet: {truth['sheet']['shape']}", f"light: {truth['camera']['light']}",
            "music: a real score" if truth.get("level") is None else f"level {truth['level']}"]
    if truth["shot"]["tilt_deg"] > 30:
        page.append("camera tilted over 30 degrees")
    found, tags = [], []
    bottom = bottom_line("clef-G2")
    for number, system in enumerate(truth["systems"]):
        chord, before, rocked, begins = [], [], False, len(found)   # the tags of the chord being read, and of the one before it

        def close() -> None:
            if len(chord) > 1:
                for mine in chord:
                    mine.append(f"chord of {len(chord)}")
                    if any("tied" in other for other in chord):
                        mine.append("tie on a chord")

        line = ["first line of the page"] if number == 0 else []
        start = [symbol for symbol in system["symbols"][:3] if symbol["type"] in ("clef", "key")]
        line += [f"clef: {symbol['token'][5:]}" for symbol in start if symbol["type"] == "clef"]
        if any(symbol.get("cut_off") for symbol in start):
            line.append("clef or key cut off by the photo's edge")
        (left, top), (_, low) = system["staff"]["lines_px"][0][0], system["staff"]["lines_px"][-1][0]
        if abs(low - top) < STAFF_SMALL:
            line.append(f"staff under {STAFF_SMALL} px tall in the photo")
        for symbol in system["symbols"]:
            if symbol["type"] == "clef":
                bottom = bottom_line(symbol["token"])
            elif symbol["type"] == "key":
                line = [tag for tag in line if not tag.startswith("key")] + ([f"key of {abs(symbol['sharps'])} sharps or flats"] if abs(symbol["sharps"]) >= 4 else [])
            if symbol["type"] not in ("note", "rest", "grace", "measure-rest", "multirest"):
                continue
            mine = list(line)
            if symbol["type"] in ("note", "grace"):
                mine += place_tags(STEPS.index(symbol["step"].lower()) + 7 * symbol["octave"] - bottom)
                if symbol.get("accidental_shown"):
                    mine.append("accidental drawn")
                if symbol.get("tie"):
                    mine.append("tied")
                mine += [f"mark: {name}" for name in symbol.get("articulations") or []]
                if symbol.get("tremolo"):
                    mine.append("tremolo strokes")
                if abs(symbol.get("alter", 0)) == 2:
                    mine.append("double sharp or flat")
            if symbol["type"] == "note":
                if symbol.get("under") and chord:
                    chord.append(mine)
                else:
                    close()
                    before, chord, rocked = chord, [mine], bool(symbol.get("rocked"))
                    for other in before if rocked else []:   # a tremolo joins this chord to the one before
                        other.append("tremolo between two notes")
                if rocked:
                    mine.append("tremolo between two notes")
            else:
                close()
                before, chord, rocked = [], [], False
            if symbol["type"] == "grace":
                mine.append("grace note")
            if symbol["type"] in ("measure-rest", "multirest"):
                mine.append("rest of whole measures")
            if symbol.get("tuplet"):
                mine.append(tuplet_tag(*symbol["tuplet"], symbol.get("tuplet_shown")))
            if symbol.get("duration") in (32, 64):
                mine.append("32nd note" if symbol["duration"] == 32 else "64th note")
            if symbol.get("dots", 0) >= 2:
                mine.append("double dot")
            for event in events([symbol["token"]]):
                found.append(event)
                tags.append(mine)
        close()
        # A tremolo's two notes are each written at the length of both and sound for half of it, which no symbol's
        # own token says: the line is spelled as the generator spells it, and its events are taken from that.
        spelled = [token for symbol in system["symbols"]
                   for token in (["tremolo"] * bool(symbol.get("rocked")) + ["chord"] * bool(symbol.get("under")) + [symbol.get("token")]) if token]
        whole = events(spelled)
        if len(whole) == len(found) - begins:
            found[begins:] = whole
    return found, tags, page


def tags_of_musicxml(path: Path, lengths: set) -> tuple[list[tuple], list[list[str]]]:
    """A MusicXML page's events, in the experiment's order (`score.Reading`), and what is drawn at each. `lengths` are
    the note lengths the reader has tokens for: a note of another length is tagged as one it cannot write."""
    from music21 import chord, clef, converter, expressions, harmony, key, layout, meter, note, spanner, stream

    score = converter.parse(path)
    found, tags = [], []
    for part in score.parts:
        bottom, time, sharps, first, shown = 31, None, 0, True, None   # the treble clef's bottom line, E4
        for count, measure in enumerate(part.getElementsByClass(stream.Measure)):
            sounding = [e for e in measure.recurse() if isinstance(e, note.GeneralNote) and not isinstance(e, harmony.ChordSymbol)]
            if count and any(isinstance(e, layout.SystemLayout) and e.isNew for e in measure.recurse()):
                first = False
            # A measure's notes are read from left to right, whichever voice the file keeps them in: by when each
            # starts, and the first voice's first among those that start together. OpenScore's files put some notes of
            # a single line into a second voice; read voice by voice, they would come after notes printed to their
            # right.
            spans = [[(float(e.getOffsetInHierarchy(measure)), float(e.getOffsetInHierarchy(measure) + e.duration.quarterLength)) for e in voice.notes] for voice in measure.voices]
            voices = 1 + sum(1 for at, mine in enumerate(spans) if any(a < d - 1e-6 and c < b - 1e-6 for other in spans[:at] for a, b in mine for c, d in other))
            signs = (clef.Clef, key.KeySignature, meter.TimeSignature)   # a sign comes before the note it stands before
            for element in sorted(measure.recurse(), key=lambda e: (e.getOffsetInHierarchy(measure), not isinstance(e, signs))):
                if isinstance(element, clef.Clef) and getattr(element, "lowestLine", None) is not None:
                    bottom = element.lowestLine
                elif isinstance(element, key.KeySignature):
                    sharps = element.sharps
                elif isinstance(element, meter.TimeSignature):
                    time = element.ratioString
                if not isinstance(element, note.GeneralNote) or isinstance(element, harmony.ChordSymbol):
                    continue
                mine = ["first line of the page"] if first else []
                if abs(sharps) >= 4:
                    mine.append(f"key of {abs(sharps)} sharps or flats")
                if voices > 1:
                    mine.append("two voices on the staff")
                if element.duration.isGrace:
                    length = "grace"
                    mine.append("grace note")
                elif element.isRest and len(sounding) == 1:
                    length = "measure"
                    mine.append("rest of whole measures")
                else:
                    length = Fraction(element.duration.quarterLength).limit_denominator(96)
                    if length not in lengths:
                        mine.append("a length the reader has no token for")
                for tuplet in element.duration.tuplets:
                    # MusicXML marks a tuplet on its first and last notes only, and an unmarked one on none: a note
                    # in a tuplet with no group open is in one the page does not mark.
                    if tuplet.type in ("start", "startStop"):
                        shown = "none" if tuplet.tupletActualShow in (None, "none") and not tuplet.bracket else "bracket" if tuplet.bracket else "number"
                    mine.append(tuplet_tag(tuplet.numberNotesActual, tuplet.numberNotesNormal, shown or "none"))
                    if tuplet.type in ("stop", "startStop"):
                        shown = None
                if element.duration.type in ("32nd", "64th", "128th", "breve"):
                    mine.append(f"{element.duration.type} note")
                if element.duration.dots >= 2:
                    mine.append("two dots")
                if element.tie:
                    mine.append("tied")
                for mark in element.expressions:
                    if isinstance(mark, expressions.Tremolo):
                        mine.append("tremolo strokes")
                    else:
                        mine.append(f"mark: {type(mark).__name__.lower()}")
                mine += [f"mark: {type(mark).__name__.lower()}" for mark in element.articulations]
                for site in element.getSpannerSites():
                    if isinstance(site, spanner.Ottava):
                        mine.append("under an octave line")
                    elif isinstance(site, expressions.TremoloSpanner):
                        mine.append("tremolo between two notes")
                if getattr(element, "notehead", "normal") not in ("normal", None):
                    mine.append(f"notehead: {element.notehead}")
                if element.style.noteSize == "cue":
                    mine.append("cue note")
                if element.isRest:
                    pitches = [("rest", [])]
                else:
                    sounds = sorted(element.pitches, reverse=True) if isinstance(element, chord.Chord) else [element.pitch]
                    stops = [f"chord of {len(sounds)}"] if len(sounds) > 1 else []
                    pitches = []
                    for pitch in sounds:
                        extra = stops + place_tags(pitch.diatonicNoteNum - bottom)
                        if pitch.accidental is not None and pitch.accidental.name in ("double-sharp", "double-flat"):
                            extra.append("double sharp or flat")
                        if pitch.accidental is not None and pitch.accidental.displayStatus:
                            extra.append("accidental drawn")
                        pitches.append((pitch.nameWithOctave, extra))
                for name, extra in pitches:
                    found.append((name, length))
                    tags.append(mine + extra + ([f"time {time}"] if time else []))
    return found, tags


def rank(pages: list[tuple[list[list[str]], list[tuple[int, str]]]]) -> list[dict]:
    """For every tag: how many notes carry it, how many mistakes fall on them, and how many more that is than the
    notes without the tag would make at their own rate. `pages` holds, for each page, its events' tags and its
    mistakes. Most excess first: the top of the list is what the reader most needs to be shown."""
    notes, wrong, sheets = Counter(), Counter(), Counter()
    total = errors = 0
    for tags, found in pages:
        total += len(tags)
        errors += len(found)
        for mine in tags:
            notes.update(set(mine))
        for at, _ in found:
            wrong.update(set(tags[at]))
        sheets.update({tag for mine in tags for tag in mine})
    rows = []
    for tag, count in notes.items():
        rest = (errors - wrong[tag]) / (total - count) if total > count else 0.0
        rows.append({"tag": tag, "pages": sheets[tag], "notes": count, "wrong": wrong[tag], "share": wrong[tag] / count, "excess": wrong[tag] - count * rest})
    return sorted(rows, key=lambda row: -row["excess"])


def told_apart(notes: int, wrong: int, total: int, errors: int) -> float:
    """How well a tag splits the notes into those read wrong and those read right: the likelihood-ratio statistic
    of the two-by-two table. Of two tags with the same mistakes on them, the one on fewer notes scores higher."""
    wrong, errors = min(wrong, notes), min(errors, total)
    score = 0.0
    for mine, group in ((wrong, notes), (errors - wrong, total - notes)):
        for seen, expected in ((mine, group * errors / total), (group - mine, group * (total - errors) / total)):
            if seen > 0 and expected > 0:
                score += 2 * seen * math.log(seen / expected)
    return score


def peeled(pages: list[tuple[list[list[str]], list[tuple[int, str]]]], count: int = 25) -> list[dict]:
    """The gap list: the tag that best tells the notes read wrong from the rest, then the best among the notes that
    do not carry it, and so on. Tags come together on a page (a part full of two-note tremolos is also in 2/4 and low
    on its staff), and `rank` blames each of them for the same mistakes. Here a mistake is counted for one tag alone.
    A row's `left` is the notes and mistakes that remain once its notes are taken away."""
    rows = []
    for _ in range(count):
        total, errors = sum(len(tags) for tags, _ in pages), sum(len(found) for _, found in pages)
        ranked = [row for row in rank(pages) if row["excess"] >= 1]
        if not ranked:
            break
        top = max(ranked, key=lambda row: told_apart(row["notes"], row["wrong"], total, errors))
        kept = []
        for tags, found in pages:
            keep = [number for number, mine in enumerate(tags) if top["tag"] not in mine]
            place = {number: at for at, number in enumerate(keep)}
            kept.append(([tags[number] for number in keep], [(place[at], what) for at, what in found if at in place]))
        pages = kept
        rows.append({**top, "left": {"notes": sum(len(tags) for tags, _ in pages), "wrong": sum(len(found) for _, found in pages)}})
    return rows


def join_pages(reads: dict[str, list]) -> dict[str, list]:
    """A reading with the pictures of one score joined: `<name>.p1`, `<name>.p2` and on are the pages one score was
    drawn on, and their lines follow each other in page order under `<name>`. Any other name is kept as it is."""
    paged = {}
    for name in reads:
        found = re.fullmatch(r"(.+)\.p(\d+)", name)
        paged.setdefault(found[1] if found else name, []).append((int(found[2]) if found else 0, name))
    return {name: [line for _, picture in sorted(pages) for line in reads[picture]] for name, pages in paged.items()}
