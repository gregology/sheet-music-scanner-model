"""The reader's vocabulary, and what a line's tokens mean as notes.

The dataset's tokens name pitches (the generator's README, "Tokens"): `clef-G2`, `key-2b`,
`time-3/4`, `note-Bb4_8.`, `rest-4`, `barline`, `tie`, `chord` between two notes played at once, and `tremolo`
between two notes or chords played in turn. A reader can instead be trained on positions, which name what is drawn
where: `note-S3_8.` is an eighth note in the staff's third space, and `acc-b` a flat drawn before the next note. `to_positions` and `to_pitches` turn one into the other by the
rules of notation. A rest of several measures can be written with its count digit by digit (`split_counts`), so a
reader reads the number and need not have met it. README.md, "Tokens". Number 0 is kept for CTC's blank.
"""

from __future__ import annotations

import json
import re
from fractions import Fraction
from pathlib import Path

UNKNOWN = "<unknown>"
STEPS = "cdefgab"
SHARP_ORDER = "fcgdaeb"
CLEF_NOTE = {"G": 32, "F": 24, "C": 28}   # the note each clef's shape names, in steps above C0: G4, F3 and C4
ACCIDENTALS = {2: "acc-##", 1: "acc-#", 0: "acc-n", -1: "acc-b", -2: "acc-bb"}
DRAWN = {token[4:]: alter for alter, token in ACCIDENTALS.items()}   # what each drawn accidental does to its note
BARLINES = {"barline", "barline-double", "barline-end", "repeat-end", "repeat-start"}
PITCH = re.compile(r"([A-G])(#*|b*)(\d+)")


class Vocabulary:
    def __init__(self, tokens: list[str]):
        self.tokens = ["<blank>", UNKNOWN] + sorted(set(tokens) - {"<blank>", UNKNOWN})
        self.number = {token: number for number, token in enumerate(self.tokens)}

    def __len__(self) -> int:
        return len(self.tokens)

    def encode(self, tokens: list[str]) -> list[int]:
        return [self.number.get(token, 1) for token in tokens]

    def decode(self, numbers) -> list[str]:
        return [self.tokens[number] for number in numbers]

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.tokens[2:]))

    @classmethod
    def load(cls, path: Path) -> "Vocabulary":
        return cls(json.loads(path.read_text()))


def length(text: str) -> Fraction:
    """`8.` is a dotted eighth, `4..` a double-dotted quarter, `8:3` an eighth in a triplet, `16:6` a sixteenth of six
    in the time of four: a tuplet's number alone means that many in the time of the power of two below it. Any other
    tuplet names both numbers: `8:2/3` is an eighth of two in the time of three. The length in quarter notes."""
    text, _, tuplet = text.partition(":")
    value = Fraction(4, int(text.rstrip("."))) * (2 - Fraction(1, 2 ** text.count(".")))
    if not tuplet:
        return value
    count, _, time_of = tuplet.partition("/")
    if int(count) <= 0 or (time_of and int(time_of) <= 0):
        raise ValueError(text)
    return value * Fraction(int(time_of) if time_of else 2 ** (int(count).bit_length() - 1), int(count))


def split_counts(tokens: list[str]) -> list[str]:
    """`multirest-12` as `multirest`, `count-1`, `count-2`: the sign, then the number over it, digit by digit."""
    out: list[str] = []
    for token in tokens:
        if token.startswith("multirest-"):
            out += ["multirest", *(f"count-{digit}" for digit in token[10:])]
        else:
            out.append(token)
    return out


def tie_to_no_note(tokens: list[str]) -> bool:
    """Whether a tie in the line leads to a rest or a grace note, not to a note. No real page has such a tie; the first
    dataset's composer wrote them. A tie at the end of the line leads to the next line's first note."""
    for place, token in enumerate(tokens):
        if token == "tie":
            after = next((other for other in tokens[place + 1:] if other.split("-")[0] in ("note", "rest", "multirest", "grace")), "note-")
            if not after.startswith("note-"):
                return True
    return False


def events(tokens: list[str]) -> list[tuple]:
    """A page's tokens as the notes and rests the experiment scores: (pitch or "rest", length). Pitches are spelled as
    music21 spells them (`B-4`), a whole-measure rest has the length "measure", a grace note the length "grace".
    The two notes or chords a `tremolo` joins are each written at the length of both together, and each sounds for
    half of it: `note-C4_2 tremolo note-E4_2` is two quarter notes, as a score's file gives them."""
    found = []
    digits = None   # the count of a rest of several measures, while its digits are being read
    chord_at = None          # where in `found` the notes of the last chord begin; a note alone is a chord of one
    stacked = False          # the next note is played with the one before
    halved = rocking = False   # the chord being read is a tremolo's second; the next chord will be
    for token in [*tokens, ""]:
        kind, _, rest = token.partition("-")
        if digits is not None and kind == "count" and rest.isdigit():
            digits += rest
            continue
        if digits is not None:   # the count is over: a sign with no number, or one too long to be a count, is one measure's rest
            found.extend([("rest", "measure")] * (int(digits) if 0 < len(digits) < 4 else 1))
            digits = None
        if token == "multirest":
            digits = ""
            continue
        if token == "chord":
            stacked = True
            continue
        if token == "tremolo":
            for at in range(len(found) if chord_at is None else chord_at, len(found)):
                if not isinstance(found[at][1], str):   # a rest of whole measures or a grace note has no length to halve
                    found[at] = (found[at][0], found[at][1] / 2)
            rocking = chord_at is not None
            continue
        if token == "tie":
            continue
        joins, stacked = stacked and kind == "note" and chord_at is not None, False
        if not joins:
            chord_at, halved, rocking = None, rocking and kind == "note", False
        try:
            if kind == "note":
                pitch, _, long = rest.partition("_")
                long = length(long)
                if not joins:
                    chord_at = len(found)
                found.append((pitch.replace("b", "-"), long / 2 if halved else long))
            elif kind == "grace":
                found.append((rest.partition("_")[0].replace("b", "-"), "grace"))
            elif kind == "rest":
                found.append(("rest", "measure" if rest == "measure" else length(rest)))
            elif kind == "multirest":
                found.extend([("rest", "measure")] * int(rest))
        except (ValueError, ZeroDivisionError):
            continue   # a token that is not well formed is a wrong reading, and scores as a missing note
    return found


def key_alter(sharps: int, step: str) -> int:
    """What a key signature of `sharps` sharps (flats below zero) does to a letter: +1, -1 or 0."""
    if sharps > 0:
        return 1 if step in SHARP_ORDER[:sharps] else 0
    return -1 if step in SHARP_ORDER[::-1][:-sharps] else 0


def bottom_line(clef: str) -> int:
    """The note on the staff's bottom line under a clef token such as `clef-F4` or `clef-G2_8vb`, in steps above C0."""
    shape, line = clef[5], int(clef[6])
    return CLEF_NOTE[shape] - 2 * (line - 1) - (7 if clef.endswith("_8vb") else 0)


def place_name(steps: int) -> str:
    """A place on the staff, `steps` above its bottom line: L1 to L5 are its lines from the bottom, S1 to S4 the spaces
    between; S0 and S5 are the spaces just outside, and L0, L6 and beyond the ledger lines."""
    return f"L{steps // 2 + 1}" if steps % 2 == 0 else f"S{(steps + 1) // 2}"


def place_steps(name: str) -> int:
    number = int(name[1:])
    if name[0] == "L":
        return 2 * (number - 1)
    if name[0] == "S":
        return 2 * number - 1
    raise ValueError(name)


def key_sharps(text: str) -> int:
    """`2b` is two flats: -2."""
    return 0 if text == "0" else int(text[:-1]) * (1 if text[-1] == "#" else -1)


def to_positions(tokens: list[str], graces: bool = False) -> list[str]:
    """A line's pitch tokens as position tokens: each note as its place on the staff, after the accidental drawn before
    it, if any. An accidental is drawn where a note differs from what the key signature and the measure so far give
    its place, as the generator draws them; a note tied from the one before keeps that note's accidental unshown.
    In a chord each tied note is followed by its own `tie`, and the ties lead to the same places in the next chord.
    A line is read alone, so a note tied from the line before, whose accidental the page does not draw, gets one here.
    The first dataset's grace notes never show an accidental; from the truth's format 2 they show them as any note
    does, and `graces` says so."""
    out: list[str] = []
    bottom, sharps, marked = bottom_line("clef-G2"), 0, {}
    last = None
    opening: set[int] = set()   # the places tied on from the chord last read; a note alone is a chord of one
    held: set[int] = set()      # the places tied into the chord being read
    stacked = False             # the next note is played with the one before
    for token in tokens:
        kind, _, rest = token.partition("-")
        if kind in ("note", "grace"):
            pitch, _, long = rest.partition("_")
            letter, signs, octave = PITCH.fullmatch(pitch).groups()
            step, alter = letter.lower(), len(signs) * (-1 if signs.startswith("b") else 1)
            count = STEPS.index(step) + 7 * int(octave)
            if kind == "note" and not stacked:
                held, opening = opening, set()
            if kind == "note" or graces:
                if count in (held if kind == "note" else opening):
                    marked[count] = alter
                elif alter != marked.get(count, key_alter(sharps, step)):
                    out.append(ACCIDENTALS[alter])
                    marked[count] = alter
                last = count   # a `tie` belongs to the note or grace note right before it
            stacked = False
            out.append(f"{kind}-{place_name(count - bottom)}_{long}")
            continue
        stacked = token == "chord"
        if token == "tie":
            if last is not None:
                opening.add(last)
        elif kind == "clef":
            bottom = bottom_line(token)
        elif kind == "key":
            sharps = key_sharps(rest)
        elif token in BARLINES:
            marked = {}
        out.append(token)
    return out


def to_pitches(tokens: list[str]) -> list[str]:
    """Position tokens, as a reader gives them, as pitch tokens: each place on the staff named under the clef, and
    altered by the accidental before it, by an earlier one in its measure, by the note it is tied from, or by the key
    signature. A chord's tied notes each lead to the same place in the next chord. A line with no clef is read in the
    treble clef. A token that cannot be read is kept as it is."""
    out: list[str] = []
    bottom, sharps, marked = bottom_line("clef-G2"), 0, {}
    pending = last = None
    opening: dict[int, int] = {}   # the places tied on from the chord last read, and how each is altered
    held: dict[int, int] = {}      # the places tied into the chord being read
    stacked = False                # the next note is played with the one before
    for token in tokens:
        kind, _, rest = token.partition("-")
        if kind == "acc":
            pending = DRAWN.get(rest)
            continue
        if kind in ("note", "grace"):
            place, _, long = rest.partition("_")
            try:
                count = bottom + place_steps(place)
            except ValueError:
                out.append(token)
                pending, stacked = None, False
                continue
            step = STEPS[count % 7]
            if kind == "note" and not stacked:
                held, opening = opening, {}
            if pending is not None:
                alter = pending
            elif kind == "note" and count in held:
                alter = held[count]
            else:
                alter = marked.get(count, key_alter(sharps, step))
            if kind == "note" or pending is not None:
                marked[count] = alter   # an accidental holds to the end of the measure, a grace note's too
            last = (count, alter)   # a `tie` belongs to the note or grace note right before it
            pending, stacked = None, False
            out.append(f"{kind}-{step.upper()}{'#' * alter if alter > 0 else 'b' * -alter}{count // 7}_{long}")
            continue
        pending = None   # an accidental belongs to the note right after it
        stacked = token == "chord"
        if token == "tie":
            if last is not None:
                opening[last[0]] = last[1]
        elif kind == "clef" and re.fullmatch(r"clef-[GFC][1-5](_8vb)?", token):
            bottom = bottom_line(token)
        elif kind == "key" and re.fullmatch(r"key-(0|[1-7][#b])", token):
            sharps = key_sharps(rest)
        elif token in BARLINES:
            marked = {}
        out.append(token)
    return out
