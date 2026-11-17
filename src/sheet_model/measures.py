"""A measure's arithmetic: whether the notes read in a measure add up to its time signature, and a repair of triplet
marks where they do not. README.md, "The measure check".

A triplet printed with no mark looks like plain notes; only the measure's length tells them apart. A line reader sees
the time signature on a page's first line alone, so this works on a page: the time signature is carried from line to
line as it was read. The app's `MeasureCheck.swift` is this rule's Swift twin, held to it by `tools/measure_fixture.py`.
"""
from __future__ import annotations

from fractions import Fraction

from .tokens import length

BARS = {"barline", "barline-double", "barline-end", "repeat-start", "repeat-end"}
TIMED = ("note", "rest", "multirest", "grace", "chord", "count", "tremolo")
# The lengths a triplet is written in: the reader has `2:3`, `4:3` and `8:3` and no token for a sixteenth in one.
TRIPLETS = ("2", "4", "8")


def meter(token: str) -> Fraction | None:
    """A time signature's measure, in quarter notes."""
    text = token[5:]
    if text in ("C", "C|"):
        return Fraction(4)
    try:
        count, unit = text.split("/")
        return Fraction(int(count) * 4, int(unit))
    except (ValueError, ZeroDivisionError):
        return None


def measures(lines: list[list[str]]) -> list[tuple[int, list[int], Fraction | None]]:
    """A page's measures in reading order: (line, the places in the line of its notes and rests, the measure's length
    by the time signature last read). A line that ends without a barline ends its measure."""
    out, time = [], None
    for number, tokens in enumerate(lines):
        mine: list[int] = []
        for place, token in enumerate(tokens):
            if token.startswith("time-"):
                time = meter(token) or time
            if token in BARS:
                if mine:
                    out.append((number, mine, time))
                mine = []
            elif token.split("-")[0] in TIMED:
                mine.append(place)
        if mine:
            out.append((number, mine, time))
    return out


def parts(token: str) -> tuple[str, str] | None:
    """A note's or rest's written length and its tuplet: `8.` and ``, `8` and `3`. None for any other token."""
    kind = token.split("-")[0]
    if kind == "note" and "_" in token:
        text = token.split("_")[1]
    elif kind == "rest" and token != "rest-measure":
        text = token[5:]
    else:
        return None
    base, _, tuplet = text.partition(":")
    return base, tuplet


def total(tokens: list[str], places: list[int]) -> Fraction | None:
    """The measure's notes and rests added up, in quarter notes. A chord's notes sound together and count once; a
    grace note takes no time. The two notes or chords a tremolo joins are each written at the length of both
    together, so the second is not counted. None for a measure the sum says nothing about: a whole-measure rest, a
    rest of several measures, or a token whose length cannot be read."""
    sum_, stacked, rocked = Fraction(0), False, False
    for place in places:
        token = tokens[place]
        kind = token.split("-")[0]
        if kind == "chord":
            stacked = True
            continue
        if kind == "tremolo":
            rocked = True
            continue
        if kind in ("multirest", "count") or token == "rest-measure":
            return None
        if kind == "grace":
            continue
        if stacked:
            stacked = False
            continue
        if rocked and kind == "note":
            rocked = False
            continue
        rocked = False
        found = parts(token)
        try:
            sum_ += length(":".join(part for part in found if part))
        except (TypeError, ValueError, ZeroDivisionError):
            return None
    return sum_


def repair(tokens: list[str], sure: list[float], places: list[int], time: Fraction, limit: float = 1.0) -> dict[int, str] | None:
    """The cheapest change of triplet marks that makes the measure add up: {place: new token}. A triplet is three
    notes or rests in a row of one written length. The cost of a change is how sure the reader was of the tokens it
    changes, so the notes it doubted go first, and no token it was surer of than `limit` is changed. None when no
    change of marks makes the sum right, when two changes cost the same, when the measure is too long to search, or
    when it holds a tremolo between two notes: the rule was not measured on one."""
    if any(tokens[place] == "tremolo" for place in places):
        return None
    timed, under, stacked = [], {}, False   # `under`: the notes stacked on a chord's first note, which change with it
    for place in places:
        kind = tokens[place].split("-")[0]
        if kind == "chord":
            stacked = True
        elif stacked:
            stacked = False
            if timed and parts(tokens[place]) is not None:
                under.setdefault(timed[-1][0], []).append(place)
        elif parts(tokens[place]) is not None:
            timed.append((place, *parts(tokens[place])))
    if len(timed) > 20:
        return None
    best: list = [None, None, False]   # cost, plan, tied

    def search(at: int, sum_: Fraction, cost: float, plan: list) -> None:
        if best[0] is not None and cost > best[0]:
            return
        if at == len(timed):
            if sum_ == time and plan:
                if best[0] is not None and cost == best[0]:
                    best[2] = True
                else:
                    best[:] = [cost, list(plan), False]
            return
        place, base, tuplet = timed[at]
        if tuplet not in ("", "3"):   # six in the time of four and the like are left as read
            search(at + 1, sum_ + length(f"{base}:{tuplet}"), cost, plan)
            return
        try:
            plain = length(base)
        except (ValueError, ZeroDivisionError):
            return
        if not tuplet:
            search(at + 1, sum_ + plain, cost, plan)
        elif sure[place] <= limit:
            search(at + 1, sum_ + plain, cost + sure[place], plan + [(place, base, "")])
        group = timed[at:at + 3]
        if base.rstrip(".") in TRIPLETS and "." not in base and len(group) == 3 and all(other == base and mark in ("", "3") for _, other, mark in group):
            change = [(spot, base, "3") for spot, _, mark in group if not mark]
            if all(sure[spot] <= limit for spot, _, _ in change):
                search(at + 3, sum_ + 3 * length(f"{base}:3"), cost + sum(sure[spot] for spot, _, _ in change), plan + change)

    search(0, Fraction(0), 0.0, [])
    if best[1] is None or best[2]:
        return None
    out = {}
    for first, base, tuplet in best[1]:
        for place in [first, *under.get(first, [])]:
            head = tokens[place].split("_")[0] + "_" if tokens[place].startswith("note") else "rest-"
            out[place] = head + base + (f":{tuplet}" if tuplet else "")
    return out


def repaired(lines: list[dict], limit: float = 1.0) -> tuple[list[list[str]], int, int]:
    """A page's lines with every measure that does not add up repaired where `repair` finds a way: the new lines,
    how many measures failed the sum, and how many were changed. `lines` are a reading's: tokens and sureness."""
    out = [list(line["tokens"]) for line in lines]
    failed = changed = 0
    for number, places, time in measures([line["tokens"] for line in lines]):
        if time is None:
            continue
        sum_ = total(lines[number]["tokens"], places)
        if sum_ is None or sum_ == time:
            continue
        failed += 1
        plan = repair(lines[number]["tokens"], lines[number]["sure"], places, time, limit)
        if plan:
            changed += 1
            for place, token in plan.items():
                out[number][place] = token
    return out, failed, changed
