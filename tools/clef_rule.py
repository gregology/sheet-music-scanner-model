"""Scores each reader on the experiment's photos with a page clef rule, to measure how much of its mistakes are
misread clefs. TRAINING.md, "A page's clef".

Usage (on a machine with a GPU, from the repository's folder):
    uv run python tools/clef_rule.py <experiment folder> runs/clef_rule.json runs/r10k ...

Every line of every image is read once per run, as `sheet_model.real` reads it, and the page scored four ways:
- as read;
- page clef: each line read under the clef in force at the end of the line before it, as this rule set that line;
  the first line under the second line's opening clef;
- agree: a line keeps its opening clef when the line before ends with it or the line after opens with it, as read;
  otherwise it takes the clef the line before ends with;
- treble: every line's opening clef set to the treble clef. Right on every sheet but d, which changes clef, so on
  them it is the most a clef fix can recover.
A line read under a new clef goes back to places on the staff and forward to pitches again, so its key and
accidentals apply anew; a line whose clef is kept keeps the reader's own tokens. On the sheets but d it also counts
the lines whose opening clef is not the treble clef, and how many of them the page clef puts right or wrong.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from rapidfuzz.distance import Levenshtein

from sheet_model.model import Reader, read
from sheet_model.real import page_lines
from sheet_model.tokens import Vocabulary, events, to_pitches, to_positions

E, OUT = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(E))
from music21 import converter  # noqa: E402
from score import Reading, label_of  # noqa: E402

TREBLE = "clef-G2"


def opening(line):
    return line[0] if line and line[0].startswith("clef-") else None


def closing(line):
    return next((t for t in reversed(line) if t.startswith("clef-")), None)


def under(line, clef, positions):
    """The line's pitch tokens read under `clef` instead of the clef it opens with."""
    if clef is None or opening(line) == clef:
        return line if not positions else to_pitches(line)
    places = line if positions else to_positions(line)
    places = [clef] + (places[1:] if opening(places) else places)
    return to_pitches(places)


def page_rule(lines, positions):
    out, clefs = [], []
    for n, line in enumerate(lines):
        if n == 0:
            want = next((opening(other) for other in lines[1:] if opening(other)), None)
        else:
            want = clefs[-1]
        want = want or opening(line)
        out.append(under(line, want, positions))
        clefs.append(closing([want] + (line[1:] if opening(line) else line)) if want else closing(line))
    return out, [c for c in clefs]


def agree_rule(lines, positions):
    """A line keeps its opening clef when it matches the clef the line before ends with or the next line opens with,
    as read; otherwise it takes the clef the line before ends with (the first line: the next line's opening clef)."""
    out = []
    for n, line in enumerate(lines):
        before = closing(lines[n - 1]) if n else None
        after = opening(lines[n + 1]) if n + 1 < len(lines) else None
        mine = opening(line)
        if mine is not None and mine in (before, after):
            want = mine
        else:
            want = before or after or mine
        out.append(under(line, want, positions))
    return out


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pictures, truths, sheets = {}, {}, {}
    for path in sorted((E / "lines").glob("*.jpg")):
        pictures[path.stem] = page_lines(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE))
        sheet = label_of(path.stem)[0][0]
        sheets[path.stem] = sheet
        if sheet not in truths:
            truths[sheet] = Reading(converter.parse(E / "sheets" / f"{sheet}.musicxml")).events
    results = {}
    for run in map(Path, sys.argv[3:]):
        saved = torch.load(run / "best.pt", map_location=device)
        s = saved["settings"]
        vocabulary = Vocabulary.load(run / "vocabulary.json")
        positions = json.loads((run / "settings.json").read_text()).get("tokens") == "positions"
        model = Reader(s["classes"], s["width"], s["layers"], s["heads"], s["dropout"]).to(device).eval()
        model.load_state_dict(saved["model"])
        rows, lines_count = [], defaultdict(int)
        for stem, lines in pictures.items():
            with torch.no_grad():
                out = read(model(torch.from_numpy(np.stack(lines).astype(np.float32) / 127.5 - 1)[:, None].to(device)).float())
            read_lines = [vocabulary.decode(numbers) for numbers, _ in out]
            sure = [float(c[0]) if len(c) else 0.0 for _, c in out]
            as_read = [to_pitches(line) if positions else line for line in read_lines]
            ruled, rule_clefs = page_rule(read_lines, positions)
            treble = [under(line, TREBLE, positions) for line in read_lines]
            agreed = agree_rule(read_lines, positions)
            truth = truths[sheets[stem]]
            row = {"image": stem, "sheet": sheets[stem], "conditions": label_of(stem)[1], "notes": len(truth)}
            for name, tokens in (("read", as_read), ("page clef", ruled), ("agree", agreed), ("treble", treble)):
                row[name] = Levenshtein.distance(truth, events([t for line in tokens for t in line]))
            if sheets[stem] != "d":
                for n, line in enumerate(read_lines):
                    wrong = opening(line) != TREBLE
                    rule_clef = rule_clefs[n - 1] if n else next((opening(o) for o in read_lines[1:] if opening(o)), None)
                    lines_count["lines"] += 1
                    lines_count["misread"] += wrong
                    if wrong:
                        lines_count["misread, first line"] += n == 0
                        lines_count["misread, page clef puts right"] += (rule_clef or opening(line)) == TREBLE
                        row.setdefault("misread", []).append({"line": n, "clef": opening(line), "sure": round(sure[n], 2)})
                    else:
                        lines_count["read right, page clef puts wrong"] += (rule_clef or opening(line)) != TREBLE
            rows.append(row)
        results[run.name] = {"rows": rows, "lines": dict(lines_count)}
        groups = defaultdict(list)   # the mean of each image's share of notes wrong, as real.py reports it
        for row in rows:
            groups["d" if row["sheet"] == "d" else row["conditions"]].append(row)
        print(run.name, dict(lines_count), flush=True)
        groups["all"] = rows
        for group, members in sorted(groups.items()):
            means = [100 * np.mean([m[name] / m["notes"] for m in members]) for name in ("read", "page clef", "agree", "treble")]
            print(f"   {group:8s} read {means[0]:5.1f}%  page clef {means[1]:5.1f}%  agree {means[2]:5.1f}%  treble {means[3]:5.1f}%  ({len(members)} images)", flush=True)
    OUT.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
