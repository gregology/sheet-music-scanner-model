"""Scores a reader on the experiment's real photos, the test that counts.

Usage: uv run python -m sheet_model.real <run folder> <experiment folder> [--finder <finder run>]

The experiment is the folder I compared OCR engines in before training anything: my phone photos of printed sheets
whose true scores are known (`photos/`, `sheets/`), its scorer (`score.py`), and each photo's lines as homr found and
straightened them (`lines/`, one strip of 1280 by 256 px a line). Here each strip is brought to the reader's own line
picture, read, and the page's notes scored as the experiment scores every engine: the share of notes and rests wrong.
So the number compares with homr's and copisteria's on the same lines. The photos and true scores are published in
the test set (huggingface.co/datasets/gregology/sheet-music-scanner-eval), whose `evaluate.py` scores a published
model on them without the rest of the experiment.

homr's line finder stands in for ours unless `--finder <run>` names one of ours: then the lines are found in every
whole photo in `experiment/photos/` and every clean picture of a printed sheet in `experiment/out/images/`, which
run_engines.py draws, straightened and read. That takes in photos homr has no lines for, such as those of the
real-music sheets `i2` to `r2`, so the tables are kept apart by engraver: sheets `a1` to `h1` are Verovio's, `i2` to
`r2` MuseScore's. These photos are never trained on.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from rapidfuzz.distance import Levenshtein

from .lines import straighten
from .model import Reader, read
from .tokens import Vocabulary, events, to_pitches

STRIP = (1280, 256)   # homr's line picture
EDGE = 80             # the white border straighten_lines.py puts round the stacked strips


def find_staff(strip: np.ndarray) -> list | None:
    """The five staff lines in a level strip, as `straighten` wants them. homr centres the staff roughly and at a size
    of its choosing, so the lines are looked for: five evenly spaced dark rows, darker than the rows between them."""
    dark = 255.0 - cv2.GaussianBlur(strip, (0, 0), 1.0).astype(np.float32)
    profile = dark[:, 150:-150].mean(axis=1)
    best, found = 0.0, None
    for space in np.arange(8.0, 26.0, 0.25):
        for top in np.arange(20.0, STRIP[1] - 20.0 - 4 * space, 0.5):
            on = np.interp(top + space * np.arange(5), np.arange(len(profile)), profile)
            off = np.interp(top + space * (np.arange(4) + 0.5), np.arange(len(profile)), profile)
            score = on.min() - off.mean()   # every line present, and paper between them
            if score > best:
                best, found = score, (top, space)
    if found is None or best < 8:
        return None
    top, space = found
    rows = np.clip(np.round(top + space * np.arange(5)).astype(int), 0, STRIP[1] - 1)
    inked = (dark[rows] > 40).sum(axis=0) >= 3                          # columns where most of the five lines are drawn
    columns = np.flatnonzero(cv2.blur(inked.astype(np.float32)[None], (25, 1))[0] > 0.5)
    if len(columns) < 100:
        return None
    return [[(float(x), float(top + space * number)) for x in np.linspace(columns[0], columns[-1], 9)] for number in range(5)]


def page_lines(page: np.ndarray) -> list[np.ndarray]:
    """The reader's line pictures for one page of stacked strips."""
    pictures = []
    for top in range(EDGE, page.shape[0] - EDGE, STRIP[1]):
        strip = page[top:top + STRIP[1], EDGE:EDGE + STRIP[0]]
        staff = find_staff(strip) if strip.shape == (STRIP[1], STRIP[0]) else None
        if staff is not None:
            pictures.append(straighten(strip, staff)[0])
    return pictures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run", type=Path)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("--save", type=Path, default=None, help="a folder to save each page's line pictures in, to look at")
    parser.add_argument("--finder", type=Path, default=None, help="a line finder's run: find the lines in the experiment's photos with it, not homr")
    arguments = parser.parse_args()
    sys.path.insert(0, str(arguments.experiment))
    from music21 import converter   # needed only here: the experiment's truth is MusicXML
    from score import Reading, label_of

    device = "cuda" if torch.cuda.is_available() else "cpu"
    saved = torch.load(arguments.run / "best.pt", map_location=device)
    settings = saved["settings"]
    vocabulary = Vocabulary.load(arguments.run / "vocabulary.json")
    positions = json.loads((arguments.run / "settings.json").read_text()).get("tokens") == "positions"
    model = Reader(settings["classes"], settings["width"], settings["layers"], settings["heads"], settings["dropout"]).to(device).eval()
    model.load_state_dict(saved["model"])

    finder = None
    if arguments.finder:
        from .finder import Finder, find   # here, since the finder's refinement uses `bends`, which uses this module

        found = torch.load(arguments.finder / "best.pt", map_location=device)
        finder = Finder(tuple(found["settings"]["widths"])).to(device).eval()
        finder.load_state_dict(found["model"])

    if finder is None:
        images = sorted((arguments.experiment / "lines").glob("*.jpg"))
    else:
        images = sorted([*(arguments.experiment / "photos").glob("*.jpg"), *(arguments.experiment / "out" / "images").glob("*-render.png")], key=lambda p: p.stem)
    truths, rows = {}, []
    for path in images:
        sheet, conditions = label_of(path.stem)
        if sheet[0] not in truths:
            truths[sheet[0]] = Reading(converter.parse(arguments.experiment / "sheets" / f"{sheet[0]}.musicxml")).events
        truth = truths[sheet[0]]
        if finder is None:
            pictures = page_lines(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE))
        else:   # the whole photo, turned upright as its camera says, and our own line finder
            photo = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            pictures = [straighten(photo, lines)[0] for lines in find(photo, finder, device)]
        if arguments.save:
            arguments.save.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(arguments.save / f"{path.stem}.png"), np.vstack(pictures) if pictures else np.zeros((10, 10), np.uint8))
        tokens, sure = [], []
        if pictures:
            batch = torch.from_numpy(np.stack(pictures).astype(np.float32) / 127.5 - 1)[:, None].to(device)
            with torch.no_grad():
                for numbers, confidence in read(model(batch).float()):
                    line = vocabulary.decode(numbers)
                    tokens += to_pitches(line) if positions else line   # line by line: each starts with its own clef and key
                    sure += confidence
        found = events(tokens)
        rows.append({"image": path.stem, "sheet": sheet, "conditions": conditions, "lines": len(pictures), "notes_wrong": Levenshtein.distance(truth, found) / len(truth), "confidence": float(np.mean(sure)) if sure else 0.0})
    print("| Image | Lines | Notes wrong | Mean confidence |\n|---|---|---|---|")
    for row in rows:
        print(f"| {row['image']} | {row['lines']} | {100 * row['notes_wrong']:.1f}% | {row['confidence']:.2f} |")
    print("\n| Engraver | Conditions | Images | Notes wrong, mean | Worst |\n|---|---|---|---|---|")
    for engraver, conditions in sorted({(row["sheet"][1:], row["conditions"]) for row in rows}):
        group = [row["notes_wrong"] for row in rows if row["sheet"][1:] == engraver and row["conditions"] == conditions]
        print(f"| {engraver} | {conditions} | {len(group)} | {100 * sum(group) / len(group):.1f}% | {100 * max(group):.1f}% |")
    # with our finder, apart from the score on homr's lines, which it would otherwise replace
    name = f"real-{arguments.finder.name}.json" if arguments.finder else "real.json"
    (arguments.run / name).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
