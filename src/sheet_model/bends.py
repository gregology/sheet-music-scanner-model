"""How far each line picture's staff bends away from straight, and whether a reader's mistakes sit on the lines that
bend most. TRAINING.md, "Bent lines".

Usage: uv run python -m sheet_model.bends <experiment> <lines folder> [<run>...]

The staff's place is followed along each line in slices of about four staff spaces. A line's bend is how far the staff
strays, highest to lowest, once a straight tilt is taken out, in staff spaces; its size change is how much its staff
space grows or shrinks from one end to the other, as a share. Printed for the experiment's lines as homr found and
straightened them, for 400 prepared training lines and for the same lines after the training's variation; then, for
each run, the share of notes read wrong on the lines of photos and of clean pictures, by bend.
"""

from __future__ import annotations

import difflib
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from .lines import PX_PER_SPACE, straighten
from .model import Reader, read
from .real import EDGE, STRIP, find_staff
from .tokens import Vocabulary, events, to_pitches
from .train import varied

SLICE = 48                                   # pixels, about four staff spaces
SHIFTS = np.arange(-2.0, 2.0001, 0.05)       # the staff's shifts tried, in staff spaces
SIZES = np.arange(0.88, 1.1201, 0.01)        # the staff space's sizes tried, as a share of the expected
BINS = ("under 0.1", "0.1 to 0.2", "0.2 and over")


def follow(picture: np.ndarray, columns: int, tries: np.ndarray = SHIFTS) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The middle column of each slice that holds staff, and in each, the staff's shift from where it belongs, in staff
    spaces, and its size, as a share of the expected.

    Each slice scores every shift: all five lines must be dark and the rows between them light, so a shift by a whole
    space, which lays four lines right and one on paper, scores low. The shifts are followed along the line a step of at
    most 0.3 spaces from slice to slice, each step paid for, so one confusing slice cannot jump the staff: a staff bends,
    it does not jump. The size is then the one that best fits each slice near its shift. `tries` are the shifts tried,
    in staff spaces."""
    ink = 255 - picture.astype(np.float32)
    flat = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (31, 1)))   # staff lines, not notes
    middle = picture.shape[0] / 2 - 0.5
    on, off = (np.arange(5) - 2) * PX_PER_SPACE, (np.arange(4) - 1.5) * PX_PER_SPACE
    at = np.arange(picture.shape[0])
    slices = [(left + SLICE / 2, flat[:, left:left + SLICE].mean(axis=1)) for left in range(0, columns - SLICE, SLICE)]
    slices = [(c, p) for c, p in slices if p.max() >= 20]  # no staff: past the line's end, or a gap
    if not slices:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    centres, profiles = np.array([c for c, _ in slices]), [p for _, p in slices]

    def score(profile: np.ndarray, shifts: np.ndarray, sizes: np.ndarray) -> np.ndarray:
        centre = middle + shifts[:, None, None] * PX_PER_SPACE
        lines = np.interp(centre + sizes[None, :, None] * on, at, profile).min(axis=2)
        between = np.interp(centre + sizes[None, :, None] * off, at, profile).mean(axis=2)
        return lines - between                              # [shift, size]

    table = np.array([score(p, tries, np.ones(1))[:, 0] for p in profiles])
    step = np.abs(tries[:, None] - tries[None, :])
    cost = np.where(step <= 0.3 + 1e-9, 40.0 * step, np.inf)
    total, back = table[0].copy(), []
    for scores in table[1:]:
        options = total[:, None] - cost
        back.append(options.argmax(axis=0))
        total = options.max(axis=0) + scores
    path = [int(total.argmax())]
    for pointers in reversed(back):
        path.append(int(pointers[path[-1]]))
    shifts = tries[path[::-1]]
    sizes = []
    for profile, shift in zip(profiles, shifts):
        near = tries[np.abs(tries - shift) <= 0.15]
        found = score(profile, near, SIZES)
        sizes.append(SIZES[np.unravel_index(found.argmax(), found.shape)[1]])
    return centres, shifts, np.array(sizes)


def measure(picture: np.ndarray, columns: int) -> tuple[float, float]:
    """A line's bend, in staff spaces, and its size change, as a share; not a number if too little of it holds staff."""
    _, shifts, sizes = follow(picture, columns)
    if len(shifts) < 3:
        return float("nan"), float("nan")
    x = np.arange(len(shifts)) * SLICE / PX_PER_SPACE
    rest = shifts - np.polyval(np.polyfit(x, shifts, 1), x)
    return float(rest.max() - rest.min()), float(abs(np.polyfit(x, sizes, 1)[0]) * x[-1])


def inked(picture: np.ndarray) -> int:
    """How many columns hold the line: up to its last column with ink."""
    dark = np.flatnonzero((255 - picture).max(axis=0) > 60)
    return int(dark[-1]) + 1 if len(dark) else 0


def summary(name: str, values: list[tuple[float, float]]) -> str:
    values = np.array([v for v in values if v[0] == v[0]])
    bend, size = values[:, 0], values[:, 1]
    return (f"{name}, {len(values)} lines: bend median {np.median(bend):.2f}, 90th {np.percentile(bend, 90):.2f}, "
            f"most {bend.max():.2f} spaces; size change median {np.median(size):.3f}, 90th {np.percentile(size, 90):.3f}, most {size.max():.3f}")


def main() -> None:
    experiment, folder, runs = Path(sys.argv[1]), Path(sys.argv[2]), [Path(p) for p in sys.argv[3:]]
    sys.path.insert(0, str(experiment))
    from music21 import converter   # the experiment's truth is MusicXML
    from score import Reading, label_of

    pages = []   # (sheet, conditions, [(picture, bend)])
    for path in sorted((experiment / "lines").glob("*.jpg")):
        page = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        lines = []
        for top in range(EDGE, page.shape[0] - EDGE, STRIP[1]):
            strip = page[top:top + STRIP[1], EDGE:EDGE + STRIP[0]]
            staff = find_staff(strip) if strip.shape == (STRIP[1], STRIP[0]) else None
            if staff is not None:
                picture = straighten(strip, staff)[0]
                lines.append((picture, measure(picture, inked(picture))))
        pages.append((*label_of(path.stem), lines))
    for conditions in sorted({c for _, c, _ in pages}):
        print(summary(f"test {conditions}", [m for _, c, lines in pages if c == conditions for _, m in lines]))
    print(summary("test, all", [m for _, _, lines in pages for _, m in lines]))

    records = [json.loads(t) for listing in sorted(folder.glob("*.jsonl"))[:10] for t in listing.read_text().splitlines()]
    random.Random(1).shuffle(records)
    pictures = [(cv2.imread(str(folder / r["image"]), cv2.IMREAD_GRAYSCALE), r["columns"]) for r in records[:400]]
    print(summary("training, as prepared", [measure(p, inked(p)) for p, _ in pictures]))
    random.seed(2)
    bent = [np.clip(varied(p, columns), 0, 255).astype(np.uint8) for p, columns in pictures]
    print(summary("training, as varied", [measure(p, inked(p)) for p in bent]), flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    truths = {}
    for run in runs:
        saved = torch.load(run / "best.pt", map_location=device)
        s = saved["settings"]
        vocabulary = Vocabulary.load(run / "vocabulary.json")
        positions = json.loads((run / "settings.json").read_text()).get("tokens") == "positions"
        model = Reader(s["classes"], s["width"], s["layers"], s["heads"], s["dropout"]).to(device).eval()
        model.load_state_dict(saved["model"])
        counts = {(kind, b): [0, 0] for kind in ("photos", "clean pictures") for b in BINS}
        for sheet, conditions, lines in pages:
            if not lines:
                continue
            if sheet[0] not in truths:
                truths[sheet[0]] = Reading(converter.parse(experiment / "sheets" / f"{sheet[0]}.musicxml")).events
            batch = torch.from_numpy(np.stack([p for p, _ in lines]).astype(np.float32) / 127.5 - 1)[:, None].to(device)
            with torch.no_grad():
                readings = read(model(batch).float())
            found, owner = [], []    # every note read, and the line it was read on
            for number, (numbers, _) in enumerate(readings):
                tokens = vocabulary.decode(numbers)
                notes = events(to_pitches(tokens) if positions else tokens)
                found += notes
                owner += [number] * len(notes)
            wrong = [0] * len(lines)
            for op, i0, i1, j0, j1 in difflib.SequenceMatcher(None, truths[sheet[0]], found, autojunk=False).get_opcodes():
                if op != "equal" and owner:
                    wrong[owner[min(j0, len(owner) - 1)]] += max(i1 - i0, j1 - j0)
            kind = "clean pictures" if conditions == "render" else "photos"
            for number, (_, (bend, _)) in enumerate(lines):
                b = BINS[0] if not bend >= 0.1 else BINS[1] if bend < 0.2 else BINS[2]
                counts[kind, b][0] += wrong[number]
                counts[kind, b][1] += owner.count(number)
        for (kind, b), (w, n) in counts.items():
            print(f"{run.name}, {kind}, lines bent {b}: {w} wrong of {n} notes read, {100 * w / max(n, 1):.1f}%")


if __name__ == "__main__":
    main()
