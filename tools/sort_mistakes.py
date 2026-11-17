"""Scores a reading of many pages against their true scores, and ranks what was drawn where the mistakes fall.

Usage, from the repository's folder, after `tools/read_pages.py` wrote `reads.json` for the same folder:
    uv run python tools/sort_mistakes.py reads.json --survey <folder> --run runs/r40k-pos [--save out.json]
    uv run python tools/sort_mistakes.py reads.json --dataset <folder> [--save out.json]

`--survey`: each picture `<name>.png` has its true score beside it as `<name>.musicxml`
(the test set's `survey/` is one, and its `scripts/make_survey.py` writes it). A reading of pictures named `<name>.p1`,
`<name>.p2` and on, as `experiment/make_survey_verovio.py` draws them, is scored against the same folder. `--run` gives the reader's vocabulary, to
tag notes of a length it has no token for. `--dataset`: each photo `<name>.jpg` has the generator's truth beside it
as `<name>.json`.

It prints the share of notes wrong, the mistakes by kind, and for every tag (what the page shows at a note) the notes
that carry it, the mistakes on them, and how many more mistakes that is than the other notes would make. Then the
gap list: the same ranking made again and again, each time without the notes of the tag that came first. For a
dataset it also prints the score of each kind of photo (`kind_of_photo`), and how many mistakes sit on lines whose
clef or key signature was read wrong. README.md, "Three measures".
"""
import argparse
import json
import math
from collections import Counter
from pathlib import Path

from sheet_model.finder import LONG, SIDE
from sheet_model.mistakes import join_pages, mistakes, openings, peeled, rank, read_events, tags_of_example, tags_of_musicxml
from sheet_model.tokens import length


SQUARE_ON = 20   # degrees of camera tilt: past 20 the pilot's share of notes wrong rose from under 1% to 2.5% and on


def kind_of_photo(truth: dict) -> str:
    """Which of the two accuracy goals a photo is held to. "Well taken": a flat sheet, in bright or indoor light, the
    camera within `SQUARE_ON` degrees of straight above it. "Bent, dim or at an angle": the rest. A photo with any
    symbol of the music partly outside the picture is held to neither: no reader can name a clef it cannot see."""
    if any(symbol.get("cut_off") for system in truth["systems"] for symbol in system["symbols"]):
        return "music cut off by the photo's edge"
    if truth["sheet"]["shape"] == "flat" and truth["camera"]["light"] in ("bright", "indoor") and truth["shot"]["tilt_deg"] <= SQUARE_ON:
        return "well taken"
    return "bent, dim or at an angle"


def smallest_space(truth: dict) -> float:
    """How far apart the lines of the photo's smallest staff are in the line finder's picture, in pixels: the photo
    shrunk to fit `SIDE` by `LONG`."""
    scale = min(SIDE / truth["photo"]["width"], LONG / truth["photo"]["height"])
    heights = []
    for system in truth["systems"]:
        top, bottom = system["staff"]["lines_px"][0], system["staff"]["lines_px"][4]
        heights.append(min(math.dist(a, b) for a, b in zip(top, bottom)))
    return round(min(heights) / 4 * scale, 2)


def summary(rows: list[dict]) -> str:
    notes, wrong = sum(row["notes"] for row in rows), sum(row["wrong"] for row in rows)
    mean = sum(row["wrong"] / row["notes"] for row in rows) / len(rows)
    clean = sum(1 for row in rows if not row["wrong"])
    return f"{len(rows)} pages, {notes} notes, {wrong} wrong: {100 * wrong / notes:.2f}% of all notes, {100 * mean:.2f}% as the mean of the pages; {clean} pages with no note wrong ({100 * clean / len(rows):.0f}%)"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("reads", type=Path)
    parser.add_argument("--survey", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--run", type=Path, help="the reader's run, for its vocabulary")
    parser.add_argument("--save", type=Path)
    parser.add_argument("--top", type=int, default=40)
    arguments = parser.parse_args()
    reads = join_pages(json.loads(arguments.reads.read_text()))
    lengths = set()
    if arguments.run:
        for token in json.loads((arguments.run / "vocabulary.json").read_text()):
            if token.startswith(("note-L2_", "rest-")) and token != "rest-measure":
                lengths.add(length(token.split("_")[1] if token.startswith("note") else token[5:]))

    rows, pages, kinds, skipped, starts = [], [], Counter(), [], Counter()
    for name, lines in sorted(reads.items()):
        folder = arguments.survey or arguments.dataset
        try:
            if arguments.survey:
                truth, tags = tags_of_musicxml(folder / f"{name}.musicxml", lengths)
                page = []
            else:
                example = json.loads((folder / f"{name}.json").read_text())
                truth, tags, page = tags_of_example(example)
                page.append(kind_of_photo(example))
                for how, notes, wrong, mended in openings([system["tokens"] for system in example["systems"]], [line["tokens"] for line in lines]) or []:
                    starts.update({(how, "lines"): 1, (how, "notes"): notes, (how, "wrong"): wrong, (how, "mended"): mended})
        except Exception as failure:   # a true score music21 cannot read is a page left out, and said so
            skipped.append((name, f"{type(failure).__name__}: {failure}"))
            continue
        if not truth:
            skipped.append((name, "no notes in the true score"))
            continue
        found = mistakes(truth, read_events([line["tokens"] for line in lines]))
        kinds.update(kind for _, kind in found)
        pages.append(([mine + page for mine in tags], found))
        rows.append({"page": name, "lines": len(lines), "notes": len(truth), "wrong": len(found), "tags": page, "kinds": dict(Counter(kind for _, kind in found))})
        if arguments.dataset:
            rows[-1]["tilt"], rows[-1]["space"] = example["shot"]["tilt_deg"], smallest_space(example)

    print(summary(rows))
    for group in sorted({tag for row in rows for tag in row["tags"]}):
        print(f"- {group}: {summary([row for row in rows if group in row['tags']])}")
    shares = sorted(row["wrong"] / row["notes"] for row in rows)
    print("Pages by share of notes wrong: " + ", ".join(f"{label} {sum(1 for s in shares if low <= s < high)}" for label, low, high in (("none", 0, 1e-9), ("under 1%", 1e-9, 0.01), ("1 to 5%", 0.01, 0.05), ("5 to 20%", 0.05, 0.2), ("over 20%", 0.2, 9e9))))
    print("\n| Kind of mistake | Count |\n|---|---|")
    for kind, count in kinds.most_common():
        print(f"| {kind} | {count} |")
    if starts:
        print("\n| How the line's start was read | Lines | Notes | Wrong | Share wrong | Wrong with the start put right |\n|---|---|---|---|---|---|")
        for how in ("clef and key right", "key wrong", "clef wrong"):
            lines_, notes, wrong, mended = (starts[(how, what)] for what in ("lines", "notes", "wrong", "mended"))
            print(f"| {how} | {lines_} | {notes} | {wrong} | {100 * wrong / max(notes, 1):.2f}% | {mended} |")
    ranked = rank(pages)
    print("\n| What is drawn | Pages | Notes | Wrong | Share wrong | More than the rest would make |\n|---|---|---|---|---|---|")
    for row in ranked[: arguments.top]:
        print(f"| {row['tag']} | {row['pages']} | {row['notes']} | {row['wrong']} | {100 * row['share']:.1f}% | {row['excess']:.0f} |")
    gaps = peeled(pages)
    print("\nThe gap list: each mistake counted once, for the tag that explains most.")
    print("\n| What is drawn | Pages | Notes | Wrong | Share wrong | Notes left | Wrong left | Share wrong left |\n|---|---|---|---|---|---|---|---|")
    for row in gaps:
        left = row["left"]
        print(f"| {row['tag']} | {row['pages']} | {row['notes']} | {row['wrong']} | {100 * row['share']:.1f}% | {left['notes']} | {left['wrong']} | {100 * left['wrong'] / max(left['notes'], 1):.2f}% |")
    if skipped:
        print(f"\n{len(skipped)} pages left out:")
        for name, why in skipped[:20]:
            print(f"- {name}: {why[:120]}")
    if arguments.save:
        arguments.save.write_text(json.dumps({"pages": rows, "kinds": dict(kinds), "tags": ranked, "gaps": gaps, "skipped": skipped, "starts": {f"{how}: {what}": count for (how, what), count in starts.items()}}))


if __name__ == "__main__":
    main()
