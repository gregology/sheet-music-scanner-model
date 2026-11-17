"""Reads every picture given with a reader, through a line finder's lines, and saves each line's tokens.

Usage (on a machine with a GPU, from the repository's folder):
    uv run python tools/read_pages.py runs/r40k-pos runs/f30k reads.json <folder> [--count 2000] [--true-lines | --flatten]

The folder's `.png` and `.jpg` pictures are read, and those of the folders inside it, in the order of their names.
`reads.json` maps each picture's path inside the folder, without its ending, to its lines in reading order: the
tokens as the reader wrote them (places on the staff, for a position reader) and how sure it was of each. A run that
stops carries on when started again: pictures already in the file are not read twice. `--count` reads only the first
so many pictures.

`--true-lines` is for a generated dataset: each photo's lines are cut along the staff lines its truth gives
(`<name>.json` beside it) and the finder is not asked. The same photos read both ways show how many of the mistakes
are the finder's.

`--flatten` is for a generated dataset too: each photo is first warped so that its page's four corners, which the
truth gives, become an upright rectangle (`sheet_model.lines.flatten`), and the finder and the reader work on that
picture. It shows the most that finding a page's corners on the phone could give.

Reading is kept apart from scoring so that one reading on a GPU serves every count made from it afterwards:
`tools/sort_mistakes.py` sorts a reading's mistakes, `tools/measure_rule.py` tries a rule on it. README.md, "Three
measures".
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from sheet_model.finder import Finder, find
from sheet_model.lines import flatten, straighten
from sheet_model.model import Reader, read
from sheet_model.tokens import Vocabulary


def pictures_in(folder: Path) -> list[Path]:
    """The folder's pictures, by name. A dataset keeps a working folder beside its shards: nothing in a folder whose
    name starts with a dot is a picture of a page."""
    found = [p for p in folder.rglob("*") if p.suffix in (".png", ".jpg") and not any(part.startswith(".") for part in p.relative_to(folder).parts)]
    return sorted(found, key=lambda p: str(p.relative_to(folder)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("reader", type=Path)
    parser.add_argument("finder", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--true-lines", action="store_true")
    parser.add_argument("--flatten", action="store_true")
    arguments = parser.parse_args()
    if arguments.true_lines and arguments.flatten:
        parser.error("the true lines are in the photo's pixels, not the flattened page's: give one of --true-lines and --flatten")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    saved = torch.load(arguments.reader / "best.pt", map_location=device)
    settings = saved["settings"]
    vocabulary = Vocabulary.load(arguments.reader / "vocabulary.json")
    model = Reader(settings["classes"], settings["width"], settings["layers"], settings["heads"], settings["dropout"]).to(device).eval()
    model.load_state_dict(saved["model"])
    found = torch.load(arguments.finder / "best.pt", map_location=device)
    finder = Finder(tuple(found["settings"]["widths"])).to(device).eval()
    finder.load_state_dict(found["model"])

    reads = json.loads(arguments.out.read_text()) if arguments.out.exists() else {}
    paths = pictures_in(arguments.folder)[: arguments.count]
    for number, path in enumerate(paths):
        name = str(path.relative_to(arguments.folder).with_suffix(""))
        if name in reads:
            continue
        photo = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if arguments.flatten:
            page = json.loads(path.with_suffix(".json").read_text())["page"]
            grid = page["grid_px"]
            photo = flatten(photo, [grid[0][0], grid[0][-1], grid[-1][-1], grid[-1][0]], page["height_mm"] / page["width_mm"])
        if arguments.true_lines:
            staves = [np.array(system["staff"]["lines_px"]) for system in json.loads(path.with_suffix(".json").read_text())["systems"]]
        else:
            staves = find(photo, finder, device)
        lines = [straighten(photo, staff)[0] for staff in staves]
        reads[name] = []
        if lines:
            batch = torch.from_numpy(np.stack(lines).astype(np.float32) / 127.5 - 1)[:, None].to(device)
            with torch.no_grad():
                for numbers, sure in read(model(batch).float()):
                    reads[name].append({"tokens": vocabulary.decode(numbers), "sure": [round(float(value), 3) for value in sure]})
        if number % 50 == 49 or number == len(paths) - 1:
            arguments.out.write_text(json.dumps(reads))
            print(f"{number + 1} of {len(paths)}", flush=True)
    arguments.out.write_text(json.dumps(reads))


if __name__ == "__main__":
    main()
