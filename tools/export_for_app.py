"""Makes what the iPhone app bundles and tests against: the line finder and the line reader as Core ML programs, the
reader's vocabulary, and a made-up page with this code's reading of it, which the app's own code must reproduce.

Usage: uv run python tools/export_for_app.py <reader run> <finder run> <out folder> [<photo> ...]

Writes `Reader.mlpackage`, `Finder.mlpackage` and `reader.json` (the run's name, what its tokens are, and its
vocabulary in the order of the reader's outputs) to the folder. For each photo it writes `<name>.expected.json`: the
staves found, the line pictures' columns, each line's tokens as read and as pitches, and how sure the reader was.
The photo is read as the app reads it: grey by the BT.601 weights, and no larger than 4032 pixels on its long side.
README.md, "Core ML".
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from sheet_model.export import core_ml
from sheet_model.finder import LONG, SIDE, Finder, find
from sheet_model.lines import straighten
from sheet_model.model import Reader, read
from sheet_model.tokens import Vocabulary, to_pitches

LONGEST = 4032   # the app decodes a photo no larger than this on its long side


def main() -> None:
    reader_run, finder_run, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    saved = torch.load(reader_run / "best.pt", map_location="cpu")
    s = saved["settings"]
    reader = Reader(s["classes"], s["width"], s["layers"], s["heads"], s["dropout"]).eval()
    reader.load_state_dict(saved["model"])
    found = torch.load(finder_run / "best.pt", map_location="cpu")
    finder = Finder(tuple(found["settings"]["widths"])).eval()
    finder.load_state_dict(found["model"])
    vocabulary = Vocabulary.load(reader_run / "vocabulary.json")
    settings = json.loads((reader_run / "settings.json").read_text())
    positions = settings.get("tokens") == "positions"

    # The reader goes out at 32 bits: at 16 its answer is all zeros in the iOS simulator on an Intel Mac, where the
    # app's tests ran until 2026-11-17 (2026-11-14, Xcode 26.3; the same file is right on that Mac outside the
    # simulator). The M2 build host that replaced it has not been tried. The finder,
    # which is all convolutions, is right there at 16.
    program = core_ml(reader, half=False)
    program.short_description = f"Sheet Scanner's line reader, from {reader_run.name}"
    program.save(str(out / "Reader.mlpackage"))
    program = core_ml(finder, (LONG, SIDE), "marks")
    program.short_description = f"Sheet Scanner's line finder, from {finder_run.name}"
    program.save(str(out / "Finder.mlpackage"))
    (out / "reader.json").write_text(json.dumps({"reader": reader_run.name, "finder": finder_run.name, "positions": positions, "tokens": vocabulary.tokens}))

    for path in map(Path, sys.argv[4:]):
        colour = cv2.imread(str(path), cv2.IMREAD_COLOR)
        scale = LONGEST / max(colour.shape[:2])
        if scale < 1:
            colour = cv2.resize(colour, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        photo = cv2.cvtColor(colour, cv2.COLOR_BGR2GRAY)
        staves = find(photo, finder)
        lines = []
        for staff in staves:
            picture, columns = straighten(photo, staff)
            with torch.no_grad():
                numbers, sure = read(reader(torch.from_numpy(picture.astype(np.float32) / 127.5 - 1)[None, None]).float())[0]
            written = vocabulary.decode(numbers)
            lines.append({"staff": np.round(np.asarray(staff), 2).tolist(), "columns": columns, "written": written,
                          "pitches": to_pitches(written) if positions else written, "sure": [round(c, 4) for c in sure]})
        (out / f"{path.stem}.expected.json").write_text(json.dumps({"width": photo.shape[1], "height": photo.shape[0], "lines": lines}))
        print(path.name, photo.shape, len(lines), "lines;", sum(len(line["pitches"]) for line in lines), "tokens")


if __name__ == "__main__":
    main()
