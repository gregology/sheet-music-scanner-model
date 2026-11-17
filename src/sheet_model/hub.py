"""Load a published finder and reader, and read a photo of a page of music into its lines.

    from sheet_model.hub import load

    scanner = load()                 # gregology/sheet-music-scanner on Hugging Face, at REVISION
    page = scanner.read("page.jpg")
    for line in page.lines:
        print(line.pitches)

`load` also takes a local folder laid out as the Hugging Face repo is: `config.json`, `vocabulary.json`,
`reader.safetensors` and `finder.safetensors`. `tools/publish.py` writes that folder from two training runs.

A photo is read as the iPhone app reads it: grey by the BT.601 weights, no larger than 4032 pixels on its long side,
each staff found, straightened and read, then each measure's arithmetic checked (README.md, "The measure check").
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
from safetensors.torch import load_file

from .finder import Finder, find
from .lines import straighten
from .measures import repaired
from .model import Reader, read
from .tokens import Vocabulary, to_pitches

REPO = "gregology/sheet-music-scanner"
REVISION = "v1.0"
LONGEST = 4032   # px: the app decodes a photo no larger than this on its long side
FILES = ["config.json", "vocabulary.json", "reader.safetensors", "finder.safetensors"]


@dataclass
class Line:
    staff: np.ndarray        # the staff's five lines in the photo, top to bottom: 5 by 17 by (x, y)
    picture: np.ndarray      # the straightened line the reader saw: 160 by 1600, grey
    written: list[str]       # the tokens as the reader wrote them: places on the staff, before the measure check
    sure: list[float]        # how sure the reader was of each written token
    tokens: list[str]        # the written tokens after the measure check
    pitches: list[str]       # the same as pitches, the dataset's own tokens


@dataclass
class Page:
    width: int
    height: int
    lines: list[Line] = field(default_factory=list)
    failed: int = 0          # measures whose notes did not add up to the time signature, as read
    mended: int = 0          # of those, how many the measure check changed


class Scanner:
    def __init__(self, folder: Path, device: str = "cpu"):
        self.folder, self.device = Path(folder), device
        self.config = json.loads((self.folder / "config.json").read_text())
        reader, finder = self.config["reader"], self.config["finder"]
        self.vocabulary = Vocabulary.load(self.folder / "vocabulary.json")
        self.reader = Reader(reader["classes"], reader["width"], reader["layers"], reader["heads"], reader["dropout"])
        self.reader.load_state_dict(load_file(self.folder / "reader.safetensors"))
        self.reader.to(device).eval()
        self.finder = Finder(tuple(finder["widths"]))
        self.finder.load_state_dict(load_file(self.folder / "finder.safetensors"))
        self.finder.to(device).eval()
        self.positions = self.config["tokens"] == "positions"

    def grey(self, photo, longest: int | None = LONGEST) -> np.ndarray:
        """A path, or an RGB or grey array, as the grey photo the app would read: no larger than `longest` pixels on
        its long side. `longest=None` keeps the photo's own size, as the published scores were measured."""
        if isinstance(photo, (str, Path)):
            colour = cv2.imread(str(photo), cv2.IMREAD_COLOR)   # turned upright by its EXIF orientation
            if colour is None:
                raise ValueError(f"not a picture: {photo}")
            picture = cv2.cvtColor(colour, cv2.COLOR_BGR2GRAY)
        else:
            picture = np.asarray(photo)
            if picture.ndim == 3 and picture.shape[2] == 4:
                picture = picture[:, :, :3]
            if picture.ndim == 3:
                picture = cv2.cvtColor(picture, cv2.COLOR_RGB2GRAY)
        scale = (longest or max(picture.shape[:2])) / max(picture.shape[:2])
        if scale < 1:
            picture = cv2.resize(picture, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        return picture.astype(np.uint8)

    def read(self, photo, check: bool = True, limit: float = 1.0, longest: int | None = LONGEST) -> Page:
        """Every line of music in the photo, top to bottom. `photo` is a path, or an RGB or grey array as PIL and
        Gradio give it; convert an OpenCV (BGR) picture first. `check` runs the measure check, which may change a
        triplet mark however sure the reader was, as the app does; a `limit` below 1 forbids changing a token the
        reader was surer of than that. `longest` is as for `grey`."""
        picture = self.grey(photo, longest)
        page = Page(picture.shape[1], picture.shape[0])
        staves = find(picture, self.finder, self.device)
        pictures = [straighten(picture, staff)[0] for staff in staves]
        if not pictures:
            return page
        batch = torch.from_numpy(np.stack(pictures).astype(np.float32) / 127.5 - 1)[:, None].to(self.device)
        with torch.no_grad():
            readings = read(self.reader(batch).float())
        lines = [{"tokens": self.vocabulary.decode(numbers), "sure": sure} for numbers, sure in readings]
        checked = [line["tokens"] for line in lines]
        if check:
            checked, page.failed, page.mended = repaired(lines, limit)
        for staff, line_picture, line, tokens in zip(staves, pictures, lines, checked):
            page.lines.append(Line(np.asarray(staff), line_picture, line["tokens"], [float(s) for s in line["sure"]], tokens,
                                   to_pitches(tokens) if self.positions else tokens))
        return page


def load(where: str | Path = REPO, revision: str = REVISION, device: str | None = None) -> Scanner:
    """The scanner in a local folder, or in a Hugging Face model repo at a tag, branch or commit."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if Path(where).is_dir():
        return Scanner(Path(where), device)
    from huggingface_hub import snapshot_download

    return Scanner(Path(snapshot_download(str(where), revision=revision, allow_patterns=FILES)), device)
