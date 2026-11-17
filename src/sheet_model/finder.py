"""The line finder: a photo in, where each staff's five lines run through it out. README.md, "The line finder".

A small network looks at the whole photo, shrunk, and marks three things at every pixel: whether it lies on a staff,
how near it is to a staff's middle line, and how far apart that staff's lines are. Tracing, which is geometry, follows
each middle line across the picture and sets the five lines around it. Then each staff is straightened and its lines
moved to where the photo, at its full size, shows them: the network's picture is too small to place a line to a tenth
of a staff space, and the reader needs that.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
from torch import nn

from .bends import follow
from .lines import MARGIN, PX_PER_SPACE, WIDTH, staff_path, straighten

SIDE, LONG = 768, 1024   # the network's picture, in pixels: a portrait photo shrunk to fit, padded below and right
POINTS = 17              # points along each found line, as the dataset gives them
SPREAD = 0.35            # how wide the middle line's mark is, in staff spaces either side
PEAK = 0.4               # the least mark a middle line is traced on


def shrink(photo: np.ndarray) -> tuple[np.ndarray, float]:
    """The photo, grey, shrunk to fit the network's picture at its top left, the rest filled with its own middle grey;
    and the scale it was shrunk by."""
    height, width = photo.shape
    scale = min(SIDE / width, LONG / height)
    small = cv2.resize(photo, (max(1, round(width * scale)), max(1, round(height * scale))), interpolation=cv2.INTER_AREA)
    picture = np.full((LONG, SIDE), int(np.median(small[::4, ::4])), np.uint8)
    picture[:small.shape[0], :small.shape[1]] = small[:LONG, :SIDE]
    return picture, scale


def targets(staves, scale: float) -> np.ndarray:
    """What the network should mark for these staves, each five lines of (x, y) points in the photo: 3 by LONG by
    SIDE, for on a staff (0 or 1), near its middle line (1 on it, falling off over SPREAD spaces), and the log of
    its staff space in the network's pixels, where it is on a staff."""
    inside = np.zeros((LONG, SIDE), np.uint8)
    middle = np.zeros((LONG, SIDE), np.float32)
    space = np.zeros((LONG, SIDE), np.float32)
    for lines in staves:
        lines = np.asarray(lines, np.float64) * scale
        centre, down, _ = staff_path(lines)
        band = np.concatenate([lines[0] - down / 2, (lines[4] + down / 2)[::-1]])
        mask = np.zeros_like(inside)
        cv2.fillPoly(mask, [np.round(band * 16).astype(np.int32)], 1, cv2.LINE_8, shift=4)
        rows, columns = np.nonzero(mask)
        if len(rows) == 0:
            continue
        order = np.argsort(centre[:, 0])
        at_y = np.interp(columns, centre[order, 0], centre[order, 1])
        gap = np.interp(columns, centre[order, 0], np.linalg.norm(down, axis=1)[order])
        near = np.exp(-0.5 * ((rows - at_y) / np.maximum(SPREAD * gap, 0.7)) ** 2).astype(np.float32)
        inside[rows, columns] = 1
        middle[rows, columns] = np.maximum(middle[rows, columns], near)
        space[rows, columns] = np.log(np.maximum(gap, 0.5))
    return np.stack([inside.astype(np.float32), middle, space])


def block(inputs: int, outputs: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(inputs, outputs, 3, 1, 1, bias=False), nn.BatchNorm2d(outputs), nn.SiLU(),
                         nn.Conv2d(outputs, outputs, 3, 1, 1, bias=False), nn.BatchNorm2d(outputs), nn.SiLU())


class Finder(nn.Module):
    """A U-Net: four halvings to see a staff whole with its neighbours, and back up to every pixel."""

    def __init__(self, widths: tuple[int, ...] = (16, 32, 64, 128, 192)):
        super().__init__()
        self.settings = {"widths": list(widths)}
        self.down = nn.ModuleList([block(1, widths[0])] + [block(a, b) for a, b in zip(widths, widths[1:])])
        self.up = nn.ModuleList([block(b + a, a) for a, b in zip(widths, widths[1:])])
        self.pool, self.grow = nn.MaxPool2d(2), nn.Upsample(scale_factor=2, mode="nearest")
        self.out = nn.Conv2d(widths[0], 3, 1)

    def forward(self, pictures: torch.Tensor) -> torch.Tensor:
        """`pictures`: batch by 1 by LONG by SIDE, paper near +1 and ink near -1. Returns batch by 3 by LONG by SIDE:
        scores for on a staff and near its middle line, before the sigmoid, and the log of the staff space."""
        kept, x = [], pictures
        for number, step in enumerate(self.down):
            x = step(x if number == 0 else self.pool(x))
            kept.append(x)
        for step, skip in zip(reversed(self.up), reversed(kept[:-1])):
            x = step(torch.cat([self.grow(x), skip], 1))
        return self.out(x)


def trace(marks: np.ndarray, scale: float, step: int = 4) -> list[np.ndarray]:
    """Each staff the marks show, as five lines of POINTS (x, y) points in the photo, top to bottom.

    `marks` is the network's output for one picture, as probabilities and a staff space in pixels. In every `step`th
    column the middle lines are the peaks of the middle mark; a peak joins the staff whose last peak is near it in
    height, a few columns back, and a staff is kept if it runs at least twelve staff spaces."""
    inside, middle, space = marks
    staves: list[list[tuple[float, float, float]]] = []
    for x in range(step, SIDE - step, step):
        column = middle[:, x]
        peaks = np.flatnonzero((column[1:-1] >= PEAK) & (column[1:-1] >= column[:-2]) & (column[1:-1] > column[2:]) & (inside[1:-1, x] > 0.5)) + 1
        for y in peaks:
            a, b, c = column[y - 1], column[y], column[y + 1]
            curve = a - 2 * b + c
            fine = y + 0.5 * (a - c) / curve if curve < 0 else float(y)   # the top of the parabola through the three rows
            gap = float(space[y, x])
            best, distance = None, 0.6 * gap
            for staff in staves:
                last_x, last_y, last_gap = staff[-1]
                if 0 < x - last_x <= max(8 * last_gap, 3 * step):
                    slope = (staff[-1][1] - staff[-4][1]) / (staff[-1][0] - staff[-4][0]) if len(staff) >= 4 else 0.0
                    off = abs(fine - (last_y + slope * (x - last_x)))
                    if off < distance:
                        best, distance = staff, off
            if best is None:
                staves.append([(float(x), fine, gap)])
            else:
                best.append((float(x), fine, gap))
    found = []
    for staff in staves:
        xs, ys, gaps = (np.array(v) for v in zip(*staff))
        gap = float(np.median(gaps))
        if xs[-1] - xs[0] < 12 * gap:
            continue
        ys = np.array([np.median(ys[max(0, i - 2):i + 3]) for i in range(len(ys))])
        gaps = np.array([np.median(gaps[max(0, i - 4):i + 5]) for i in range(len(gaps))])
        at = np.linspace(xs[0], xs[-1], POINTS)
        y, g = np.interp(at, xs, ys), np.interp(at, xs, gaps)
        slope = np.gradient(y, at)
        normal = np.stack([-slope, np.ones_like(slope)], 1) / np.sqrt(1 + slope ** 2)[:, None]
        centre = np.stack([at, y], 1)
        found.append(np.stack([centre + (k - 2) * g[:, None] * normal for k in range(5)]) / scale)
    found.sort(key=lambda lines: float(lines[2][:, 1].mean()))
    return found


def refine(photo: np.ndarray, lines: np.ndarray) -> np.ndarray:
    """The staff's lines moved to where the photo shows them: straightened, the staff's shift and size are measured in
    slices along it, and each point is moved by what the slices near it found. First widely, to a twentieth of a
    space; then near, to a hundredth."""
    lines = np.asarray(lines, np.float64)
    for tries in (np.arange(-2.0, 2.0001, 0.05), np.arange(-0.3, 0.3001, 0.01)):
        picture, columns = straighten(photo, lines)
        centres, shifts, sizes = follow(picture, columns, tries)
        if len(centres) < 3:
            break
        centre, down, along = staff_path(lines)
        squeeze = min(1.0, WIDTH / round((along[-1] + 2 * MARGIN) * PX_PER_SPACE))   # straighten squeezes a line too long
        at = (along + MARGIN) * PX_PER_SPACE * squeeze
        shift, size = np.interp(at, centres, shifts), np.interp(at, centres, sizes)
        centre = centre + shift[:, None] * down
        lines = np.stack([centre + (k - 2) * size[:, None] * down for k in range(5)])
    return lines


def find(photo: np.ndarray, model: Finder, device: str = "cpu") -> list[np.ndarray]:
    """Every staff in a grey photo, as five lines of (x, y) points, top to bottom, refined against the photo."""
    picture, scale = shrink(photo)
    with torch.no_grad():
        scores = model(torch.from_numpy(picture.astype(np.float32) / 127.5 - 1)[None, None].to(device))[0].float().cpu().numpy()
    marks = np.stack([1 / (1 + np.exp(-scores[0])), 1 / (1 + np.exp(-scores[1])), np.exp(scores[2])])
    return [refine(photo, lines) for lines in trace(marks, scale)]
