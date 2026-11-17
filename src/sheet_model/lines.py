"""Cuts one line of music out of a photo and straightens it.

Given where a staff's five lines run in the photo, `straighten` makes a level picture of that line of music at a
fixed scale: the staff in the middle, room above and below for notes on ledger lines. A bent, tilted or distant
staff comes out the same size and shape as a flat one, which is all the line reader ever sees.
README.md, "The line picture".
"""

from __future__ import annotations

import cv2
import numpy as np

HEIGHT = 160          # px
WIDTH = 1600          # px: room for the longest line; a shorter one is padded on the right
SPACES = 14           # the picture's height in staff spaces: the staff's 4, and 5 above and below it
PX_PER_SPACE = HEIGHT / SPACES
MARGIN = 2.0          # staff spaces kept beyond each end of the staff
FLAT_WIDTH = 2400     # px: a flattened page's width; a letter page's staff lines are then about 20 px apart


def straighten(photo: np.ndarray, lines, paper: int | None = None) -> tuple[np.ndarray, int]:
    """`photo` is grey, 8 bits. `lines` is the staff's five lines, top to bottom, each a run of (x, y) points from left
    to right. Returns the line picture, HEIGHT by WIDTH, and how many of its columns hold the line."""
    middle, down, along = staff_path(lines)
    space = np.linalg.norm(down, axis=1)
    columns = int(round((along[-1] + 2 * MARGIN) * PX_PER_SPACE))
    at = np.arange(columns) / PX_PER_SPACE - MARGIN             # each column's place along the staff, in spaces

    def follow(values: np.ndarray) -> np.ndarray:
        """`values` at each column: between the points by interpolation, beyond the ends by carrying the end's slope on."""
        inside = np.stack([np.interp(at, along, values[:, k]) for k in range(2)], axis=1)
        before, after = at < 0, at > along[-1]
        first, last = (values[1] - values[0]) / max(along[1], 1e-9), (values[-1] - values[-2]) / max(along[-1] - along[-2], 1e-9)
        inside[before] = values[0] + at[before, None] * first
        inside[after] = values[-1] + (at[after, None] - along[-1]) * last
        return inside

    centre, step = follow(middle), follow(down)
    rows = (np.arange(HEIGHT) + 0.5 - HEIGHT / 2) / PX_PER_SPACE     # each row's height above or below the staff's middle
    source = centre[None, :, :] + rows[:, None, None] * step[None, :, :]

    # A staff much larger in the photo than in the line picture is shrunk first, so every photo pixel counts and
    # thin lines are not skipped over.
    shrink = float(np.median(space)) / PX_PER_SPACE
    x0, y0 = np.floor(source.reshape(-1, 2).min(axis=0)).astype(int) - 2
    x1, y1 = np.ceil(source.reshape(-1, 2).max(axis=0)).astype(int) + 3
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, photo.shape[1]), min(y1, photo.shape[0])
    if paper is None:
        paper = int(np.percentile(photo[y0:y1:4, x0:x1:4], 80)) if x1 > x0 and y1 > y0 else 255
    if x1 <= x0 or y1 <= y0:
        return np.full((HEIGHT, WIDTH), paper, np.uint8), 0
    patch = photo[y0:y1, x0:x1]
    scale = 1.0
    if shrink > 1.3:
        scale = 1 / shrink
        patch = cv2.resize(patch, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    map_x = ((source[..., 0] - x0 + 0.5) * scale - 0.5).astype(np.float32)
    map_y = ((source[..., 1] - y0 + 0.5) * scale - 0.5).astype(np.float32)
    line = cv2.remap(patch, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=paper)
    if columns > WIDTH:   # a line longer than the picture is squeezed to fit; rare
        line, columns = cv2.resize(line, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA), WIDTH
    picture = np.full((HEIGHT, WIDTH), paper, np.uint8)
    picture[:, :columns] = line
    return picture, columns


def staff_path(lines) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The staff's middle line at each point, one staff space as a step down the page there, and how far along the
    staff each point is, in staff spaces."""
    top, bottom = np.asarray(lines[0], np.float64), np.asarray(lines[-1], np.float64)
    middle, down = (top + bottom) / 2, (bottom - top) / 4
    space = np.linalg.norm(down, axis=1)
    steps = np.linalg.norm(np.diff(middle, axis=0), axis=1) / ((space[:-1] + space[1:]) / 2)
    return middle, down, np.concatenate([[0.0], np.cumsum(steps)])


def staff_rows() -> np.ndarray:
    """The rows of a line picture its five staff lines fall on, top to bottom."""
    return HEIGHT / 2 + (np.arange(5) - 2) * PX_PER_SPACE - 0.5


def flatten(photo: np.ndarray, corners, tall_per_wide: float) -> np.ndarray:
    """The page in a grey photo as if the camera were straight above it: the photo warped so that the page's four
    corners (top left, top right, bottom right, bottom left, in the photo's pixels) become an upright rectangle
    `FLAT_WIDTH` wide and `tall_per_wide` times as tall. Only the camera's angle is taken out: a bent page stays bent.
    What lies outside the photo is filled with middle grey. An investigation's step, not yet the app's: TRAINING.md,
    "Flattening the page"."""
    tall = round(FLAT_WIDTH * tall_per_wide)
    warp = cv2.getPerspectiveTransform(np.float32(corners), np.float32([[0, 0], [FLAT_WIDTH, 0], [FLAT_WIDTH, tall], [0, tall]]))
    return cv2.warpPerspective(photo, warp, (FLAT_WIDTH, tall), flags=cv2.INTER_LINEAR, borderValue=128)
