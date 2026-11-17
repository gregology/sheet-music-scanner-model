import random

import numpy as np

from sheet_model.lines import HEIGHT, PX_PER_SPACE, WIDTH, staff_rows
from sheet_model.bends import measure
from sheet_model.train import curve, warped

LEVEL = np.float32([[1, 0, 0], [0, 1, 0]])   # a move that leaves the picture where it is


def level_staff() -> np.ndarray:
    """A line picture holding a level staff where the reader expects it, and nothing else: each line two pixels
    thick, its middle exactly on its row."""
    rows = np.arange(HEIGHT, dtype=np.float32)[:, None]
    ink = np.clip(1.5 - np.abs(rows - staff_rows()[None, :]), 0, 1).sum(axis=1)
    return np.broadcast_to(235 - 215 * np.clip(ink, 0, 1)[:, None], (HEIGHT, WIDTH)).astype(np.uint8)


def staff_at(picture: np.ndarray, column: int, near: np.ndarray) -> np.ndarray:
    """The middle of each line in one column: the centre of the ink within half a space of where it is looked for."""
    ink = 235.0 - picture[:, column]
    rows = np.arange(HEIGHT)
    found = []
    for row in near:
        close = np.abs(rows - row) < PX_PER_SPACE / 2
        found.append((rows[close] * ink[close]).sum() / max(ink[close].sum(), 1e-9))
    return np.array(found)


# A bend in the wrong units, a tilt left in it or a size that ignores what was asked would train the reader on lines
# bent far more or less than the photos' lines, which bend up to 0.39 staff spaces.
def test_a_curve_spans_the_size_asked_and_adds_no_tilt():
    along = np.linspace(0, 1, 1200)
    for seed in range(20):
        bend = curve(along, 0.3 * PX_PER_SPACE, random.Random(seed))
        assert np.isclose(bend.max() - bend.min(), 0.3 * PX_PER_SPACE, rtol=0.02)
        assert abs(np.polyfit(along, bend, 1)[0]) < 0.05 * PX_PER_SPACE


# A map that moved the staff the other way, by a different amount, or about the wrong row would bend and grow the
# training lines unlike the curve says, so the measured bend would not be the bend trained on.
def test_a_warped_staff_follows_the_bend_and_grows_about_its_middle():
    along = np.clip(np.arange(WIDTH) / 1200, 0, 1)
    bend = curve(along, 0.4 * PX_PER_SPACE, random.Random(3))
    growth = 1 + 0.08 * (along - 0.5)
    picture = warped(level_staff(), LEVEL, bend, growth, 235)
    middle = HEIGHT / 2 - 0.5
    for column in range(20, 1200, 60):
        expected = middle + (staff_rows() - middle) * growth[column] + bend[column]
        found = staff_at(picture, column, expected)
        assert np.allclose(found, expected, atol=0.3), (column, found, expected)


# A measure that misread a bend, or saw one in a level staff, would size the training's bends wrong and pin the
# readers' mistakes on lines that are not bent. It follows the staff in slices four spaces wide, so it smooths a sharp
# bend a little: within a tenth of a space, the width of the bins mistakes are counted in.
def test_the_bend_measured_is_the_bend_made():
    assert np.allclose(measure(level_staff(), 1200), 0)
    along = np.clip(np.arange(WIDTH) / 1200, 0, 1)
    for seed, size in [(1, 0.2), (2, 0.4)]:
        bend = curve(along, size * PX_PER_SPACE, random.Random(seed))
        picture = warped(level_staff(), LEVEL, bend, 1 + 0.08 * (along - 0.5), 235)
        bent, grown = measure(picture, 1200)
        assert abs(bent - size) <= 0.1 and abs(grown - 0.08) <= 0.02, (size, bent, grown)
