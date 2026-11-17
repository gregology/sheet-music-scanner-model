from fractions import Fraction

import cv2
import numpy as np
import torch

from sheet_model.lines import FLAT_WIDTH, HEIGHT, PX_PER_SPACE, WIDTH, flatten, staff_rows, straighten
from sheet_model.model import STEPS, Reader, read
from sheet_model.tokens import Vocabulary, events


def bent_staff(space=21.0, tilt=0.12, bow=60.0):
    """A photo of one staff that climbs and bows, and the path of its five lines."""
    photo = np.full((900, 2400), 235, np.uint8)
    xs = np.linspace(300, 2100, 17)
    middle = 450 + tilt * (xs - 1200) + bow * ((xs - 1200) / 900) ** 2
    lines = [[(float(x), float(y + (number - 2) * space)) for x, y in zip(xs, middle)] for number in range(5)]
    for line in lines:   # drawn to a sixteenth of a pixel, so each line's middle is where `lines` says
        cv2.polylines(photo, [np.round(np.array(line) * 16).astype(np.int32)], False, 20, 3, cv2.LINE_AA, shift=4)
    return photo, lines


# A straightener that left the staff tilted, bowed or at the wrong size would hand the reader lines unlike any it
# trained on, and every pitch would be read against lines that are not where it expects.
def test_a_bent_staff_comes_out_level_and_at_the_fixed_size():
    photo, lines = bent_staff()
    picture, columns = straighten(photo, lines)
    assert picture.shape == (HEIGHT, WIDTH) and 0 < columns <= WIDTH
    dark = 255 - picture[:, 40:columns - 40].astype(np.float32)
    for column in range(0, dark.shape[1], 50):
        rows = np.sort(np.argsort(dark[:, column])[-15:])                 # the darkest rows in this column: the five lines
        found = [rows[np.abs(rows - expected) < PX_PER_SPACE / 2].mean() for expected in staff_rows()]
        assert np.allclose(found, staff_rows(), atol=1.5), (column, found)


# A wrong length or spelling here would score a correct reading as wrong against the experiment's true scores.
def test_tokens_become_the_notes_the_experiment_scores():
    assert events(["clef-G2", "key-2b", "note-Bb4_8.", "tie", "note-F#5_8:3", "rest-4", "barline", "rest-measure", "grace-D5_8", "multirest-3"]) == [
        ("B-4", Fraction(3, 4)), ("F#5", Fraction(1, 3)), ("rest", Fraction(1)), ("rest", "measure"), ("D5", "grace"), ("rest", "measure"), ("rest", "measure"), ("rest", "measure")]


# CTC spells a token over several steps and separates repeats with blanks. Reading it wrongly doubles or drops notes.
def test_reading_drops_repeats_and_blanks_and_keeps_a_confidence_for_each_token():
    scores = torch.full((1, 6, 4), -10.0)
    for step, token in enumerate([2, 2, 0, 2, 3, 0]):
        scores[0, step, token] = 10.0
    tokens, sure = read(scores)[0]
    assert tokens == [2, 2, 3] and len(sure) == 3 and all(value > 0.99 for value in sure)


# The reader must give one row of scores for each of its steps, whatever the batch: the size Core ML will be fixed to.
def test_the_reader_gives_scores_for_every_step():
    vocabulary = Vocabulary(["clef-G2", "note-C4_4", "barline"])
    model = Reader(len(vocabulary), width=32, layers=1, heads=2).eval()
    with torch.no_grad():
        scores = model(torch.zeros(2, 1, HEIGHT, WIDTH))
    assert scores.shape == (2, STEPS, len(vocabulary))
    assert vocabulary.decode(vocabulary.encode(["note-C4_4", "never-seen"])) == ["note-C4_4", "<unknown>"]


# Corners taken in another order turn the page on its side or mirror it, and every staff with it: the finder then
# finds lines running down the picture, or the reader reads the music backwards.
def test_a_flattened_page_is_upright_with_its_top_left_corner_first():
    photo = np.full((1000, 1000), 235, np.uint8)
    corners = [(300, 200), (800, 300), (900, 900), (100, 700)]   # a page seen at an angle: top left, top right, bottom right, bottom left
    cv2.circle(photo, (345, 265), 12, 0, -1)   # an ink blot just inside the page's top left corner
    flat = flatten(photo, corners, 1.25)
    assert flat.shape == (round(FLAT_WIDTH * 1.25), FLAT_WIDTH)
    rows, columns = np.nonzero(flat < 100)
    assert rows.mean() < flat.shape[0] / 4 and columns.mean() < flat.shape[1] / 4
