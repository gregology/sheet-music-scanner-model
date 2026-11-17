import numpy as np
import torch

from sheet_model.export import core_ml
from sheet_model.finder import LONG, SIDE, Finder, refine, shrink, targets, trace
from test_reader import bent_staff


def page_of_staves():
    """A photo the size of the dataset's, with three staves across it, climbing and bowing, and their lines."""
    photo = np.full((4032, 3024), 235, np.uint8)
    staves = []
    for top in (900, 1900, 2900):
        part, lines = bent_staff(space=20.0, tilt=0.05, bow=40.0)
        lines = [[(x + 300, y + top - 450) for x, y in line] for line in lines]
        photo[top - 450:top + 450, 300:2700] = part[:, :2400]
        staves.append(lines)
    return photo, staves


def error_in_spaces(found, truth) -> float:
    """The largest distance, down the page, between a found line and its true line, in staff spaces."""
    truth = np.asarray(truth, np.float64)
    space = float(np.median(np.linalg.norm(truth[4] - truth[0], axis=1) / 4))
    worst = 0.0
    for got, want in zip(found, truth):
        inner = (want[:, 0] >= got[0, 0]) & (want[:, 0] <= got[-1, 0])
        worst = max(worst, float(np.abs(np.interp(want[inner, 0], got[:, 0], got[:, 1]) - want[inner, 1]).max()))
    return worst / space


# Tracing that set the lines in the wrong place or at the wrong spacing, lost a staff or split one in two would hand
# the straightener lines unlike the staff, even from a network that marked the picture perfectly.
def test_perfect_marks_trace_back_to_every_staff():
    photo, staves = page_of_staves()
    picture, scale = shrink(photo)
    assert picture.shape == (LONG, SIDE)
    marks = targets(staves, scale)
    marks[2] = np.exp(marks[2]) * (marks[0] > 0)
    found = trace(marks, scale)
    assert len(found) == 3
    for got, want in zip(found, staves):
        assert error_in_spaces(got, want) < 0.1
        assert error_in_spaces(refine(photo, got), want) < 0.06
        assert got[2][0, 0] < want[2][0][0] + 2 * 20 and got[2][-1, 0] > want[2][-1][0] - 2 * 20   # the whole staff, near enough


# A refinement that moved the lines the wrong way, or not to the photo's own lines, would leave every line as far off
# as the network's small picture placed it: a few tenths of a space, which the readers' mistakes rise with.
def test_refining_moves_lines_onto_the_photos_own():
    photo, truth = bent_staff()
    truth = np.asarray(truth, np.float64)
    middle, down = (truth[0] + truth[4]) / 2, (truth[4] - truth[0]) / 4
    off = np.stack([middle + 0.3 * down + (k - 2) * 1.06 * down for k in range(5)])   # 0.3 spaces low, lines 6% apart too far
    assert error_in_spaces(off, truth) > 0.3
    assert error_in_spaces(refine(photo, off), truth) < 0.05


# A finder with a layer Core ML cannot convert could not ship (AGENTS.md); one that changes the picture's size would
# not line its marks up with the photo.
def test_the_finder_keeps_the_pictures_size_and_converts_to_core_ml():
    finder = Finder((4, 8, 8, 8, 8)).eval()
    with torch.no_grad():
        assert finder(torch.zeros(1, 1, LONG, SIDE)).shape == (1, 3, LONG, SIDE)
    description = core_ml(finder, (LONG, SIDE), "marks").get_spec().description
    assert list(description.output[0].type.multiArrayType.shape) == [1, 3, LONG, SIDE]


# Staves listed at a scale other than their picture's would teach the finder to mark places where no staff is; a staff
# the generator put outside the photo would teach it to mark paper.
def test_prepared_photos_and_their_staves_line_up(tmp_path):
    import cv2
    import json

    from sheet_model.train_finder import Photos, prepare_shard

    photo, staves = page_of_staves()
    shard = tmp_path / "00000"
    shard.mkdir()
    cv2.imwrite(str(shard / "00000003.jpg"), photo)
    systems = [{"staff": {"lines_px": lines, "in_photo": True}} for lines in staves] + [{"staff": {"lines_px": staves[0], "in_photo": False}}]
    (shard / "00000003.json").write_text(json.dumps({"systems": systems, "index": 3, "level": 1}))
    prepare_shard((shard, tmp_path / "out"))
    record = json.loads((tmp_path / "out" / "00000.jsonl").read_text())
    assert len(record["staves"]) == 3 and np.isclose(record["scale"], SIDE / 3024)
    picture, wanted = Photos([record | {"folder": str(tmp_path / "out")}], vary=False)[0]
    assert picture.shape == (1, LONG, SIDE) and wanted.shape == (3, LONG, SIDE)
    dark = picture[0].numpy() < 0.6                     # ink, in the picture as the network sees it
    on_staff = wanted[0].numpy() > 0.5
    assert dark[on_staff].mean() > 5 * dark[~on_staff].mean()
