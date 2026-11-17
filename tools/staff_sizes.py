"""How small the staffs of each picture are to the line finder, and how the pictures with a small staff are read.

Usage, from the repository's folder:
    uv run python tools/staff_sizes.py sizes.json --find runs/f30k <folder>
    uv run python tools/staff_sizes.py sizes.json --scored sorted.json [more.json] [--limit 3]

`--find` asks the finder for every staff of every `.png` and `.jpg` in the folder and the folders inside it, and
saves, for each picture, each staff's distance from line to line in the finder's own picture (the photo shrunk to
768 by 1,024 pixels): the middle distance along the staff and the least, before and after the staff is refined
against the photo. `--scored` takes pages scored by `tools/sort_mistakes.py --save` or `tools/measure_rule.py
--save` and prints how the pictures whose smallest staff is under `--limit` pixels were read, and the rest. The
measure is the app's: the smallest staff's middle distance, after refining. README.md, "Three measures".
"""
import argparse
import json
from pathlib import Path


def spaces(staff, scale: float) -> list[float]:
    import numpy as np
    gaps = np.linalg.norm(np.array(staff)[4] - np.array(staff)[0], axis=1) / 4 * scale
    return [round(float(np.median(gaps)), 3), round(float(gaps.min()), 3)]


def find_sizes(run: Path, folder: Path) -> dict:
    import cv2
    import numpy as np
    import torch

    from sheet_model.finder import Finder, refine, shrink, trace

    device = "cuda" if torch.cuda.is_available() else "cpu"
    saved = torch.load(run / "best.pt", map_location=device)
    finder = Finder(tuple(saved["settings"]["widths"])).to(device).eval()
    finder.load_state_dict(saved["model"])
    paths = sorted((p for p in folder.rglob("*") if p.suffix in (".png", ".jpg") and not any(part.startswith(".") for part in p.relative_to(folder).parts)), key=lambda p: str(p.relative_to(folder)))
    out = {}
    for path in paths:
        photo = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        picture, scale = shrink(photo)
        with torch.no_grad():
            scores = finder(torch.from_numpy(picture.astype(np.float32) / 127.5 - 1)[None, None].to(device))[0].float().cpu().numpy()
        traced = trace(np.stack([1 / (1 + np.exp(-scores[0])), 1 / (1 + np.exp(-scores[1])), np.exp(scores[2])]), scale)
        out[str(path.relative_to(folder).with_suffix(""))] = {"traced": [spaces(staff, scale) for staff in traced], "refined": [spaces(refine(photo, staff), scale) for staff in traced]}
    return out


def smallest(sizes: dict) -> float | None:
    """The smallest staff's middle distance from line to line, after refining: what the app warns on."""
    return min((staff[0] for staff in sizes["refined"]), default=None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sizes", type=Path)
    parser.add_argument("--find", nargs=2, metavar=("RUN", "FOLDER"), type=Path)
    parser.add_argument("--scored", type=Path, nargs="+")
    parser.add_argument("--limit", type=float, default=3.0)
    arguments = parser.parse_args()
    if arguments.find:
        arguments.sizes.write_text(json.dumps(find_sizes(*arguments.find)))
    sizes = json.loads(arguments.sizes.read_text())
    print(f"{len(sizes)} pictures; {sum(1 for found in sizes.values() if (smallest(found) or 9e9) < arguments.limit)} with a staff under {arguments.limit} pixels from line to line")
    for path in arguments.scored or []:
        saved = json.loads(path.read_text())
        pages = saved["pages"] if "pages" in saved else next(iter(saved.values()))["pages"]
        small, rest = [0, 0, 0, 0], [0, 0, 0, 0]   # pictures, notes, wrong, pictures with no note wrong
        for page in pages:
            if page["page"] not in sizes:
                continue   # a clean picture beside the photos has no size here
            wrong = page["wrong"] if "wrong" in page else page["before"]
            into = small if (smallest(sizes[page["page"]]) or 9e9) < arguments.limit else rest
            for place, more in enumerate((1, page["notes"], wrong, int(wrong == 0))):
                into[place] += more
        print(f"{path.name}: " + "; ".join(f"{name}: {count} pictures, {wrong} of {notes} notes wrong ({100 * wrong / max(notes, 1):.2f}%), {clean} with no note wrong" for name, (count, notes, wrong, clean) in (("small staff", small), ("the rest", rest))))


if __name__ == "__main__":
    main()
