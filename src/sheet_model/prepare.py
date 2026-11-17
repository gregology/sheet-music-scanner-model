"""Turns generated photos into line pictures the reader trains on.

Usage: uv run python -m sheet_model.prepare <generated folder> <lines folder> [--workers 12]

Each line of music whole in its photo is cut out and straightened with the truth's staff lines, nudged a little, since
a real line finder will not be exact. `<lines>/<shard>/<example>-<line>.jpg` is the picture and `<lines>/<shard>.jsonl`
lists each with its tokens and how hard its photo was. A shard already listed is skipped.
"""

from __future__ import annotations

import argparse
import json
import random
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

from .lines import straighten


def nudged(lines: list, rng: random.Random) -> list:
    """The staff lines as an imperfect line finder would give them: a little high or low, large or small, tilted, and
    short or long at the ends."""
    lines = np.asarray(lines, np.float64)
    down = (lines[4] - lines[0]) / 4                                  # one staff space at each point
    grow = 1 + rng.gauss(0, 0.03)
    shift = rng.gauss(0, 0.15) + np.linspace(-1, 1, lines.shape[1]) * rng.gauss(0, 0.12)   # in spaces; the second term tilts
    middle = (lines[0] + lines[4]) / 2 + down * shift[:, None]
    return [(middle + down * grow * (number - 2)).tolist() for number in range(5)]


def finished(shard: Path) -> bool:
    """Whether the generator is done with a shard: each of its thousand numbers is an example, or `<number>.failed`
    where the example failed one of the generator's checks."""
    return len(list(shard.glob("*.json"))) + len(list(shard.glob("*.failed"))) >= 1000


def prepare_shard(job: tuple[Path, Path]) -> tuple[str, int]:
    shard, out = job
    listing = out / f"{shard.name}.jsonl"
    if listing.exists():
        return shard.name, -1
    (out / shard.name).mkdir(parents=True, exist_ok=True)
    records = []
    for truth_path in sorted(shard.glob("*.json")):
        truth = json.loads(truth_path.read_text())
        photo = cv2.imread(str(truth_path.with_suffix(".jpg")), cv2.IMREAD_GRAYSCALE)
        if photo is None:
            continue
        camera, shot = truth["camera"], truth["shot"]
        for number, system in enumerate(truth["systems"]):
            staff = system["staff"]
            if staff["in_photo"] < 1 or any(symbol.get("cut_off") for symbol in system["symbols"]):
                continue
            rng = random.Random(f"{truth['seed']}/{truth['index']}/{number}/line")
            picture, columns = straighten(photo, nudged(staff["lines_px"], rng))
            if columns == 0:
                continue
            name = f"{shard.name}/{truth_path.stem}-{number}.jpg"
            cv2.imwrite(str(out / name), picture, [cv2.IMWRITE_JPEG_QUALITY, 92])
            records.append({"image": name, "columns": columns, "tokens": system["tokens"], "format": truth.get("format", 1), "example": truth["index"], "line": number, "level": truth["level"], "source": truth.get("source", {}).get("kind", "composer"),
                            "light": camera["light"], "electrons": camera["electrons"], "shake": camera["shake_px"], "defocus": camera["defocus_px"],
                            "tilt": shot["tilt_deg"], "shape": truth["sheet"]["shape"], "handwriting": sum(mark["system"] == number for mark in truth["handwriting"])})
    partial = listing.with_suffix(".part")
    partial.write_text("".join(json.dumps(record) + "\n" for record in records))
    partial.rename(listing)   # the listing appears last and whole: its presence marks the shard done
    return shard.name, len(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("generated", type=Path)
    parser.add_argument("lines", type=Path)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--whole-shards-only", action="store_true", help="skip a shard the generator is still filling")
    arguments = parser.parse_args()
    arguments.lines.mkdir(parents=True, exist_ok=True)
    shards = sorted(folder for folder in arguments.generated.iterdir() if folder.is_dir() and folder.name.isdigit())
    if arguments.whole_shards_only:
        shards = [shard for shard in shards if finished(shard)]
    with Pool(arguments.workers) as pool:
        for name, count in pool.imap_unordered(prepare_shard, [(shard, arguments.lines) for shard in shards]):
            print(f"{name}: {'already prepared' if count < 0 else f'{count} lines'}", flush=True)


if __name__ == "__main__":
    main()
