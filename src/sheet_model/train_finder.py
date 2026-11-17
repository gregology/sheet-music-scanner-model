"""Prepares the synthetic photos for the line finder, and trains it.

Usage: uv run python -m sheet_model.train_finder prepare <generated dataset> data/finder [--workers 12]
       uv run python -m sheet_model.train_finder train data/finder runs/<run> [--epochs 20] [--examples 30000]
           [--also <another dataset's prepared photos> ...]

Preparing shrinks each photo to the network's picture and lists its staves' lines. Training holds back one example in
fifty and, after each epoch, traces the staves in 200 of them and measures how far each found line is from the true
one, in staff spaces, and how many staves were missed or made up. `<run>/log.jsonl` has a line for each epoch,
`<run>/best.pt` the finder with the least line error so far, and a run that stops carries on from `<run>/last.pt`.
README.md, "The line finder".
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .finder import LONG, SIDE, Finder, shrink, targets, trace

HELD_BACK = 50   # one example in this many is kept for measuring
MEASURED = 200   # held-back examples traced after each epoch


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
        picture, scale = shrink(photo)
        name = f"{shard.name}/{truth_path.stem}.jpg"
        cv2.imwrite(str(out / name), picture, [cv2.IMWRITE_JPEG_QUALITY, 92])
        staves = [system["staff"]["lines_px"] for system in truth["systems"] if system["staff"].get("in_photo", True)]
        records.append({"image": name, "photo": truth_path.with_suffix(".jpg").name, "scale": scale, "staves": staves,
                        "example": truth["index"], "level": truth["level"], "source": truth.get("source", {}).get("kind", "composer")})
    listing.write_text("".join(json.dumps(r) + "\n" for r in records))
    return shard.name, len(records)


def prepare(dataset: Path, out: Path, workers: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    shards = sorted(p for p in dataset.iterdir() if p.is_dir())
    with Pool(workers) as pool:
        for name, count in pool.imap_unordered(prepare_shard, [(s, out) for s in shards]):
            print(name, "already prepared" if count < 0 else f"{count} photos", flush=True)


def load_records(folder: Path, examples: int | None) -> tuple[list[dict], list[dict]]:
    train, held = [], []
    for listing in sorted(folder.glob("*.jsonl")):
        for text in listing.read_text().splitlines():
            record = json.loads(text)
            if examples is not None and record["example"] >= examples:
                continue
            record["folder"] = str(folder)
            (held if record["example"] % HELD_BACK == 0 else train).append(record)
    return train, held


class Photos(Dataset):
    def __init__(self, records: list[dict], vary: bool):
        self.records, self.vary = records, vary

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, number: int):
        record = self.records[number]
        picture = cv2.imread(str(Path(record["folder"]) / record["image"]), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        if self.vary:   # lighter, darker, softer or noisier than the generator made it
            if random.random() < 0.3:
                picture = cv2.GaussianBlur(picture, (0, 0), random.uniform(0.3, 1.0))
            picture = (picture - 128) * random.uniform(0.7, 1.2) + 128 + random.uniform(-30, 30)
            if random.random() < 0.3:
                picture += np.random.normal(0, random.uniform(2, 8), picture.shape).astype(np.float32)
        picture = np.clip(picture, 0, 255) / 127.5 - 1
        return torch.from_numpy(picture)[None], torch.from_numpy(targets(record["staves"], record["scale"]))


def loss_of(scores: torch.Tensor, wanted: torch.Tensor) -> torch.Tensor:
    """On a staff and near its middle line: binary cross-entropy, the middle line's weighed more, since it covers few
    pixels and the tracing follows it. The staff space: the error in its log, where there is a staff."""
    on = wanted[:, 0]
    inside = F.binary_cross_entropy_with_logits(scores[:, 0], on)
    middle = F.binary_cross_entropy_with_logits(scores[:, 1], wanted[:, 1], pos_weight=torch.tensor(4.0, device=scores.device))
    space = ((scores[:, 2] - wanted[:, 2]).abs() * on).sum() / on.sum().clamp(min=1)
    return inside + middle + space


def line_errors(found: list[np.ndarray], staves: list) -> tuple[list[float], int, int]:
    """For each true staff, the largest distance down the page between its middle line and the found staff's nearest
    it, in staff spaces, over the columns both cover; how many true staves no found staff lies on; how many found
    staves lie on no true staff."""
    errors, used = [], set()
    for lines in staves:
        lines = np.asarray(lines, np.float64)
        middle = (lines[0] + lines[4]) / 2
        space = float(np.median(np.linalg.norm(lines[4] - lines[0], axis=1) / 4))
        best, best_error = None, 1.0
        for number, got in enumerate(found):
            centre = got[2]
            inner = (middle[:, 0] >= centre[0, 0]) & (middle[:, 0] <= centre[-1, 0])
            if inner.sum() < 3:
                continue
            error = float(np.abs(np.interp(middle[inner, 0], centre[:, 0], centre[:, 1]) - middle[inner, 1]).max()) / space
            if error < best_error:
                best, best_error = number, error
        if best is not None:
            errors.append(best_error)
            used.add(best)
    return errors, len(staves) - len(errors), len(found) - len(used)


def measure(model: Finder, records: list[dict], device: str) -> dict:
    model.eval()
    errors, missed, extra, total = [], 0, 0, 0
    with torch.no_grad():
        for record in records[:MEASURED]:
            picture = cv2.imread(str(Path(record["folder"]) / record["image"]), cv2.IMREAD_GRAYSCALE)
            with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
                scores = model(torch.from_numpy(picture.astype(np.float32) / 127.5 - 1)[None, None].to(device))[0].float().cpu().numpy()
            marks = np.stack([1 / (1 + np.exp(-scores[0])), 1 / (1 + np.exp(-scores[1])), np.exp(scores[2])])
            found = trace(marks, record["scale"])
            e, m, x = line_errors(found, record["staves"])
            errors += e
            missed, extra, total = missed + m, extra + x, total + len(record["staves"])
    errors = np.array(errors) if errors else np.array([1.0])
    return {"line_error_median": round(float(np.median(errors)), 4), "line_error_95th": round(float(np.percentile(errors, 95)), 4),
            "missed": round(missed / max(total, 1), 4), "made_up": round(extra / max(total, 1), 4)}


def train(arguments) -> None:
    arguments.run.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True
    train_records, held_records = load_records(arguments.photos, arguments.examples)
    for more in (load_records(folder, None) for folder in arguments.also):
        train_records, held_records = train_records + more[0], held_records + more[1]
    held_records.sort(key=lambda record: record["example"])   # so the photos measured each epoch come from every dataset
    loader = DataLoader(Photos(train_records, vary=True), arguments.batch, shuffle=True, num_workers=arguments.workers,
                        drop_last=True, persistent_workers=arguments.workers > 0)
    print(f"{len(train_records)} photos to train on, {len(held_records)} held back, on {device}", flush=True)

    model = Finder().to(device).to(memory_format=torch.channels_last)
    (arguments.run / "settings.json").write_text(json.dumps(model.settings | {"examples": arguments.examples, "batch": arguments.batch, "rate": arguments.rate,
                                                                            "epochs": arguments.epochs, "photos": len(train_records), "picture": [LONG, SIDE]}, indent=1))
    optimizer = torch.optim.AdamW(model.parameters(), arguments.rate, weight_decay=0.01)
    steps = arguments.epochs * len(loader)
    warm = min(500, steps // 10 + 1)
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(1, (step + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1, step / max(1, steps)))))
    epoch, best = 0, 1e9
    last = arguments.run / "last.pt"
    if last.exists():
        saved = torch.load(last, map_location=device)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        schedule.load_state_dict(saved["schedule"])
        epoch, best = saved["epoch"], saved["best"]
        print(f"carrying on after epoch {epoch}", flush=True)

    while epoch < arguments.epochs:
        model.train()
        start, total, count = time.time(), 0.0, 0
        for number, (pictures, wanted) in enumerate(loader):
            pictures = pictures.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            wanted = wanted.to(device, non_blocking=True)
            with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
                scores = model(pictures)
            loss = loss_of(scores.float(), wanted)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            schedule.step()
            total, count = total + loss.item(), count + 1
            if number % 200 == 199:
                print(f"epoch {epoch + 1} step {number + 1}/{len(loader)}: loss {total / count:.4f}, {count * arguments.batch / (time.time() - start):.0f} photos a second", flush=True)
        epoch += 1
        found = measure(model, held_records, device)
        entry = {"epoch": epoch, "loss": round(total / max(count, 1), 4), **found, "minutes": round((time.time() - start) / 60, 1)}
        with open(arguments.run / "log.jsonl", "a") as log:
            log.write(json.dumps(entry) + "\n")
        print(json.dumps(entry), flush=True)
        if found["line_error_median"] + found["missed"] < best:
            best = found["line_error_median"] + found["missed"]
            torch.save({"model": model.state_dict(), "settings": model.settings, "epoch": epoch}, arguments.run / "best.pt")
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "schedule": schedule.state_dict(), "epoch": epoch, "best": best}, last)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    making = commands.add_parser("prepare")
    making.add_argument("dataset", type=Path)
    making.add_argument("out", type=Path)
    making.add_argument("--workers", type=int, default=12)
    training = commands.add_parser("train")
    training.add_argument("photos", type=Path)
    training.add_argument("run", type=Path)
    training.add_argument("--epochs", type=int, default=20)
    training.add_argument("--examples", type=int, default=None, help="train on examples numbered below this")
    training.add_argument("--also", type=Path, nargs="*", default=[], help="the prepared photos of more generated datasets, to train on with the first; --examples limits only the first")
    training.add_argument("--batch", type=int, default=8)
    training.add_argument("--rate", type=float, default=1e-3)
    training.add_argument("--workers", type=int, default=8)
    arguments = parser.parse_args()
    if arguments.command == "prepare":
        prepare(arguments.dataset, arguments.out, arguments.workers)
    else:
        train(arguments)


if __name__ == "__main__":
    main()
