"""Trains the line reader on prepared line pictures.

Usage: uv run python -m sheet_model.train <lines folder> <run folder> [--epochs 30] [--examples 10000] [--batch 48]
       [--tokens pitches|positions] [--counts whole|digits] [--also <another dataset's lines folder> ...]

One line in fifty examples is held back to measure each epoch: the share of tokens the reader gets wrong on lines it
never trained on. `<run>/log.jsonl` has a line for each epoch, `<run>/best.pt` the best reader so far, and a run that
stops carries on from `<run>/last.pt` when started again. README.md, "Train".
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from rapidfuzz.distance import Levenshtein
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .lines import HEIGHT, PX_PER_SPACE, WIDTH
from .model import STEPS, Reader, read
from .tokens import Vocabulary, split_counts, tie_to_no_note, to_positions

HELD_BACK = 50   # one example in this many is kept for measuring
# How far the training variation bends lines. The test photos' lines bend, past a straight tilt, by 0.10 staff spaces at
# the median and up to 0.39, the prepared training lines by 0.05 and 0.38; their staff space changes along a line by up
# to 15%. TRAINING.md, "Bent lines".
BENT = 0.75      # the share of lines bent
BEND = 0.4       # the most a line is bent, in staff spaces, from its highest point to its lowest
GROWTH = 0.08    # the most a line's staff space grows or shrinks from one end to the other, as a share


def load_records(folders: list[Path], examples: int | None = None, tokens: str = "pitches", counts: str = "whole") -> tuple[list[dict], list[dict]]:
    """The lines to train on and the lines held back, from every shard's listing in every folder, with their tokens as
    `tokens` and `counts` ask. Each folder holds one generated dataset's lines. A line whose truth ties a note to a
    rest is left out: TRAINING.md, "Pages made like real ones"."""
    train, held = [], []
    for lines in folders:
        for listing in sorted(lines.glob("*.jsonl")):
            for text in listing.read_text().splitlines():
                record = json.loads(text)
                if examples is not None and record["example"] >= examples:
                    continue
                if tie_to_no_note(record["tokens"]):
                    continue
                record["folder"] = str(lines)
                if tokens == "positions":
                    record["tokens"] = to_positions(record["tokens"], graces=record.get("format", 1) >= 2)
                if counts == "digits":
                    record["tokens"] = split_counts(record["tokens"])
                if len(record["tokens"]) > STEPS // 2 - 4:
                    continue   # more tokens than CTC can spell in the reader's steps
                (held if record["example"] % HELD_BACK == 0 else train).append(record)
    return train, held


class Lines(Dataset):
    def __init__(self, records: list[dict], vocabulary: Vocabulary, vary: bool):
        self.records, self.vocabulary, self.vary = records, vocabulary, vary

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, number: int):
        record = self.records[number]
        picture = cv2.imread(str(Path(record["folder"]) / record["image"]), cv2.IMREAD_GRAYSCALE)
        if self.vary:
            picture = varied(picture, record["columns"])
        return torch.from_numpy(picture.astype(np.float32) / 127.5 - 1)[None], torch.tensor(self.vocabulary.encode(record["tokens"]))


def varied(picture: np.ndarray, columns: int = WIDTH) -> np.ndarray:
    """The line a little off, as a line finder leaves it, and lighter, darker or softer: so the reader does not learn
    one exact placing or exposure. It is moved, grown and tilted; most lines are bent; and the staff space grows or
    shrinks along the line. `columns` is how many columns hold the line. TRAINING.md, "Bent lines"."""
    grow, lean = random.gauss(1, 0.025), random.gauss(0, 0.004)
    move = np.float32([[grow, lean, random.gauss(0, 3)], [lean, grow, random.gauss(0, 2.5) + (1 - grow) * HEIGHT / 2]])
    paper = int(np.percentile(picture[::8, ::8], 80))
    along = np.clip(np.arange(WIDTH) / max(columns, 1), 0, 1)
    bend = curve(along, random.uniform(0, BEND) * PX_PER_SPACE) if random.random() < BENT else np.zeros(WIDTH)
    growth = 1 + random.uniform(-GROWTH, GROWTH) * (along - 0.5)
    picture = warped(picture, move, bend, growth, paper)
    if random.random() < 0.3:
        picture = cv2.GaussianBlur(picture, (0, 0), random.uniform(0.3, 1.0))
    contrast, light = random.uniform(0.75, 1.2), random.uniform(-25, 25)
    return np.clip((picture.astype(np.float32) - 128) * contrast + 128 + light, 0, 255)


def curve(along: np.ndarray, size: float, rng: random.Random = random) -> np.ndarray:
    """A smooth bend: how far down the staff moves at each column, in pixels. A few slow waves, with any straight tilt
    taken out, since the move tilts, scaled so the bend spans `size` pixels from its highest to its lowest. `along` is
    each column's place along the line, from 0 to 1."""
    waves = [(k, rng.gauss(0, 1) / k, rng.uniform(0, 2 * math.pi)) for k in (1, 2, 3)]

    def wave(t: np.ndarray) -> np.ndarray:
        return sum(height * np.cos(math.pi * k * t + phase) for k, height, phase in waves)

    grid = np.linspace(0, 1, 101)
    tilt = np.polyfit(grid, wave(grid), 1)
    rest = wave(grid) - np.polyval(tilt, grid)
    return (wave(along) - np.polyval(tilt, along)) * size / max(float(rest.max() - rest.min()), 1e-9)


def warped(picture: np.ndarray, move: np.ndarray, bend: np.ndarray, growth: np.ndarray, paper: int) -> np.ndarray:
    """`picture` moved by the affine `move`, as `cv2.warpAffine` takes it; then, at each column, the staff moved down by
    `bend` pixels and its space grown by `growth` about the middle row. One resampling, so the lines stay sharp."""
    back = cv2.invertAffineTransform(move)
    middle = HEIGHT / 2 - 0.5
    x = np.broadcast_to(np.arange(WIDTH, dtype=np.float32), (HEIGHT, WIDTH))
    y = middle + (np.arange(HEIGHT, dtype=np.float32)[:, None] - bend[None, :] - middle) / growth[None, :]
    source_x = (back[0, 0] * x + back[0, 1] * y + back[0, 2]).astype(np.float32)
    source_y = (back[1, 0] * x + back[1, 1] * y + back[1, 2]).astype(np.float32)
    return cv2.remap(picture, source_x, source_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=paper)


def batch_of(items):
    pictures, targets = zip(*items)
    return torch.stack(pictures), torch.cat(targets), torch.tensor([len(target) for target in targets])


def measure(model: Reader, loader: DataLoader, vocabulary: Vocabulary, records: list[dict], device: str) -> dict:
    """The share of tokens wrong on the held-back lines, overall and for each light level."""
    model.eval()
    wrong, total = {}, {}
    position = 0
    with torch.no_grad():
        for pictures, targets, lengths in loader:
            with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
                scores = model(pictures.to(device))
            at = 0
            for (tokens, _), size in zip(read(scores.float()), lengths.tolist()):
                truth = targets[at:at + size].tolist()
                at += size
                for group in ("all", records[position]["light"]):
                    wrong[group] = wrong.get(group, 0) + Levenshtein.distance(truth, tokens)
                    total[group] = total.get(group, 0) + size
                position += 1
    return {group: round(wrong[group] / total[group], 4) for group in sorted(total)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("lines", type=Path)
    parser.add_argument("run", type=Path)
    parser.add_argument("--also", type=Path, nargs="*", default=[], help="the prepared lines of more generated datasets, to train on with the first; --examples limits only the first")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--examples", type=int, default=None, help="train on examples numbered below this: for comparing dataset sizes")
    parser.add_argument("--batch", type=int, default=48)
    parser.add_argument("--rate", type=float, default=3e-4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--tokens", choices=["pitches", "positions"], default="pitches", help="what the reader writes for a note: its pitch, or its place on the staff (README.md, \"Tokens\")")
    parser.add_argument("--counts", choices=["whole", "digits"], default="whole", help="how a rest of several measures is written: one token for each count, or the count digit by digit (README.md, \"Tokens\")")
    arguments = parser.parse_args()
    arguments.run.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True

    train_records, held_records = load_records([arguments.lines], arguments.examples, arguments.tokens, arguments.counts)
    for more in (load_records([folder], None, arguments.tokens, arguments.counts) for folder in arguments.also):
        train_records, held_records = train_records + more[0], held_records + more[1]
    vocabulary_path = arguments.run / "vocabulary.json"
    if vocabulary_path.exists():
        vocabulary = Vocabulary.load(vocabulary_path)
    else:
        vocabulary = Vocabulary([token for record in train_records for token in record["tokens"]])
        vocabulary.save(vocabulary_path)
    train = DataLoader(Lines(train_records, vocabulary, vary=True), arguments.batch, shuffle=True, num_workers=arguments.workers, collate_fn=batch_of, drop_last=True, persistent_workers=arguments.workers > 0)
    held = DataLoader(Lines(held_records, vocabulary, vary=False), arguments.batch, num_workers=arguments.workers, collate_fn=batch_of)
    print(f"{len(train_records)} lines to train on, {len(held_records)} held back, {len(vocabulary)} tokens, on {device}", flush=True)

    model = Reader(len(vocabulary), arguments.width, arguments.layers).to(device)
    (arguments.run / "settings.json").write_text(json.dumps(model.settings | {"examples": arguments.examples, "batch": arguments.batch, "rate": arguments.rate, "epochs": arguments.epochs, "lines": len(train_records), "tokens": arguments.tokens, "counts": arguments.counts, "also": [str(folder) for folder in arguments.also]}, indent=1))
    optimizer = torch.optim.AdamW(model.parameters(), arguments.rate, weight_decay=0.01)
    steps = arguments.epochs * len(train)
    warm = min(1000, steps // 10 + 1)
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(1, (step + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1, step / max(1, steps)))))
    loss_of = nn.CTCLoss(blank=0, zero_infinity=True)
    epoch, best = 0, 1e9
    last = arguments.run / "last.pt"
    if last.exists():
        saved = torch.load(last, map_location=device)
        model.load_state_dict(saved["model"]), optimizer.load_state_dict(saved["optimizer"]), schedule.load_state_dict(saved["schedule"])
        epoch, best = saved["epoch"], saved["best"]
        print(f"carrying on after epoch {epoch}", flush=True)

    while epoch < arguments.epochs:
        model.train()
        began, running, seen = time.monotonic(), 0.0, 0
        for step, (pictures, targets, lengths) in enumerate(train):
            with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
                scores = model(pictures.to(device, non_blocking=True))
            spelled = scores.float().log_softmax(-1).transpose(0, 1)   # CTC wants steps first, and full precision
            loss = loss_of(spelled, targets.to(device), torch.full((len(lengths),), STEPS, dtype=torch.long), lengths)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            schedule.step()
            running, seen = running + loss.item(), seen + 1
            if (step + 1) % 200 == 0:
                print(f"epoch {epoch + 1} step {step + 1}/{len(train)}: loss {running / seen:.3f}, {(step + 1) * arguments.batch / (time.monotonic() - began):.0f} lines a second", flush=True)
        epoch += 1
        errors = measure(model, held, vocabulary, held_records, device)
        entry = {"epoch": epoch, "loss": round(running / max(seen, 1), 4), "tokens_wrong": errors, "minutes": round((time.monotonic() - began) / 60, 1), "rate": schedule.get_last_lr()[0]}
        with (arguments.run / "log.jsonl").open("a") as log:
            log.write(json.dumps(entry) + "\n")
        print(json.dumps(entry), flush=True)
        if errors["all"] < best:
            best = errors["all"]
            torch.save({"model": model.state_dict(), "settings": model.settings, "epoch": epoch, "tokens_wrong": errors}, arguments.run / "best.pt")
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "schedule": schedule.state_dict(), "epoch": epoch, "best": best}, last)


if __name__ == "__main__":
    main()
