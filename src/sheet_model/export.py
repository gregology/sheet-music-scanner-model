"""Converts a trained line reader to Core ML, and checks that the converted reader writes the same tokens.

Usage: uv run python -m sheet_model.export <run> <experiment>     converts; on Linux or macOS
       uv run python -m sheet_model.export <run> --check          compares; on macOS only

Converting writes `<run>/reader.mlpackage` and `<run>/check.npz`: line pictures from the experiment's first pages, and
the PyTorch reader's likeliest token at every step of each, with its probability. Checking runs the Core ML reader on those pictures and prints how many lines it
reads to the same tokens, and how long a line takes. Core ML runs a model only on Apple's systems, so the check needs
a Mac or an iPhone; PyTorch no longer runs on Intel Macs, so the check needs only coremltools and NumPy.
README.md, "Core ML".

The converted reader takes one line picture, 1 by 1 by HEIGHT by WIDTH, as training gives it (paper near +1, ink near
-1), and gives scores for every step and token, before the softmax, as the PyTorch reader does. Its weights are
16-bit, as the Neural Engine runs them.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

PAGES = 12   # the experiment's first pages, by name: all five conditions of sheets a and b, and two of c


def tokens(best: np.ndarray) -> list[list[int]]:
    """CTC's rule, as `model.read` applies it to each step's likeliest token: repeats dropped, then blanks."""
    lines = []
    for steps in best:
        line, last = [], 0
        for token in steps.tolist():
            if token != 0 and token != last:
                line.append(token)
            last = token
        lines.append(line)
    return lines


def core_ml(model, shape: tuple[int, int] | None = None, output: str = "scores", half: bool = True):
    """A model as a Core ML program, for iOS 18 and later: the line reader, unless `shape` gives another picture's
    height and width. `half` keeps the weights and the sums at 16 bits, which halves the file and is what the phone's
    neural engine runs; without it they stay at 32 bits."""
    import coremltools as ct
    import torch

    from .lines import HEIGHT, WIDTH

    torch.backends.mha.set_fastpath_enabled(False)   # PyTorch's fused attention is one op Core ML does not know
    example = torch.zeros(1, 1, *(shape or (HEIGHT, WIDTH)))
    with torch.no_grad():
        exported = torch.export.export(model.eval(), (example,)).run_decompositions({})
    return ct.convert(exported, inputs=[ct.TensorType(name="picture", shape=example.shape)], outputs=[ct.TensorType(name=output)],
                      convert_to="mlprogram", compute_precision=ct.precision.FLOAT16 if half else ct.precision.FLOAT32,
                      minimum_deployment_target=ct.target.iOS18)


def convert(run: Path, experiment: Path) -> None:
    import cv2
    import torch

    from .model import Reader
    from .real import page_lines

    saved = torch.load(run / "best.pt", map_location="cpu")
    s = saved["settings"]
    model = Reader(s["classes"], s["width"], s["layers"], s["heads"], s["dropout"])
    model.load_state_dict(saved["model"])
    reader = core_ml(model)
    reader.short_description = f"Sheet Scanner's line reader, from {run.name}"
    reader.save(str(run / "reader.mlpackage"))

    pictures = [p for path in sorted((experiment / "lines").glob("*.jpg"))[:PAGES]
                for p in page_lines(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE))]
    batch = torch.from_numpy(np.stack(pictures).astype(np.float32) / 127.5 - 1)[:, None]
    with torch.no_grad():
        sure, best = model(batch).softmax(-1).max(-1)
    np.savez_compressed(run / "check.npz", pictures=np.stack(pictures), best=best.numpy().astype(np.int16), sure=sure.numpy())
    print(f"{run / 'reader.mlpackage'}: converted; {len(pictures)} lines from {PAGES} pages saved to check it with")


def check(run: Path) -> None:
    import coremltools as ct

    saved = np.load(run / "check.npz")
    pictures, best, sure = saved["pictures"], saved["best"], saved["sure"]
    for units in (ct.ComputeUnit.ALL, ct.ComputeUnit.CPU_ONLY):
        reader = ct.models.MLModel(str(run / "reader.mlpackage"), compute_units=units)
        found, chance, times = [], [], []
        for picture in pictures:
            start = time.perf_counter()
            scores = reader.predict({"picture": (picture.astype(np.float32) / 127.5 - 1)[None, None]})["scores"][0].astype(np.float64)
            times.append(time.perf_counter() - start)
            probabilities = np.exp(scores - scores.max(-1, keepdims=True))
            probabilities /= probabilities.sum(-1, keepdims=True)
            found.append(probabilities.argmax(-1))
            chance.append(probabilities.max(-1))
        found, chance = np.stack(found), np.stack(chance)
        same = sum(a == b for a, b in zip(tokens(found), tokens(best)))
        print(f"{units.name}: {same} of {len(pictures)} lines read to the same tokens; {(found == best).mean():.2%} of steps "
              f"pick the same token; its probability differs by at most {np.abs(chance - sure)[found == best].max():.3f}; "
              f"a line takes {1000 * np.median(times[1:]):.0f} ms (median after the first)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run", type=Path)
    parser.add_argument("experiment", type=Path, nargs="?")
    parser.add_argument("--check", action="store_true", help="run the converted reader on the saved lines; macOS only")
    arguments = parser.parse_args()
    if arguments.check:
        check(arguments.run)
    elif arguments.experiment is None:
        sys.exit("give the experiment folder to convert, or --check")
    else:
        convert(arguments.run, arguments.experiment)


if __name__ == "__main__":
    main()
