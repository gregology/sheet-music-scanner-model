"""Writes a reader run and a finder run as the Hugging Face model repo holds them: weights only, as safetensors, with
the settings needed to build each network and the reader's vocabulary. Optionally the Core ML programs too.

Usage: uv run python tools/publish.py <reader run> <finder run> <out folder> --datasets v0 v1 ... [--core-ml]

A run's `best.pt` is a pickle with the epoch and settings inside. Hugging Face scans a pickle and `torch.load` runs
code from one; a safetensors file holds tensors and nothing else. `config.json` records the run names, epochs and
settings, so a published version traces back to its checkpoint. `--core-ml` writes `coreml/Reader.mlpackage` and
`coreml/Finder.mlpackage` as the app bundles them, which needs coremltools (`uv sync --group dev`).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from sheet_model.lines import HEIGHT, PX_PER_SPACE, WIDTH
from sheet_model.finder import LONG, SIDE
from sheet_model.tokens import Vocabulary


def weights(run: Path, out: Path) -> dict:
    saved = torch.load(run / "best.pt", map_location="cpu")
    save_file({name: tensor.contiguous() for name, tensor in saved["model"].items()}, str(out))
    return {"run": run.name, "epoch": saved.get("epoch"), "weights": sum(t.numel() for t in saved["model"].values())} | saved["settings"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("reader", type=Path)
    parser.add_argument("finder", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--datasets", nargs="+", required=True, help="the generated datasets the reader trained on, by name: a run's settings name only the folders after its first")
    parser.add_argument("--core-ml", action="store_true")
    arguments = parser.parse_args()
    arguments.out.mkdir(parents=True, exist_ok=True)

    trained = json.loads((arguments.reader / "settings.json").read_text())
    reader = weights(arguments.reader, arguments.out / "reader.safetensors")
    reader |= {"lines": trained["lines"], "epochs": trained["epochs"], "batch": trained["batch"], "rate": trained["rate"],
               "datasets": arguments.datasets}
    finder = weights(arguments.finder, arguments.out / "finder.safetensors")
    Vocabulary.load(arguments.reader / "vocabulary.json").save(arguments.out / "vocabulary.json")
    config = {
        "format": 3,
        "tokens": trained["tokens"],
        "counts": trained["counts"],
        "line_picture": {"height": HEIGHT, "width": WIDTH, "px_per_staff_space": round(PX_PER_SPACE, 3)},
        "finder_picture": {"height": LONG, "width": SIDE},
        "reader": reader,
        "finder": finder,
    }
    (arguments.out / "config.json").write_text(json.dumps(config, indent=1) + "\n")

    if arguments.core_ml:
        from sheet_model.export import core_ml
        from sheet_model.finder import Finder
        from sheet_model.model import Reader
        from safetensors.torch import load_file

        model = Reader(reader["classes"], reader["width"], reader["layers"], reader["heads"], reader["dropout"]).eval()
        model.load_state_dict(load_file(arguments.out / "reader.safetensors"))
        (arguments.out / "coreml").mkdir(exist_ok=True)
        program = core_ml(model, half=False)   # at 16 bits it read all zeros in the iOS simulator on an Intel Mac
        program.short_description = f"Sheet Music Scanner's line reader, {reader['run']}"
        program.save(str(arguments.out / "coreml" / "Reader.mlpackage"))
        network = Finder(tuple(finder["widths"])).eval()
        network.load_state_dict(load_file(arguments.out / "finder.safetensors"))
        program = core_ml(network, (LONG, SIDE), "marks")
        program.short_description = f"Sheet Music Scanner's line finder, {finder['run']}"
        program.save(str(arguments.out / "coreml" / "Finder.mlpackage"))
    print(json.dumps(config, indent=1))


if __name__ == "__main__":
    main()
