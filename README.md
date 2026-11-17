# Sheet Music Scanner model

Sheet music scanning model I'm building for my musician wife. The model is designed to run on device in an iOS app for her iPad. I'm not a musician so any pointers would be appreciated.

This is the code: the two networks, the geometry between them, training, Core ML export, and the tools I used to work out what to fix next. The trained weights, the results and how each run went are on Hugging Face:

- **The model:** [gregology/sheet-music-scanner](https://huggingface.co/gregology/sheet-music-scanner)
- **How each run went:** [TRAINING.md](TRAINING.md)
- **The test set:** [gregology/sheet-music-scanner-eval](https://huggingface.co/datasets/gregology/sheet-music-scanner-eval)
- **The training data generator:** [sheet-music-scanner-synthetic](https://github.com/gregology/sheet-music-scanner-synthetic)

Version 1.0 gets 0.94% of notes wrong on my 81 iPhone photos of real music, and 0.37% after a check that each measure adds up. It never trained on a real photo.

## Use it

```sh
uv sync                      # Python 3.12; or pip install -e .
```

```python
from sheet_model.hub import load

scanner = load()             # the weights from Hugging Face, at revision v1.0
page = scanner.read("page.jpg")
for line in page.lines:
    print(" ".join(line.pitches))
```

`load("path/to/folder")` takes a folder laid out as the Hugging Face repo is. [`tools/publish.py`](tools/publish.py) writes one from two training runs.

## Where it runs

On an iPhone, in Core ML, with no server. That decides most of the design:

- **Fixed shapes, one pass, no loop.** A decoder that writes one token at a time needs a cache Core ML handles badly. A reader that writes every token in one pass with CTC has none of that.
- **One model file for every device.** The Neural Engine takes it on new phones, the GPU or CPU on old ones.
- **Small.** 7.8 million weights for both networks, 27 MB as the app ships them.
- **Weights the App Store can take.** Trained only on data whose licence allows it, and nothing from homr, which is AGPL.

## The parts

| Part | In | Out | Code |
|---|---|---|---|
| The line finder | The photo, shrunk to 768 × 1024 | Each staff's five lines, as points in the photo | [`finder.py`](src/sheet_model/finder.py) |
| The line picture | The photo and one staff's lines | The staff straightened, 1600 × 160 | [`lines.py`](src/sheet_model/lines.py) |
| The reader | One line picture | A token and its probability at each of 200 steps | [`model.py`](src/sheet_model/model.py) |
| Tokens | The reader's tokens | Pitches, by the rules of notation | [`tokens.py`](src/sheet_model/tokens.py) |
| The measure check | A page's tokens and probabilities | Triplet marks mended where a measure doesn't add up | [`measures.py`](src/sheet_model/measures.py) |

### The line finder

A U-Net with 1.56 million weights, four halvings deep, marks every pixel of the shrunk photo: on a staff or not, how near the staff's middle line, and the log of the staff space. Geometry follows each middle line's peaks across the picture, joins them into staves, keeps those at least twelve staff spaces long, and sets five lines around each at the marked spacing, 17 points a line. Then each line is refined in the full photo: straightened, followed in slices about four staff spaces wide, and each point moved to where the ink is. That takes lines from 0.42 staff spaces off to 0.02 in the tests.

### The line picture

`straighten` cuts one staff out of the photo along its own lines, bends and all. Every line comes out 1600 × 160 px with the staff in the middle, five staff spaces of room above and below, at 11.4 px to a staff space. A tilted, bent or distant staff comes out the same as a flat one.

### The reader

Five convolution blocks take the picture to 200 columns of 256 features, and a last convolution folds what's left of the height into each column. Four pre-norm transformer encoder layers (4 heads, GELU, learned positions) let each column see the whole line. A linear layer scores every token at every column, and CTC's rule reads them off: the likeliest token at each step, repeats dropped, then blanks. Each token's confidence is the highest probability it got over the steps that spelled it.

### Tokens

The reader writes where a note sits, not its pitch: `note-S3_4` is a quarter note in the third space, `note-L0_8` an eighth on the first ledger line below, `acc-#` a sharp before the next note. Clef, key, time, rests, barlines, ties, chords, tuplets and tremolos are tokens of their own. A rest of several measures is `multirest` then its count digit by digit, so a count the reader never met is still readable. `to_pitches` applies the clef, the key and each measure's accidentals.

Checked on 48,419 training lines, 48,385 come back from positions to the generator's pitches exactly. The other 34 are grace notes, where the generator and a musician disagree about whether an accidental carries.

The vocabulary grows with the data, since a note's token names its place and length together. Version 1.0 has 1,330 tokens.

### The measure check

A triplet printed with no 3 looks like three plain eighths. `repaired` sums each measure against the time signature last read on the page (a chord counts once, and so do a tremolo's two notes), and where the sum is off it searches for the change of triplet marks that makes it right, cheapest first by the reader's confidence. It changes nothing when no change fits, when two fit equally, or when the measure holds a tremolo. A `limit` below 1 forbids changing a token the reader was surer of than that. The app runs a Swift copy with no limit; [`tools/measure_fixture.py`](tools/measure_fixture.py) writes the pages its tests hold it to.

## Train

The data comes from [the generator](https://github.com/gregology/sheet-music-scanner-synthetic). Each example is a photo and a JSON truth with every staff line's path and every line's tokens.

```sh
uv run python -m sheet_model.prepare <generated dataset> data/lines-v0 --workers 12
uv run python -m sheet_model.train data/lines-v0 runs/reader --also data/lines-v1 data/lines-v2 \
    --tokens positions --counts digits --epochs 30
```

`prepare` cuts every line out of every photo along the truth's staff lines, nudged a little since a real finder won't be exact. About five lines and 150 kB a page.

`train` holds back one example in fifty and, after each epoch, measures the share of tokens wrong on those lines; `best.pt` is the best epoch by that measure. It varies each line as it's loaded: moved, grown, tilted, three in four bent, the staff space stretched along the line, blurred, lightened or darkened. AdamW at 3e-4 with weight decay 0.01, 1,000 steps of warm-up then cosine, bf16 autocast, batch 48. A line longer than 96 tokens is left out, since CTC can't spell more in 200 steps. A run that stops carries on from `last.pt`. Version 1.0's reader took 30 epochs of 7.6 minutes on an RTX 4090.

The finder trains on the photos' staff lines:

```sh
uv run python -m sheet_model.train_finder prepare <generated dataset> data/finder
uv run python -m sheet_model.train_finder train data/finder runs/finder --epochs 20
```

## Three measures

A model that reads its own synthetic data well tells you little. I score every reader on three sets, which are in [the test set](https://huggingface.co/datasets/gregology/sheet-music-scanner-eval) with an `evaluate.py` for the published weights:

- **My photos:** 114 iPhone photos of ten real string parts and eight made-up sheets, flat, bent, shadowed, angled and on a stand. The test that counts.
- **The survey:** 434 clean pictures of real string-quartet parts, 70,278 notes. What real notation gets misread.
- **t1:** 2,000 synthetic photos from a seed no training set uses, with their true staff lines, so the finder's mistakes can be told from the reader's.

The tools that got me from one run to the next:

```sh
uv run python tools/read_pages.py runs/reader runs/finder reads.json <folder>     # read once, on a GPU
uv run python tools/sort_mistakes.py reads.json --survey <folder> --run runs/reader
uv run python tools/sort_mistakes.py reads.json --dataset <folder>
uv run python tools/measure_rule.py reads.json --survey <folder>
```

[`sort_mistakes.py`](tools/sort_mistakes.py) tags every true note with what's drawn there (a triplet with no mark, a tie over a barline, a note four ledger lines up) and ranks the tags by how well they separate wrong notes from right ones. Its gap list ranks again and again, each time without the notes of the tag before, so each mistake is counted once. The top rows of that list are what the generator was asked for next.

## Core ML

```sh
uv run python tools/publish.py runs/reader runs/finder <out> --datasets v0 v1 v2 v3-tremolo --core-ml
```

writes the weights as safetensors, `config.json`, the vocabulary, and `coreml/Reader.mlpackage` and `coreml/Finder.mlpackage` for iOS 18. The conversion is `torch.export` then coremltools, with PyTorch's fused attention turned off so every op is one Core ML knows. The reader goes out at 32 bits, because at 16 it read all zeros in the iOS simulator on an Intel Mac; the finder is fine at 16. [`export.py`](src/sheet_model/export.py) has a check that runs the converted reader on a Mac and compares its tokens with PyTorch's.

## Tests

```sh
uv run python -m pytest
```

Each test says which mistake it catches.

## Licence

MIT. See [LICENSE](LICENSE).
