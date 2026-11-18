# How it was trained

Notes from 11 to 17 November 2026, run by run. Each round went the same way: read a test set, sort the mistakes by what's drawn on the page, change the data or the model for the worst of them, train again from scratch, and score on the same sets. Every number names its reader, its line finder, its test set and the date. All training ran on one RTX 4090.

![Notes wrong by reader on three test sets](https://huggingface.co/gregology/sheet-music-scanner/resolve/main/images/readers.png)

| Run | Date | Trained on | What changed | Real-music photos | Survey | t1 |
|---|---|---|---|---|---|---|
| r40k-pos | 11-15 | v0, v1 | Pages made like real ones | 2.51% | 6.16% | 2.99% |
| r52k-real | 11-16 | + v2 | Real scores as the music | 1.59% | 3.79% | 2.86% |
| r56k-gaps | 11-17 | + v3, both halves | Pages asked for by name | 2.19% | 3.48% | 3.34% |
| r54k-tremolo | 11-17 | + v3-tremolo only | The triplet pages dropped | 0.94% | 3.26% | 2.92% |
| r51k-clean | 11-17 | The same, no flagged pages | PDMX's licence-flagged pages left out | 1.13% | 3.32% | 2.88% |

Share of all notes wrong, every reader through the finder f30k, before the measure check. Real-music photos: my 81 photos, 13,127 notes. Survey: 434 pages, 70,278 notes. t1: 2,000 synthetic photos, 398,148 notes. The run names count the thousands of training photos.

## The first readers

The first reader, r10k, trained for an hour on 10,000 of dataset v0's photos (47,490 lines) and got 0.25% of tokens wrong on the synthetic lines it hadn't seen. On my first 41 test images, photos and clean pictures of eight made-up sheets read along homr's straightened lines, it got 2.4% of notes wrong on average. homr gets under 1% there.

The mistakes that mattered were big ones, and they came from writing pitches. A pitch token makes the reader carry the clef along the line, and one slip moves every note after it: on three bent pages r10k read a whole stretch exactly an octave low, as if under the treble clef that sounds an octave down. On another bent photo it dropped sharps that came from the key signature.

So I tried tokens for where each note sits on the staff, with code applying the clef and key afterwards. r10k-pos, the same data and settings, read every condition better than r10k, all five pages of the clef-changing sheet at 0.6% wrong or less where r10k had 4.4% to 6.6%.

Then I grew the data: 20,000 and 30,000 photos, positions both times. r20k-pos read the flat photos 37.6% wrong. Its notes' places were right; it was opening 52 of 318 treble lines with the octave-down treble clef, which differs only by a small 8 under it, and every note on those lines followed. A pitch reader learns the clef from every note on the line, a position reader from one token a line, and that seems to make it fragile (an inference I haven't tested).

**How much two runs differ.** I trained r10k twice with only the random start changed. The two differed by 0.5 points or less under each photo condition, and 1.2 on bent photos. The position readers, run on more data, ranged from 0.5% to 37.6% on the same flat photos. I haven't repeated a run since, so the later tables have no error bars.

## Bent lines

The photos' lines bend more than the training lines did. Following each staff in slices of about four staff spaces, a test line strays 0.10 staff spaces at the median from a straight tilt and up to 0.35; a training line strayed 0.05. Mistakes rose with bend: 1.4% of notes wrong on lines bent under 0.1, 6.4% on lines bent 0.2 and over (r10k).

So training now bends three lines in four by a smooth curve of up to 0.4 staff spaces, a few slow waves with the tilt taken out, and stretches the staff space up to 8% from one end to the other, as the shadowed photos' lines do.

It didn't help. Two runs with bent lines, r10k-bend and r10k-bend-2, read the bent photos no better than the spread between two plain runs: 7.3% and 6.5% against 6.3% and 7.5%. I kept the bending because it costs nothing and the next problem turned out to be the real one.

## A page's clef

The bent photos' biggest mistakes were a misread clef at the start of the first line. A printed page starts each line with the clef in force, so a line's clef can be read from the lines around it as well as from its own start. I tried a rule that reads each line under the clef the line before ended with.

For five of seven readers it recovered all of the bent photos' clef mistakes: r30k-pos went from 7.3% to 1.9% wrong on bent photos, r20k-pos from 29.3% to 0.6% on the clef-changing sheet. It failed on a pitch reader, whose notes didn't follow its own clef token, so moving the clef moved right notes.

The bigger fix came next, from the line finder: most of these misreads came from homr's straightened lines, which dip at the start where the clef is.

## The line finder

Every score so far used homr's line finder, which can't ship: homr is AGPL. f30k is mine: a U-Net with 1.56 million weights that marks staff pixels, middle lines and line spacing on the photo shrunk to 768 by 1024, then geometry to trace the staves and refine each line against the full photo. It trained for 20 epochs on v0's 29,397 photos, 139 minutes. On held-back photos its middle lines are 0.035 staff spaces from the truth at the median before refining. It missed no staff and made up 0.3%.

Every reader read the test images better on f30k's lines than on homr's: r20k-pos from 20.2% wrong to 1.2%, r30k-pos from 3.7% to 1.3%. The position readers' clef trouble was mostly homr's lines. On two angled photos f30k found a ninth staff in the grain of the wooden floor, which looks a lot like staff lines.

## Real music

Then I photographed ten pages of real music: page one of string parts from OpenScore's quartets, as MuseScore draws them, each under six conditions. Every reader read them about nine times worse than my made-up sheets: r30k-pos at 11.9% of notes wrong against 1.3%.

The photos weren't the cause. Clean pictures of the same pages scored the same as the photos, 11.9%. What the readers missed was in the music: 87 triplets read as plain notes (69 on one viola page marked with a small 3 and no bracket), 37 extra measures from misreading the bold 2, 3 or 4 over a long rest as 5, 6, 8 or 12, 20 notes dropped from double stops, 8 grace notes read as sixteenths. The generator had never drawn most of those the way MuseScore does.

It was also sure of what it got wrong: 0.988 mean confidence on the page it read 49% wrong.

## Pages made like real ones

So the generator learned to draw what was missing: chords, triplets marked three ways, grace notes in groups, rests of 2 to 99 measures drawn as MuseScore draws them, a part's page head, boards, a music stand and a stack of sheets under the page. Dataset v1, 10,000 photos.

r40k-pos, on v0 and v1, read the real music at 2.6% wrong (mean of the pages), from 11.9%. On the clean pictures no long rest's count was wrong and no double stop lost a note, and 3 of the viola page's 72 marked triplets were read as plain notes, from 69. Most of what was left was triplets and plain eighths taken for each other, and only the length of the measure tells those apart.

## Three measures

Ten pages can't say which notation a reader misreads, so I built two larger tests.

**The survey** is page one of every part of OpenScore's string quartets that MuseScore would draw: 434 pages, 70,278 notes, clean pictures. r40k-pos got 6.16% wrong. Sorting every mistake by what's drawn at the note, and counting each mistake once for the sign that explains it best:

| What's drawn | Notes | Wrong | Did the generator draw it? |
|---|---|---|---|
| Tremolo between two notes | 922 | 918 | No |
| Triplet printed with no 3 | 1,078 | 959 | One piece in five |
| 32nd notes | 985 | 354 | Yes, in two shapes |
| A length with no token (64ths, double dots, odd tuplets) | 103 | 104 | No |
| Double sharps and flats | 44 | 44 | No |

Signs the generator never drew made 30% of the mistakes. The rest were signs it did draw, in company it hadn't seen: 32nd notes come in runs of eight on real pages and in two fixed shapes in the composer's music.

I drew the survey again with Verovio, the training pages' engraver, to see whether MuseScore's drawing was the problem. It wasn't: 7.30% wrong on Verovio's pages against 6.17% on MuseScore's. The gap was in the music, which is what pointed at real scores.

**t1** is 2,000 synthetic photos from a seed no training set uses. Its true staff lines are known, so each photo can be read twice, along the finder's lines and along the true ones. r40k-pos got 2.99% wrong on the finder's lines and 1.34% on the true lines: on photos, the finder cost more than the reader.

## Flattening the page

The finder's mistakes are on steep photos. By the camera's tilt from straight above the page (r40k-pos, t1):

| Tilt | Photos | Wrong, the finder's lines | Wrong, the true lines |
|---|---|---|---|
| Under 10° | 1,129 | 1.46% | 1.44% |
| 10° to 20° | 534 | 1.96% | 1.00% |
| 20° to 30° | 183 | 3.45% | 0.95% |
| 30° to 40° | 96 | 12.21% | 1.61% |
| Over 40° | 58 | 25.51% | 3.05% |

Far staves on a steep photo end up under 3 pixels between lines in the finder's picture, and the traced lines wander. Warping the page square from its true corners first halved the steep photos' mistakes and didn't reach the true lines, since the finder never saw a stretched page. A bigger finder picture without retraining was far worse (22.69% wrong): it only knows staves at one size. So the app warns when the smallest staff comes out under 3 pixels and asks for another photo. Retraining the finder is still open.

## The measure check

For the triplets, I tried a rule before more training. It sums each measure against the time signature and, where it's off, looks for the change of triplet marks that fixes it, cheapest first by the reader's confidence. A limit can forbid changing tokens the reader was surer of than some threshold.

On the survey, with r40k-pos, the rule with no limit took 1,164 of 4,333 mistakes away and changed 295 measures: 279 for the better, 15 for the worse. On t1 it did harm, 11,898 mistakes to 12,064, because a synthetic photo's measure is more often off for another reason, like a note lost off the page's edge. A limit of 0.999 was a gain on every set, so the app shipped with that.

r52k-real was surer of its misread triplets, and at 0.999 the check left most of them alone: my real-music photos at 0.98% where no limit gave 0.35%. The app runs with no limit from version 0.9.

## Real scores as the music

The generator learned to cut pages from real scores: PDMX and the OpenScore Lieder songs, with every line of the test quartets fingerprinted and any score sharing a long run with them left out (359 of PDMX's scores). Dataset v2 is 11,998 photos, three in four cut from real scores.

r52k-real, on v0, v1 and v2, beat r40k-pos on every set: the survey 3.79% from 6.16%, my real-music photos 1.59% from 2.51%. 32nd notes went from 366 wrong to 119, chords of two from 401 to 139.

Two things were left. Unmarked triplets, 1,018 wrong of 1,078, which the line can't show. And tremolos, 764 of 922: only 40 of v2's 74,782 lines held one.

## Tremolos and unmarked tuplets

The generator can be asked for pages that hold a named sign. Dataset v3 is two halves of 2,000 pages: one with a tremolo between two notes on every page, all cut from real scores, and one with an unmarked triplet on every page.

r56k-gaps, on all of it, read tremolos far better (764 wrong to 477) and unmarked two-in-one tuplets too (558 to 273). It also read my real-music photos worse, 2.19% from 1.59%, and t1 worse, 3.34% from 2.86%.

### Unmarked triplets: the run I threw away

The triplet pages taught it to guess. An unmarked triplet looks on its line exactly like three plain eighths, so 2,000 pages of them taught the reader that three beamed eighths might be a triplet. On the survey, plain notes read as triplets went from 96 to 342. On my photos, one cello page went from 1 mistake to 68 and a violin page from 22 to 61, every one a plain eighth read as a triplet. And unmarked triplets themselves barely moved: 942 wrong of 1,078, from 1,018.

r54k-tremolo is the same run without the triplet pages. It kept the tremolo gains (498 wrong of 922) and stopped guessing (56 plain notes read as triplets). Unmarked triplets are back where they were, 1,035 of 1,078, and are left to the measure check, which only has to decide whether a measure adds up. With the check, r54k-tremolo reads my real-music photos at 0.37% and the survey at 2.10%. It's the reader in the app from version 0.10.

## Without the flagged scores

PDMX's makers flagged scores whose licence on musescore.com disagrees with the one in the file. My generator leaves them out by default; v2 and v3-tremolo were made with them in. 2,797 of those two datasets' 13,998 pages came from flagged scores: 2,222 of v2's and 575 of v3-tremolo's, or 16,451 of 282,716 lines (5.8%).

r51k-clean is r54k-tremolo's data and settings without those pages: 51,198 photos, 30 epochs, trained on 2026-11-17. Both readers were scored the same evening from their published files, with the test set's `evaluate.py`, on the RTX 4090, through the finder f30k:

| Test set | r54k-tremolo | r51k-clean |
|---|---|---|
| My photos of real music, 81 | 0.92% | 1.10% |
| The same, after the measure check | 0.36% | 0.21% |
| My photos of made-up sheets, 33 | 0.88% | 0.53% |
| The survey, 434 pages | 3.26% | 3.32% |
| The same, after the measure check | 2.12% | 2.17% |
| t1, 2,000 photos | 2.97% | 2.86% |
| t1, well taken, 273 | 0.23% | 0.20% |
| Survey tremolos read wrong, of 922, from `read_pages.py`'s reading | 498 | 469 |

Each reader wins some rows, and every gap is within what two runs of one setup have differed by before. The flagged pages were worth nothing I can measure, not even to tremolos, though they were 29% of the tremolo pages. So version 1.0 here is r51k-clean.

## Core ML

The reader converts as it is: no loop, fixed shapes, `torch.export` then coremltools to an iOS 18 program. On an Intel Mac, 119 lines from my first 12 pages read to the same tokens as PyTorch, with probabilities within 0.044. At 16 bits the reader read all zeros in the iOS simulator on that Mac, so the app ships it at 32 bits, 24 MB. The finder is fine at 16 bits, 3 MB. The app's Swift reading of a synthetic page matches PyTorch's tokens on all 7 lines and puts every staff point within a tenth of a staff space.

## Not measured

- Time on an iPhone. On the Intel Mac's CPU a line took 198 ms, but that Mac has no Neural Engine.
- How much two runs of the same setup differ, since the first readers.
- Whether the reader's confidence can flag a bad page. So far it's sure of most of what it gets wrong.
