"""Tries the measure rule on a reading of many pages: how many measures it mends and how many it breaks.

Usage, from the repository's folder, after `tools/read_pages.py` wrote `reads.json` for the same folder:
    uv run python tools/measure_rule.py reads.json --survey <folder> [--limits 1 0.99 0.9] [--save out.json]
    uv run python tools/measure_rule.py reads.json --dataset <folder>

The folders are as `tools/sort_mistakes.py` takes them. For each limit on how sure the reader may be of a token the
rule changes, it prints the notes wrong before and after, the measures that do not add up and the mistakes that fall in
them, the measures the rule changed, and of the changed ones how many lost all their mistakes (mended), how many had none and gained some (broken), and
how many got better or worse without either. README.md, "The measure check".
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from rapidfuzz.distance import Levenshtein

from sheet_model.measures import measures, repaired, total
from sheet_model.mistakes import read_events, tags_of_example, tags_of_musicxml


def by_measure(truth: list, lines: list[list[str]]) -> Counter:
    """The mistakes in each measure of a reading, counted where they fall in the reading."""
    where = []
    for number, (line, places, _) in enumerate(measures(lines)):
        where += [number] * len(read_events([[lines[line][place] for place in places]]))
    found = read_events(lines)
    if len(where) != len(found) or not found:
        return Counter({-1: Levenshtein.distance(truth, found)})
    return Counter(where[min(other, len(where) - 1)] for _, _, other in Levenshtein.editops(truth, found))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("reads", type=Path)
    parser.add_argument("--survey", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--limits", type=float, nargs="+", default=[1.0, 0.999, 0.99, 0.95, 0.9])
    parser.add_argument("--save", type=Path)
    arguments = parser.parse_args()
    reads = json.loads(arguments.reads.read_text())
    truths = {}
    for name in sorted(reads):
        try:
            if arguments.survey:
                truths[name] = tags_of_musicxml(arguments.survey / f"{name}.musicxml", set())[0]
            else:
                truths[name] = tags_of_example(json.loads((arguments.dataset / f"{name}.json").read_text()))[0]
        except Exception:   # a true score that cannot be read is left out, as the sorter leaves it out
            continue
    truths = {name: truth for name, truth in truths.items() if truth}
    notes = sum(len(truth) for truth in truths.values())
    print(f"{len(truths)} pages, {notes} notes\n")
    print("| Limit | Wrong before | Wrong after | Measures | Not adding up | Mistakes in them | Changed | Mended | Better | Same | Worse | Broken | Pages better | Pages worse |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    saved = {}
    for limit in arguments.limits:
        tally, rows = Counter(), []
        for name, truth in truths.items():
            lines = reads[name]
            before = [line["tokens"] for line in lines]
            after, failed, _ = repaired(lines, limit)
            was, now = by_measure(truth, before), by_measure(truth, after)
            tally["measures"] += len(measures(before))
            tally["failed"] += failed
            tally["before"] += sum(was.values())
            tally["after"] += sum(now.values())
            tally["pages better"] += sum(now.values()) < sum(was.values())
            tally["pages worse"] += sum(now.values()) > sum(was.values())
            for number, (line, places, time) in enumerate(measures(before)):
                sum_ = total(before[line], places) if time is not None else None
                if sum_ is not None and sum_ != time:
                    tally["inside"] += was[number]
                if all(before[line][place] == after[line][place] for place in places):
                    continue
                tally["changed"] += 1
                a, b = was[number], now[number]
                tally["mended" if a and not b else "broken" if b and not a else "better" if b < a else "worse" if b > a else "same"] += 1
            rows.append({"page": name, "notes": len(truth), "before": sum(was.values()), "after": sum(now.values())})
        print(f"| {limit:g} | {tally['before']} ({100 * tally['before'] / notes:.2f}%) | {tally['after']} ({100 * tally['after'] / notes:.2f}%) | {tally['measures']} | {tally['failed']} | {tally['inside']} | {tally['changed']} | "
              f"{tally['mended']} | {tally['better']} | {tally['same']} | {tally['worse']} | {tally['broken']} | {tally['pages better']} | {tally['pages worse']} |")
        saved[f"{limit:g}"] = {"tally": dict(tally), "pages": rows}
    if arguments.save:
        arguments.save.write_text(json.dumps(saved))


if __name__ == "__main__":
    main()
