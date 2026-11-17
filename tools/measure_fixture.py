"""Writes the fixture the app's tests hold its measure check to: pages of real readings, and what the rule makes of them.

Usage, from the repository's folder, after `tools/read_pages.py` wrote the readings:
    uv run python tools/measure_fixture.py <out.json> reads.json [more reads.json] [--pages 12]

From each reading it takes the pages the rule changes most with no limit, and for each page writes the lines as read
(the reader's tokens and how sure it was of each) and, for the limits 0.999 and 1, the measures that did not add up,
the measures changed, and every token the rule changed as (line, place, new token). The app's Swift twin of
`sheet_model.measures` must give the same on every page. README.md, "The measure check".
"""
import argparse
import json
from pathlib import Path

from sheet_model.measures import repaired

LIMITS = (0.999, 1.0)


def case(name: str, lines: list[dict]) -> dict:
    out = {"page": name, "lines": [{"words": line["tokens"], "sure": line["sure"]} for line in lines], "limits": {}}
    for limit in LIMITS:
        mended, failed, changed = repaired(lines, limit)
        changes = [[number, place, token] for number, (line, after) in enumerate(zip(lines, mended)) for place, (old, token) in enumerate(zip(line["tokens"], after)) if old != token]
        out["limits"][str(limit)] = {"failed": failed, "changed": changed, "changes": changes}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", type=Path)
    parser.add_argument("reads", type=Path, nargs="+")
    parser.add_argument("--pages", type=int, default=12)
    arguments = parser.parse_args()
    cases = []
    for path in arguments.reads:
        found = [case(name, lines) for name, lines in sorted(json.loads(path.read_text()).items()) if lines]
        # The pages with the most to say: most tokens changed with no limit, then most changed with the limit.
        found.sort(key=lambda c: (-len(c["limits"]["1.0"]["changes"]), -len(c["limits"]["0.999"]["changes"]), c["page"]))
        cases += found[: arguments.pages]
    arguments.out.write_text(json.dumps({"cases": cases}, separators=(",", ":")))
    print(f"{len(cases)} pages; " + "; ".join(f"limit {limit}: {sum(len(c['limits'][str(limit)]['changes']) for c in cases)} tokens changed in {sum(c['limits'][str(limit)]['changed'] for c in cases)} of {sum(c['limits'][str(limit)]['failed'] for c in cases)} measures that did not add up" for limit in LIMITS))


if __name__ == "__main__":
    main()
