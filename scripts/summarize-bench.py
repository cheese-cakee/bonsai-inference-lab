#!/usr/bin/env python3
"""Summarize raw llama-bench samples without hiding run-to-run spread."""

import json
import statistics
import sys
from pathlib import Path


def median_absolute_deviation(samples):
    center = statistics.median(samples)
    return statistics.median(abs(sample - center) for sample in samples)


def main():
    if len(sys.argv) != 2:
        raise SystemExit(f"Usage: {sys.argv[0]} OUTPUT_DIR")

    directory = Path(sys.argv[1])
    grouped = {}
    for path in sorted(directory.glob("[0-9][0-9]-*.json")):
        label = path.stem.split("-", 1)[1]
        for test in json.loads(path.read_text(encoding="utf-8")):
            key = (test["n_prompt"], test["n_gen"])
            samples = [float(value) for value in test["samples_ts"]]
            if not samples:
                raise ValueError(f"No throughput samples in {path}")
            grouped.setdefault((label, key), []).extend(samples)

    for key in ((256, 0), (0, 64)):
        baseline = grouped.get(("baseline", key), [])
        candidate = grouped.get(("candidate", key), [])
        if not baseline or not candidate:
            raise ValueError(f"Missing matched samples for pp{key[0]} tg{key[1]}")
        base_median = statistics.median(baseline)
        candidate_median = statistics.median(candidate)
        print(f"pp{key[0]} tg{key[1]}")
        for label, samples in (("baseline", baseline), ("candidate", candidate)):
            print(
                f"  {label}: n={len(samples)}, median={statistics.median(samples):.3f} tok/s, "
                f"MAD={median_absolute_deviation(samples):.3f}, "
                f"range={min(samples):.3f}-{max(samples):.3f}"
            )
        print(f"  candidate/baseline median ratio: {candidate_median / base_median:.3f}x")


if __name__ == "__main__":
    main()
