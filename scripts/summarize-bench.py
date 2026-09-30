#!/usr/bin/env python3
"""Summarize ABBA llama-bench runs without hiding run-to-run spread.

Reads NN-<variant>.json files written by bench-avx2-wsl.sh, writes every
repetition to samples.csv, and prints pooled medians plus one paired ratio per
bracket (runs 1-2, 3-4, ...), so drift inside the session stays visible.
"""

import csv
import json
import statistics
import sys
from pathlib import Path


def median_absolute_deviation(samples):
    center = statistics.median(samples)
    return statistics.median(abs(sample - center) for sample in samples)


def test_name(test):
    return f"pp{test['n_prompt']}" if test["n_prompt"] else f"tg{test['n_gen']}"


def read_gpu_state(path):
    if not path.exists():
        return "", ""
    fields = [field.strip() for field in path.read_text(encoding="utf-8").split(",")]
    return fields[0], fields[1]  # temperature C, graphics clock MHz


def load_samples(directory):
    rows = []
    for path in sorted(directory.glob("[0-9][0-9]-*.json")):
        run_text, variant = path.stem.split("-", 1)
        temperature, clock = read_gpu_state(directory / f"{path.stem}-gpu-before.csv")
        for test in json.loads(path.read_text(encoding="utf-8")):
            for repetition, rate in enumerate(test["samples_ts"], start=1):
                rows.append({
                    "run": int(run_text), "variant": variant, "test": test_name(test),
                    "repetition": repetition, "tokens_per_second": float(rate),
                    "gpu_temp_c_before": temperature, "gpu_clock_mhz_before": clock,
                })
    if not rows:
        raise SystemExit(f"No NN-<variant>.json files in {directory}")
    return rows


def write_csv(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows, test):
    rates = {"baseline": [], "candidate": []}
    per_run = {}
    for row in rows:
        if row["test"] == test:
            rates[row["variant"]].append(row["tokens_per_second"])
            per_run.setdefault(row["run"], (row["variant"], []))[1].append(row["tokens_per_second"])

    print(test)
    for variant, samples in rates.items():
        print(f"  {variant:9s}: n={len(samples)}, median={statistics.median(samples):.2f} tok/s, "
              f"MAD={median_absolute_deviation(samples):.2f}, range={min(samples):.2f}-{max(samples):.2f}")
    print(f"  pooled candidate/baseline median ratio: "
          f"{statistics.median(rates['candidate']) / statistics.median(rates['baseline']):.3f}x")

    ratios = []
    for first in range(1, max(per_run) + 1, 2):
        pair = dict(per_run[run] for run in (first, first + 1) if run in per_run)
        if set(pair) == {"baseline", "candidate"}:
            ratios.append(statistics.median(pair["candidate"]) / statistics.median(pair["baseline"]))
    wins = sum(ratio > 1 for ratio in ratios)
    print(f"  per-bracket ratios: {' '.join(f'{r:.3f}' for r in ratios)} "
          f"(candidate faster in {wins}/{len(ratios)})")


def main():
    if len(sys.argv) != 2:
        raise SystemExit(f"Usage: {sys.argv[0]} OUTPUT_DIR")
    directory = Path(sys.argv[1])
    rows = load_samples(directory)
    write_csv(rows, directory / "samples.csv")
    for test in sorted({row["test"] for row in rows}):
        summarize(rows, test)


if __name__ == "__main__":
    main()
