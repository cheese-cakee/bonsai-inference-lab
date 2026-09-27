# Bonsai 2 on a 6 GB RTX 4050

An inference-engineering case study for [PrismML's Ternary Bonsai 2 27B](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf), using its 5.53 GiB PTQ1_0 GGUF and PrismML's [llama.cpp fork](https://github.com/PrismML-Eng/llama.cpp). This is a focused runtime patch and reproducible experiment, **not** a new model, quantizer, or standalone inference engine.

On an RTX 4050 Laptop GPU (6141 MiB reported VRAM), the model nearly fills device memory. We kept all 65 transformer layers on the GPU, moved only `output.weight` to the CPU, and added an AVX2 fast path for the CPU's PTQ1_0 × Q8_0 dot product. The patch is in [`patches/ptq1-avx2.patch`](patches/ptq1-avx2.patch); it applies to pinned Prism fork commit `9a9394a895b96003ca842a6041cb28ac49a108f7`.

## Results and scope

| Comparison on the target laptop | Baseline | Candidate | What it means |
| --- | ---: | ---: | --- |
| `output.weight` GPU → CPU, GPU model buffer | 5395.33 MiB | 5130.10 MiB | 265.23 MiB more headroom; same model weights |
| GPU-head → CPU-head, `pp256` | 36.63 tok/s | 231.96 tok/s | Matched placement experiment; likely escaped a memory-capacity cliff |
| Generic CPU head → AVX2 CPU head, `tg64` | 9.85 ± 0.05 tok/s | 15.44 ± 1.07 tok/s | Matched 3-repeat llama-bench comparison; both keep the head on CPU |
| Generic CPU head → AVX2 CPU head, `pp512+tg128` | 41.65 ± 1.31 tok/s | 57.80 ± 0.28 tok/s | Combined-workload throughput, not per-token decode speed |

These are **exploratory measurements from separate matched experiments**, not a single universal speedup claim. The screen-recorded before/after demo showed 3.7 → 11.8 generation tok/s while capture software was using GPU memory; that is a demo observation, not a controlled benchmark. A short single-run CLI comparison showed 7.2 → 15.3 generation tok/s, but its exact figure should not be generalized. See [the technical report](docs/technical-report.md) for methodology, correctness checks, and limitations. Run [`scripts/bench-avx2-wsl.sh`](scripts/bench-avx2-wsl.sh) for a fresh ABBA-ordered comparison on your machine.

A subsequent ABBA-ordered WSL2 run on this laptop recorded six samples per variant and test: `pp256` median 185.13 → 209.80 tok/s (1.13×), and `tg64` median 7.57 → 11.52 tok/s (1.52×). The decode range was wide (baseline 6.00–8.97; candidate 8.51–15.26 tok/s), so treat these as a workload- and environment-specific result, not a guaranteed speedup. [Every per-repetition sample and its run order is committed](results/rtx4050-abba-2026-09-27.csv).

## Reproduce in WSL2 / Linux

Prerequisites: NVIDIA driver with WSL CUDA support, CUDA toolkit, CMake, C++ compiler, Python 3, and enough disk space for two runtime builds plus the 5.95 GB model file. This exact patch was exercised on an i5-13450HX / RTX 4050 Laptop GPU with CUDA 12.6 and compute capability 8.9. It requires an AVX2-capable CPU for its fast path; the generic implementation remains for other CPUs.

1. Download `Ternary-Bonsai-2-27B-PTQ1_0.gguf` from the [pinned model revision](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf/tree/6ed5e12bf84b7a63069882c91dd9e9218647d17b). Keep weights outside this Git repo. The expected file is `5,946,648,928` bytes, SHA-256 `53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3`.
2. In a Linux shell, set `LAB` to this repo's absolute path and `MODEL` to your GGUF's absolute path. If this repo lives on Windows, access it through `/mnt/c/...` in WSL. Work outside the portfolio repo so the full upstream checkouts cannot accidentally be committed:

```bash
LAB=/absolute/path/to/bonsai-inference-lab
MODEL=/absolute/path/to/Ternary-Bonsai-2-27B-PTQ1_0.gguf
mkdir -p ~/bonsai-repro && cd ~/bonsai-repro
```

3. Build a pristine pinned baseline and an independently patched candidate:

```bash
git clone --filter=blob:none --single-branch --branch prism \
  https://github.com/PrismML-Eng/llama.cpp.git prism-base
git -C prism-base switch --detach 9a9394a895b96003ca842a6041cb28ac49a108f7
git -C prism-base worktree add --detach ../prism-avx2 9a9394a895b96003ca842a6041cb28ac49a108f7
git -C prism-avx2 apply --check "$LAB/patches/ptq1-avx2.patch"
git -C prism-avx2 apply "$LAB/patches/ptq1-avx2.patch"

for tree in prism-base prism-avx2; do
  cmake -S "$tree" -B "$tree/build-cuda-89" -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=ON -DGGML_NATIVE=ON -DCMAKE_CUDA_ARCHITECTURES=89
  cmake --build "$tree/build-cuda-89" --config Release -j 8 \
    --target llama-cli llama-bench test-backend-ops
done
```

For another GPU, replace `89` with its CUDA compute capability. Preserve the same build options and model in both trees. `GGML_NATIVE=ON` targets your local CPU; do not redistribute these binaries as portable AVX2 builds.

Run the candidate interactively, with a 1024-token context and quantized KV cache to leave memory headroom:

```bash
./prism-avx2/build-cuda-89/bin/llama-cli -m "$MODEL" -ngl 99 -fa on -t 16 \
  -ot '^output.weight=CPU' -c 1024 -ctk q8_0 -ctv q8_0 -b 64 -ub 64 \
  -p 'Write a short note on AI safety.' -n 384 --temp 0 -s 42 --single-turn
```

Run the same command with `./prism-base/build-cuda-89/bin/llama-cli` for the generic CPU-head baseline. Removing `-ot '^output.weight=CPU'` gives the original all-GPU placement, but it is a **different placement comparison**, not an isolated AVX2 comparison.

## Benchmark and verify

```bash
bash "$LAB/scripts/bench-avx2-wsl.sh" "$MODEL" \
  "$PWD/prism-base/build-cuda-89/bin/llama-bench" \
  "$PWD/prism-avx2/build-cuda-89/bin/llama-bench"

"$PWD/prism-avx2/build-cuda-89/bin/test-backend-ops" test -b CPU -o MUL_MAT -p ptq1_0

g++ -O2 -std=c++17 -I prism-avx2/ggml/include -I prism-avx2/ggml/src \
  -I prism-avx2/ggml/src/ggml-cpu "$LAB/checks/ptq1-avx2-dot-check.cpp" \
  -L prism-avx2/build-cuda-89/bin \
  -Wl,-rpath,"$PWD/prism-avx2/build-cuda-89/bin" \
  -lggml-cpu -lggml-base -o /tmp/ptq1-avx2-dot-check
/tmp/ptq1-avx2-dot-check
```

The benchmark writes raw JSON samples, stderr, and GPU-state snapshots to an ignored `artifacts/benchmarks/` directory and prints medians, median absolute deviations, ranges, and candidate/baseline ratios. It validates the model digest before running. The backend test covers supported PTQ1 CPU matmul cases; inspect its per-case output rather than treating a process exit alone as broad quality proof.

## Repository map

- `patches/`: isolated AVX2 runtime change; no model weights or full upstream fork vendored.
- `checks/`: independent generic-vs-AVX2 PTQ1 dot-product oracle.
- `scripts/bench-avx2-wsl.sh` and `scripts/summarize-bench.py`: current WSL2 benchmark capture and summary.
- `docs/technical-report.md`: mechanism, experimental evidence, and known boundaries.
- `results/`: compact per-repetition benchmark samples and capture metadata; unredacted local logs remain ignored.
- `config/`, `reports/vendor-baseline-2026-09-23.md`, and the PowerShell scripts: earlier Windows Phase 0 baseline material, retained for provenance; they are **not** the current AVX2 benchmark path.

## Boundaries and credit

The strongest result is for short, single-user decode on one laptop. Long-context results were unstable, and this repo does not claim production serving, broad GPU portability, quality-equivalent behavior across a benchmark suite, or a universal 3× speedup. The model and PTQ1 format are PrismML's work; llama.cpp and its GGML backends are upstream work. This repo contains the focused experiment and patch. See the [PrismML model page](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf) and [runtime fork](https://github.com/PrismML-Eng/llama.cpp) for their licenses and attribution.
