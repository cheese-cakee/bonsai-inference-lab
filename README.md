# Bonsai 27B on a 6 GB laptop GPU

[![Verify patch](https://github.com/cheese-cakee/bonsai-inference-lab/actions/workflows/verify.yml/badge.svg)](https://github.com/cheese-cakee/bonsai-inference-lab/actions/workflows/verify.yml)

An inference-engineering case study: running PrismML's
[Ternary Bonsai 2 27B](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf) (5.53 GiB,
1.75 bits per weight) on an RTX 4050 Laptop GPU with 6 GB of VRAM, using PrismML's
[llama.cpp fork](https://github.com/PrismML-Eng/llama.cpp).

Two changes, both measured and both verifiable:

1. **Placement.** Keep all 65 transformer layers on the GPU but move the output head to the CPU.
   This frees 265 MiB of VRAM and pulls the model back from a memory-capacity cliff.
2. **Kernel.** Give the CPU a fast AVX2 PTQ1_0 × Q8_0 dot product for that head. It is **bitwise
   identical** to the generic implementation, 3.45× faster in isolation, and raises generation
   speed from 9.3 to 15.7 tokens/s.

This repository holds the patch, the verification, the benchmark tooling, and the evidence. It
is not a new model, quantizer, or inference engine.

```text
               RTX 4050 Laptop (6 GB)                        i5-13450HX
  tokens ─► [ embeddings + 65 transformer layers ] ─► [ output head, PTQ1_0 × Q8_0, AVX2 ] ─► logits
             all on GPU (-ngl 99)                      on CPU (-ot '^output.weight=CPU')
```

## Results

RTX 4050 Laptop (6141 MiB) + i5-13450HX, WSL2, CUDA 12.6. Both builds keep the output head on
the CPU and differ only in the patch. Five ABBA rounds, 30 samples per arm
([raw data](results/rtx4050-abba-2026-09-30.csv)):

| Workload | Generic CPU head | AVX2 CPU head | Pooled median ratio | Brackets won |
| --- | ---: | ---: | ---: | ---: |
| Generation, `tg64` | 9.33 tok/s | **15.70 tok/s** | **1.68×** | 10 / 10 |
| Prompt, `pp256` | 187.2 tok/s | 209.9 tok/s | 1.12× | 9 / 10 |

The placement change on its own, measured earlier in a matched run:

| Placement | GPU model buffer | `pp256` |
| --- | ---: | ---: |
| Everything on GPU | 5395.33 MiB | 36.63 tok/s |
| Output head on CPU | 5130.10 MiB | 231.96 tok/s |

And the kernel in isolation, single-threaded, 4096 × 5120 rows, 36 interleaved samples each
([raw data](results/ptq1-microbench-2026-09-30.csv)):

| Dot product | Median time | Speedup |
| --- | ---: | ---: |
| Generic | 7.94 ms | 1.00× |
| AVX2, first version | 3.49 ms | 2.27× |
| AVX2, current | 2.30 ms | 3.45× |

Generation gains most because the CPU head runs once for every generated token, about 1.3 billion
multiply-adds each time. Prompt processing also improves, by 1.12×, but it computes the head far
less often, and this repository does not isolate where that smaller gain comes from.

Details, methodology, and limits: [**technical report**](docs/technical-report.md).

## Upstream status

This work targets PrismML's fork at `9a9394a8`, when x86 had only the scalar PTQ1_0 kernel.
Since then upstream merged SSE2/SSSE3 kernels
([#248](https://github.com/PrismML-Eng/llama.cpp/pull/248)), and an AVX2/AVX-VNNI/AVX-512 kernel is
under review ([#250](https://github.com/PrismML-Eng/llama.cpp/pull/250)). On the same
microbenchmark and CPU, all three built on the current fork
([raw data](results/upstream-comparison-2026-09-30.csv)):

| Kernel | Median | Bitwise equal to generic |
| --- | ---: | --- |
| Generic | 7.84 ms | yes |
| This repository (AVX2) | 2.25 ms | yes |
| Upstream #248 (SSSE3), merged | 2.08 ms | yes |
| Upstream #250 (AVX2 + AVX-VNNI), open | 0.98 ms | no, last-bit float differences |

So this patch is not proposed upstream: the merged kernel is already slightly faster, and #250 is
about 2.3× faster still, using AVX-VNNI `vpdpbusd` and a `pshufb` unpack. The repository stays a
record of the placement experiment and the measurement method, pinned to the commit it measured.

## Verify the patch (no GPU or model needed)

```bash
bash scripts/verify.sh
```

This fetches the pinned runtime (`9a9394a8`), applies [`patches/ptq1-avx2.patch`](patches/ptq1-avx2.patch),
builds the CPU backend, and runs:

- the [generic-vs-AVX2 oracle](checks/ptq1-avx2-dot-check.cpp): 900 random rows plus every valid
  packed byte at every position, compared **bit for bit**;
- ggml's `test-backend-ops` for `MUL_MAT` with PTQ1_0 weights.

It needs Linux (or WSL2), Git, CMake, a C++17 compiler, and an AVX2 CPU. CI runs it on every push.

## Reproduce the benchmark

Prerequisites: WSL2 or Linux with an NVIDIA GPU, CUDA toolkit, CMake, a C++ compiler, Python 3,
and room for two runtime builds plus the 5.95 GB model file.

1. Download `Ternary-Bonsai-2-27B-PTQ1_0.gguf` from the
   [pinned revision](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf/tree/6ed5e12bf84b7a63069882c91dd9e9218647d17b)
   (5,946,648,928 bytes, SHA-256 `53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3`).
   Keep it outside this repository.

2. Build an unmodified baseline and a patched candidate from the same commit, with the same flags:

   ```bash
   LAB=/absolute/path/to/bonsai-inference-lab
   MODEL=/absolute/path/to/Ternary-Bonsai-2-27B-PTQ1_0.gguf
   mkdir -p ~/bonsai-repro && cd ~/bonsai-repro

   git clone --filter=blob:none --single-branch --branch prism \
     https://github.com/PrismML-Eng/llama.cpp.git prism-base
   git -C prism-base switch --detach 9a9394a895b96003ca842a6041cb28ac49a108f7
   git -C prism-base worktree add --detach ../prism-avx2 9a9394a895b96003ca842a6041cb28ac49a108f7
   git -C prism-avx2 apply "$LAB/patches/ptq1-avx2.patch"

   for tree in prism-base prism-avx2; do
     cmake -S "$tree" -B "$tree/build-cuda-89" -DCMAKE_BUILD_TYPE=Release \
       -DGGML_CUDA=ON -DGGML_NATIVE=ON -DCMAKE_CUDA_ARCHITECTURES=89
     cmake --build "$tree/build-cuda-89" -j 8 --target llama-cli llama-bench
   done
   ```

   Replace `89` with your GPU's compute capability. `GGML_NATIVE=ON` tunes for the local CPU, so
   do not redistribute these binaries.

3. Run the ABBA benchmark:

   ```bash
   ROUNDS=5 bash "$LAB/scripts/bench-avx2-wsl.sh" "$MODEL" \
     prism-base/build-cuda-89/bin/llama-bench prism-avx2/build-cuda-89/bin/llama-bench
   ```

   Each round runs baseline, candidate, candidate, baseline, with a cooldown before every run.
   The script checks the model digest, records binary hashes and GPU state, keeps every raw
   sample, and prints pooled medians plus one paired ratio per bracket. If the brackets disagree,
   the session drifted, and the pooled number should not be trusted.

4. Try it interactively:

   ```bash
   prism-avx2/build-cuda-89/bin/llama-cli -m "$MODEL" -ngl 99 -fa on -t 16 \
     -ot '^output.weight=CPU' -c 1024 -ctk q8_0 -ctv q8_0 -b 64 -ub 64 \
     -p 'Write a short note on AI safety.' -n 384 --temp 0 -s 42 --single-turn
   ```

   Removing `-ot '^output.weight=CPU'` gives the original all-GPU placement. That is a placement
   comparison, not a kernel comparison.

## Repository layout

```text
patches/ptq1-avx2.patch        the runtime change (2 files, against pinned commit 9a9394a8)
checks/                        generic-vs-AVX2 bitwise oracle and microbenchmark
scripts/verify.sh              one-command verification (also run by CI)
scripts/bench-avx2-wsl.sh      ABBA end-to-end benchmark
scripts/summarize-bench.py     pooled and per-bracket summary; writes samples.csv
results/                       committed per-repetition samples with run order and machine state
docs/technical-report.md       format, mechanism, evidence, and limits
archive/phase0-windows/        earlier Windows vendor-baseline material, kept for provenance
```

## Scope and credit

The strongest result is for short, single-user generation on one laptop. This repository does not
claim production serving, portability to other GPUs, long-context stability, or model-quality
equivalence beyond the bitwise kernel check. The model and the PTQ1_0 format are PrismML's work;
llama.cpp and ggml are upstream work. See the
[model page](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf) and the
[runtime fork](https://github.com/PrismML-Eng/llama.cpp) for their licenses.
