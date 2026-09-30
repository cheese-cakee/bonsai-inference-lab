# Technical report: a 27B ternary model on a 6 GB laptop GPU

## 1. Problem

[Ternary Bonsai 2 27B](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf) has 26.90 billion
parameters. In PrismML's PTQ1_0 format the GGUF file is 5.53 GiB (5,946,648,928 bytes). The target
GPU, an RTX 4050 Laptop, reports 6141 MiB of VRAM, and that memory is shared with the KV cache,
compute buffers, the CUDA context, and the Windows display. Loading everything onto the GPU leaves
almost no headroom, and a GPU that runs out of memory headroom does not degrade gracefully: it falls
off a cliff.

The goal was the fastest short, single-user generation on this laptop without changing the model's
weights or output.

## 2. The PTQ1_0 format, from first principles

A ternary model's weights are each one of `{-1, 0, +1}` times a per-group scale. PTQ1_0 stores
groups of 128 weights in a 28-byte block, 1.75 bits per weight:

| Field | Bytes | Holds |
| --- | ---: | --- |
| `qs` | 24 | 120 weights, 5 per byte |
| `qh` | 2 | 8 weights, 4 per byte |
| `d` | 2 | FP16 scale: the largest absolute weight in the group |

**Quantizing.** Each weight `x` becomes the trit `t = round(x / d)` in `{-1, 0, +1}`, stored as
`t + 1` in `{0, 1, 2}`. Five trits form a base-3 number `q = t₀·81 + t₁·27 + t₂·9 + t₃·3 + t₄` in
`[0, 242]`. Since `3⁵ = 243 ≤ 256`, five trits fit in one byte, 1.6 bits each, close to the
`log₂ 3 ≈ 1.585` minimum.

**The storage trick.** The byte does not hold `q` itself but `⌈q · 256 / 243⌉`: the number scaled
into a fraction of 256. Now the leading base-3 digit can be read with a multiply and a shift:

```
digit 0 = (byte × 3) >> 8
digit n = ((uint8_t)(byte × 3ⁿ) × 3) >> 8
```

Multiplying by `3` moves the leading digit above bit 8, like multiplying a decimal fraction by 10
moves its first digit before the point. Multiplying by `3ⁿ` modulo 256 discards the first `n`
digits. Rounding up when storing guarantees the truncation never lands one digit too low. Every
operation is an 8- or 16-bit integer multiply, which SIMD units do well. This is the same idea as
llama.cpp's TQ1_0 format.

**The dot product.** In the model, a weight row is multiplied by an activation vector quantized to
Q8_0 (32 int8 values plus an FP16 scale per block). For one 128-weight block, with four activation
blocks `k`:

```
block_sum = d_w × Σ_k ( d_y[k] × Σ_{32 values} t × y )
```

The inner sums are exact integers; only the four scales introduce floating-point rounding.

## 3. Intervention 1: move only the output head to the CPU

The output projection `output.weight` maps the hidden state (5120 values) to a score for every
vocabulary token. Moving only this tensor to the CPU (`-ot '^output.weight=CPU'`), while keeping all
65 transformer layers on the GPU (`-ngl 99`), lowered the reported GPU model buffer from 5395.33 MiB
to 5130.10 MiB, freeing 265.23 MiB.

In a matched prompt-processing test (`pp256`), throughput went from 36.63 to 231.96 tok/s. The exact
mechanism was **not** directly measured. The explanation that fits the evidence is that the
all-GPU placement sat past a memory-capacity cliff and the freed headroom moved it back; no specific
paging or fallback event was traced.

The move has a cost. The output head now runs on the CPU for every generated token: 265 MiB of
weights at 1.75 bits each is about 1.3 billion ternary-by-int8 multiply-adds per token, in
PTQ1_0 × Q8_0 dot products. The fork had only a
scalar implementation of that dot product for x86.

## 4. Intervention 2: an AVX2 PTQ1_0 × Q8_0 dot product

The patch ([`patches/ptq1-avx2.patch`](../patches/ptq1-avx2.patch)) adds
`ggml_vec_dot_ptq1_0_q8_0` for AVX2 CPUs and keeps the generic version for everything else. It
changes nothing else: no model weights, no storage format, no GPU kernel, no attention or KV-cache
code.

Per 128-weight block the kernel:

1. Widens the 24 `qs` bytes to 16-bit lanes and extracts all five digits of 16 bytes at once with
   `vpmullw` (multiply by `3ⁿ`), a mask to the low byte, another multiply by 3, and a shift.
2. Decodes the two `qh` bytes the same way, with multipliers `1, 1, 3, 3, 9, 9, 27, 27` so that
   the eight trits come out in the order the generic decoder uses.
3. Assembles the four 32-weight vectors directly in registers, in exactly the generic weight order.
4. Multiplies each by its 32 int8 activations with `vpmaddubsw` / `vpmaddwd` (the sign trick in
   ggml's `mul_add_epi8`, since `vpmaddubsw` needs one unsigned operand).
5. Reduces the four blocks' integer sums together with three `vphaddd` steps instead of four
   separate horizontal sums.
6. Applies the scales with **the same floating-point operations, in the same order**, as the
   generic kernel.

Step 6 is what makes the result bitwise identical. Integer addition is associative, so the integer
work can be reordered freely. Floating-point addition is not, so the float code is kept
statement-for-statement the same as the generic kernel.

**Version history.** The first AVX2 version decoded the trits into a 128-byte stack buffer and
reloaded them, and reduced each 32-weight block separately. The current version keeps the trits in
registers and reduces the four blocks together. On the single-threaded microbenchmark in
[`checks/ptq1-avx2-dot-check.cpp`](../checks/ptq1-avx2-dot-check.cpp) (4096 rows × 5120 columns,
i5-13450HX, 24 processes alternating between the two builds, 36 samples per kernel;
[raw data](../results/ptq1-microbench-2026-09-30.csv)):

| Kernel | Median | Range | Speedup |
| --- | ---: | ---: | ---: |
| Generic (scalar) | 7.94 ms | 7.64-9.43 ms | 1.00× |
| AVX2, first version | 3.49 ms | 3.28-3.82 ms | 2.27× |
| AVX2, current | 2.30 ms | 2.18-2.62 ms | 3.45× |

The ranges of the two AVX2 versions do not overlap.

## 5. Correctness

| Check | Result |
| --- | --- |
| Random rows, generic vs AVX2, `n` = 128, 256, 5120 | 900/900 bitwise equal, 900/900 non-zero |
| Every valid packed byte (0-242) at every `qs` position | 0 mismatches over 5,832 cases |
| ggml `test-backend-ops`, `MUL_MAT` with PTQ1_0 weights, CPU | 45/45 passed |
| Negative control: a deliberately wrong `qh` multiplier | 810/900 mismatches; the check fails |

Bitwise equality with the generic kernel is a stronger oracle than "close enough": any change in
the order of float operations, or any decode error, shows up. It establishes that the CPU head
computes exactly what it computed before; it does not, by itself, say anything about model quality
relative to other placements or formats.

The whole verification runs without a GPU or the model: [`scripts/verify.sh`](../scripts/verify.sh)
fetches the pinned runtime, applies the patch, builds the CPU backend, and runs the oracle and the
backend tests. CI runs it on every push.

## 6. End-to-end measurements

**Setup.** Baseline: pristine PrismML llama.cpp at `9a9394a8`, whose x86 CPU backend uses the
generic PTQ1_0 dot product. Candidate: the same commit plus the patch. Both are CUDA builds for
`sm_89` with `GGML_NATIVE=ON`, and both run with the output head on the CPU, so the only
difference is the CPU dot product. Flags: `-ngl 99 -fa on -ot '^output.weight=CPU' -t 16 -b 64
-ub 64 -ctk q8_0 -ctv q8_0 -p 256 -n 64 -r 3`. The model, patch, and both binaries are identified by
SHA-256 in the [raw data](../results/rtx4050-abba-2026-09-30.csv).

**Order.** Five rounds of baseline, candidate, candidate, baseline (20 runs, 3 repetitions each),
with a 20-second cooldown before every run. Each adjacent baseline/candidate pair is a bracket; a
real effect should show up in every bracket, while drift within the session would not.

| Workload | Baseline median (MAD) | Candidate median (MAD) | Pooled ratio | Per-bracket ratios |
| --- | ---: | ---: | ---: | --- |
| `tg64` | 9.33 (0.15) tok/s | 15.70 (0.16) tok/s | 1.68× | 1.45-2.54, candidate faster in 10/10 |
| `pp256` | 187.22 (1.70) tok/s | 209.86 (2.98) tok/s | 1.12× | 0.95-1.16, candidate faster in 9/10 |

**Outliers are kept.** Three runs were slow: baseline run 1 (the first run of the session) and
run 12 generated at about 6.2 tok/s instead of about 9.3, and candidate run 2 at 9.0 instead of
about 15.7. Candidate run 14's `pp256` was 180 instead of about 210, which is the one bracket the
candidate lost. The same kind of slowdown hit the September 27 run, whose two brackets disagreed
(generation ratios 2.36 and 0.97, first AVX2 version), which is why that session is not used as evidence. The pooled
medians, and a candidate win in every generation bracket, are robust to these runs.

**Why generation gains most.** In generation, every step computes the output head once: about
1.3 billion ternary-by-int8 multiply-adds on the CPU while the GPU waits. A 3.45× faster dot
product shrinks that serial CPU phase. Prompt processing computes the head far less often; its
smaller 1.12× gain is measured but its source is not isolated here.

**Placement, measured separately.** Moving `output.weight` from the GPU to the CPU with the
generic kernel is a different comparison. In a matched `pp256` run it went from 36.63 to
231.96 tok/s (section 3), and a matched short-generation run showed no loss (8.57 to 9.53 tok/s
`tg64`).

## 7. Profiling notes

Nsight Systems captured GPU kernel timelines for prefill and decode. (CLI 2024.5.1 did not report
CUDA kernel data under WSL; 2026.5.1 did.) The useful conclusion was to separate short prefill
(`pp256`) from serial token generation (`tg64`) and to check memory headroom before treating the
largest GPU kernel as the bottleneck: a CUDA micro-optimization is irrelevant, or harmful, when
memory pressure dominates end-to-end time. Nsight Compute counters were blocked by
`ERR_NVGPUCTRPERM`, so no counter-derived roofline claim is made.

Two other directions, an FP16 recurrent-state variant and a CUDA GEMV for the head, were tried and
dropped after disappointing trade-offs. They are not part of this repository.

## 8. Limits

- One laptop, one model revision, one pinned runtime, short single-user workloads.
- Laptop results move with power limits, temperature, WSL and driver versions, and display load.
  The benchmark records GPU temperature and clock before every run but does not lock them.
- A 1024-token context leaves safe memory headroom. Longer contexts were unstable and need their
  own memory and latency sweep.
- The patch has not been reviewed upstream or validated for production serving, concurrency, or
  tail latency.
