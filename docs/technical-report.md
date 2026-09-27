# Technical report: PTQ1_0 inference on RTX 4050 Laptop GPU

## Problem and intervention

The 26.90B-parameter Ternary Bonsai 2 27B PTQ1_0 GGUF is 5.53 GiB (5,946,648,928 bytes) on a GPU reporting 6141 MiB total VRAM. Device occupancy from weights, KV cache, compute buffers, display, and other processes can cross a capacity cliff. A full-GPU placement is therefore not automatically fastest.

The first intervention set `-ot '^output.weight=CPU'`, while keeping all 65 transformer layers on GPU with `-ngl 99`. It lowered reported GPU model-buffer allocation from 5395.33 to 5130.10 MiB (265.23 MiB). In the matched `pp256` placement test, throughput rose from 36.63 to 231.96 tok/s. The exact memory-traffic mechanism behind that jump was **not** directly proven; the capacity-cliff explanation is an inference from headroom and timing, not a claim that a specific paging event was measured.

Moving the output projection to CPU created a new hot path: PTQ1_0 × Q8_0 dot products. The patch adds an AVX2-specific implementation on CPUs with the required support and retains the generic fallback elsewhere. It changes neither model weights nor PTQ1 storage, GPU transformer kernels, attention, or KV-cache math. Thus the clean comparison for the AVX2 effect is **generic CPU head vs AVX2 CPU head**, with every placement and benchmark flag otherwise matched.

## Evidence collected to date

| Experiment | Baseline | Candidate | Interpretation |
| --- | ---: | ---: | --- |
| GPU model buffer, all-GPU vs CPU output head | 5395.33 MiB | 5130.10 MiB | 265.23 MiB freed on GPU |
| `pp256`, all-GPU vs CPU output head | 36.63 tok/s | 231.96 tok/s | Placement effect on this memory-constrained machine |
| `tg64`, all-GPU vs generic CPU output head | 8.57 tok/s | 9.53 tok/s | Placement did not worsen short decode in that matched run |
| `tg64`, generic vs AVX2 CPU head | 9.85 ± 0.05 tok/s | 15.44 ± 1.07 tok/s | Isolated AVX2 effect, three llama-bench repetitions |
| `pp256`, generic vs AVX2 CPU head | 231.27 ± 5.49 tok/s | 238.62 ± 6.55 tok/s | Smaller prefill effect |
| `pp512+tg128`, generic vs AVX2 CPU head | 41.65 ± 1.31 tok/s | 57.80 ± 0.28 tok/s | Combined-workload throughput |

The ± figures above are from `llama-bench`'s three repetitions in the original experiments. They are not confidence intervals, and the experiments were not one globally randomized campaign. The new ABBA script records raw JSON and spread for a repeatable publication run. Avoid combining rows into one multiplier: the first three compare *placement*, while the latter three compare *CPU dot-product implementations*.

The [September 27 UTC ABBA run](../results/rtx4050-abba-2026-09-27.csv) completed with six raw samples per variant/test. For `pp256`, median throughput was 185.13 baseline and 209.80 candidate tok/s (1.13×), with median absolute deviations of 6.06 and 3.35. For `tg64`, it was 7.57 and 11.52 tok/s (1.52×), with median absolute deviations of 1.24 and 2.96. The especially broad decode ranges—6.00–8.97 and 8.51–15.26—matter more than a single headline number. The CSV includes exact run ordering, per-repetition token rates and durations, model and patch digests, build revision, and command flags. Local full logs and GPU-state snapshots remain in ignored `artifacts/`.

The screen-recorded demo's 3.7 → 11.8 generation tok/s is a **single captured run** with recording software consuming GPU resources. A separate CLI prompt returned the same visible AI-safety note at 7.2 → 15.3 generation tok/s, also a single run. These show end-to-end usability, not a quality equivalence or performance distribution.

## Correctness work

- A standalone oracle compared generic and AVX2 PTQ1 dot products: 900/900 randomized nonzero cases bitwise equal.
- An exhaustive valid packed-byte sweep covered 5832 cases, with zero mismatches.
- The CPU backend PTQ1 operation tests passed 45/45 supported cases in the prior local run.
- A fixed prompt generated the same short continuation in the compared builds.

The oracle is in `checks/ptq1-avx2-dot-check.cpp`. Bitwise dot-product equality is stronger than merely close numeric agreement for that operation, but it does **not** establish model-wide quality parity over a diverse prompt corpus. The output-head placement can also change scheduling and memory conditions even when its numerical computation is intended to be equivalent.

## Profiling and engineering rationale

Nsight Systems captured GPU kernel timelines for prefill and decode. Earlier CLI 2024.5.1 reporting did not expose CUDA kernel data in WSL; CLI 2026.5.1 did. The more useful deduction was to distinguish short prefill (`pp256`) from serial token generation (`tg64`) and check GPU headroom before treating the largest reported kernel as the sole bottleneck. A direct CUDA micro-optimization can be irrelevant or harmful if memory pressure dominates end-to-end time. Nsight Compute performance-counter access was initially blocked by `ERR_NVGPUCTRPERM`; no counter-derived roofline claim is made here.

## Limits and next release gates

- One laptop, one model revision, one pinned runtime base, and a short single-user workload; no general GPU or CPU portability claim.
- 1024-token interactive context leaves safer memory headroom. Larger contexts need their own memory and latency sweep; earlier 2048+128 measurements were unstable.
- Benchmark outcomes vary with laptop power limits, thermal state, WSL/driver version, display load, and background recording. The ABBA script records state but does not lock power or clock settings.
- The patch has not undergone independent upstream review or production-serving validation. No server-load, concurrency, latency-tail, or long-running soak claim is made.
- Experimental FP16-state and CUDA GEMV variants were explored but excluded after disappointing tradeoffs. They are not part of this release.

This is a targeted inference optimization artifact: source patch, correctness oracle, repeatable matched benchmark, and honest scope. Further work should be chosen from profiles and end-to-end measurements, not from a promised blanket speedup.
