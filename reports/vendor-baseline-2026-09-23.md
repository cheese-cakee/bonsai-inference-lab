# Literal vendor baseline: Windows, RTX 4050 Laptop 6 GB

Status: **startup validated, one correctness smoke passed, exploratory performance only**.

## Artifacts

- Setup log: `artifacts/vendor-baseline/setup-default.log`
- Server log: `artifacts/vendor-baseline/server-default.log`
- Environment manifest: `artifacts/environment/machine-state-20260920-231818.json`
- Runtime marker: Bonsai demo `bin/cuda/.llama_release` = `prism-b10709-9a9394a cuda-12.4`
- Model metadata: both GGUF files report revision `6ed5e12bf84b7a63069882c91dd9e9218647d17b`

The environment manifest was captured on September 20, before the September 23 serving run. It
pins the hardware and installed software but is not a same-session performance manifest. Capture
a fresh manifest for the measured sweep.

## Unchanged vendor path

Ran `setup.ps1` and then `scripts/start_llama_server.ps1` with no `BONSAI_*` overrides.
Setup completed after the previous task was interrupted. It downloaded:

| File | Bytes |
| --- | ---: |
| `Ternary-Bonsai-2-27B-PQ2_0.gguf` | 7,206,168,928 |
| `Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf` | 629,246,976 |

The launcher selected `-ngl 99`, 16,384 context, four server slots, multimodal projector, and
CUDA 12.4. The server reached `GET /health = {"status":"ok"}` in about 10 seconds. It logged:

> failed to fit params to free device memory: n_gpu_layers already set by user to 99, abort

At idle after load, `nvidia-smi` reported 5,859 MiB used and 63 MiB free out of 6,141 MiB.
That is a fragile state even though the server started. The Windows process had about 11.4 GB
working set, which is consistent with substantial system memory use; this is not a measured
GPU transfer or shared-memory claim.

## Exploratory requests

One 32-token response with a request-level `thinking_budget_tokens=0` returned empty visible
content and `finish_reason=length`. The server reported 65 prompt tokens at 20.52 tokens/s and
32 generated tokens at 3.77 tokens/s. This shows that that request parameter did not reliably
disable hidden reasoning in this setup.

A second request added `chat_template_kwargs: {"enable_thinking": false}` and returned the exact
`BONSAI_OK` marker with `finish_reason=stop` in 3.16 seconds. Server timing: 25 prompt tokens at
16.76 tokens/s and six generated tokens at 3.55 tokens/s. Temperature was zero. These two
requests are smoke checks, not a throughput benchmark: there was no warmup protocol, repeated
sample set, or matched environment comparison.

## Decision

The official setup and launcher work, but the default is not a stable daily configuration for
this laptop. Continue with text-only PTQ1, one slot, 4K context, bounded reasoning, and measured
GPU offload. Compare against PQ2 under the same controlled conditions before choosing a format.
