# Phase 0: Windows vendor baseline (archived)

Before the AVX2 work, the project measured PrismML's unchanged Windows demo on the same
laptop. This folder keeps that material for provenance. It is not part of the current
benchmark or verification path.

| Path | Purpose |
| --- | --- |
| `config/upstreams.json` | Pinned demo, runtime fork, and model revisions used in Phase 0 |
| `config/prompts.json` | Fixed prompt corpus with exact-match and JSON-shape expectations |
| `reports/vendor-baseline-2026-09-23.md` | What the unchanged vendor path did on a 6 GB GPU |
| `scripts/*.ps1` | Capture machine state, start the text server, and measure an endpoint |

The scripts resolve paths relative to this folder, so they still run from here
(`pwsh archive/phase0-windows/scripts/validate-config.ps1`). Their outputs land in the ignored
`archive/phase0-windows/artifacts/` directory.
