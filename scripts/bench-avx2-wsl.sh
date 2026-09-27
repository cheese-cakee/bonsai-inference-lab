#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
    echo "Usage: $0 MODEL.gguf BASELINE/llama-bench CANDIDATE/llama-bench [OUTPUT_DIR]" >&2
    exit 2
fi

model=$(realpath "$1")
baseline=$(realpath "$2")
candidate=$(realpath "$3")
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
output_dir=${4:-"$repo_root/artifacts/benchmarks/$(date -u +%Y%m%dT%H%M%SZ)"}
expected_sha=53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3

for executable in "$baseline" "$candidate"; do
    if [[ ! -x "$executable" ]]; then
        echo "Not executable: $executable" >&2
        exit 2
    fi
done

model_sha=$(sha256sum "$model" | cut -d ' ' -f 1)
if [[ "$model_sha" != "$expected_sha" ]]; then
    echo "Wrong GGUF digest: $model_sha" >&2
    echo "Expected: $expected_sha" >&2
    exit 2
fi

if [[ -e "$output_dir" ]]; then
    echo "Output directory already exists: $output_dir" >&2
    exit 2
fi
mkdir -p "$output_dir"

{
    printf 'model_sha256=%s\n' "$model_sha"
    printf 'model_bytes=%s\n' "$(stat -c %s "$model")"
    printf 'baseline=%s\n' "$baseline"
    printf 'candidate=%s\n' "$candidate"
    printf 'patch_sha256=%s\n' "$(sha256sum "$repo_root/patches/ptq1-avx2.patch" | cut -d ' ' -f 1)"
    printf 'baseline_binary_sha256=%s\n' "$(sha256sum "$baseline" | cut -d ' ' -f 1)"
    printf 'candidate_binary_sha256=%s\n' "$(sha256sum "$candidate" | cut -d ' ' -f 1)"
    printf 'cpu_model=%s\n' "$(lscpu | sed -n 's/^Model name:[[:space:]]*//p' | head -n 1)"
    printf 'utc_started=%s\n' "$(date -u --iso-8601=seconds)"
    printf 'warmup=llama-bench default per test\n'
    printf 'order=baseline,candidate,candidate,baseline\n'
    printf 'workload=pp256,tg64; r=3; ngl=99; fa=on; output.weight=CPU; threads=16; b=64; ub=64; KV=Q8_0/Q8_0\n'
    uname -a
    if command -v nvcc >/dev/null 2>&1; then
        nvcc --version | tail -n 4
    elif [[ -x /usr/local/cuda/bin/nvcc ]]; then
        /usr/local/cuda/bin/nvcc --version | tail -n 4
    else
        printf 'nvcc=not found on PATH or /usr/local/cuda/bin\n'
    fi
    nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,memory.free,pstate,clocks.gr,clocks.mem,power.draw --format=csv,noheader
} > "$output_dir/manifest.txt" 2>&1

order=(baseline candidate candidate baseline)
for index in "${!order[@]}"; do
    label=${order[$index]}
    if [[ "$label" == baseline ]]; then
        binary=$baseline
    else
        binary=$candidate
    fi
    run=$(printf '%02d' "$((index + 1))")
    nvidia-smi --query-gpu=timestamp,memory.used,memory.free,pstate,clocks.gr,clocks.mem,temperature.gpu,power.draw --format=csv,noheader > "$output_dir/$run-$label-gpu-before.csv"
    echo "Run $run/04: $label"
    "$binary" -m "$model" -ngl 99 -fa on -ot '^output.weight=CPU' -t 16 \
        -b 64 -ub 64 -ctk q8_0 -ctv q8_0 -p 256 -n 64 -r 3 -o json \
        > "$output_dir/$run-$label.json" 2> "$output_dir/$run-$label.stderr.log"
done

python3 "$repo_root/scripts/summarize-bench.py" "$output_dir"
echo "Raw evidence: $output_dir"
