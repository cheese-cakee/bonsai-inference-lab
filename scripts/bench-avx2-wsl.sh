#!/usr/bin/env bash
# Matched ABBA comparison of two llama-bench builds that differ only in the
# CPU PTQ1_0 dot product. Each round runs baseline, candidate, candidate,
# baseline so that slow drift (temperature, clocks) hits both arms equally.
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
    echo "Usage: [ROUNDS=5] [COOLDOWN_S=20] $0 MODEL.gguf BASELINE/llama-bench CANDIDATE/llama-bench [OUTPUT_DIR]" >&2
    exit 2
fi

model=$(realpath "$1")
baseline=$(realpath "$2")
candidate=$(realpath "$3")
rounds=${ROUNDS:-5}
cooldown_s=${COOLDOWN_S:-20}
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
output_dir=${4:-"$repo_root/artifacts/benchmarks/$(date -u +%Y%m%dT%H%M%SZ)"}
expected_sha=53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3
bench_flags=(-ngl 99 -fa on -ot '^output.weight=CPU' -t 16 -b 64 -ub 64 -ctk q8_0 -ctv q8_0 -p 256 -n 64 -r 3 -o json)

for executable in "$baseline" "$candidate"; do
    if [[ ! -x "$executable" ]]; then
        echo "Not executable: $executable" >&2
        exit 2
    fi
done

model_sha=$(sha256sum "$model" | cut -d ' ' -f 1)
if [[ "$model_sha" != "$expected_sha" ]]; then
    echo "Wrong GGUF digest: $model_sha (expected $expected_sha)" >&2
    exit 2
fi

if [[ -e "$output_dir" ]]; then
    echo "Output directory already exists: $output_dir" >&2
    exit 2
fi
mkdir -p "$output_dir"

{
    printf 'model_sha256=%s\n' "$model_sha"
    printf 'patch_sha256=%s\n' "$(sha256sum "$repo_root/patches/ptq1-avx2.patch" | cut -d ' ' -f 1)"
    printf 'baseline_binary_sha256=%s\n' "$(sha256sum "$baseline" | cut -d ' ' -f 1)"
    printf 'candidate_binary_sha256=%s\n' "$(sha256sum "$candidate" | cut -d ' ' -f 1)"
    printf 'cpu_model=%s\n' "$(lscpu | sed -n 's/^Model name:[[:space:]]*//p' | head -n 1)"
    printf 'gpu=%s\n' "$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader)"
    printf 'kernel=%s\n' "$(uname -r)"
    printf 'utc_started=%s\n' "$(date -u --iso-8601=seconds)"
    printf 'order=%s rounds of baseline,candidate,candidate,baseline; cooldown %ss before each run\n' "$rounds" "$cooldown_s"
    printf 'flags=%s\n' "${bench_flags[*]}"
} > "$output_dir/manifest.txt"

run=0
for ((round = 1; round <= rounds; round++)); do
    for label in baseline candidate candidate baseline; do
        run=$((run + 1))
        binary=$baseline
        [[ "$label" == candidate ]] && binary=$candidate
        tag=$(printf '%02d-%s' "$run" "$label")
        sleep "$cooldown_s"
        nvidia-smi --query-gpu=temperature.gpu,clocks.gr,memory.used,power.draw --format=csv,noheader,nounits \
            > "$output_dir/$tag-gpu-before.csv"
        echo "Run $run/$((rounds * 4)): $label"
        "$binary" -m "$model" "${bench_flags[@]}" > "$output_dir/$tag.json" 2> "$output_dir/$tag.stderr.log"
    done
done

python3 "$repo_root/scripts/summarize-bench.py" "$output_dir"
echo "Raw evidence: $output_dir"
