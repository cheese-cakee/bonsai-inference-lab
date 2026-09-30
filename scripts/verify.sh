#!/usr/bin/env bash
# Proves, without a GPU or the model, that the patch applies to the pinned
# runtime and that the AVX2 kernel is bitwise identical to the generic one.
set -euo pipefail

readonly UPSTREAM_URL=https://github.com/PrismML-Eng/llama.cpp.git
readonly UPSTREAM_COMMIT=9a9394a895b96003ca842a6041cb28ac49a108f7

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
work_dir=${1:-"$repo_root/artifacts/verify"}
src="$work_dir/llama.cpp"
build="$src/build-cpu"
jobs=$(nproc)

if ! grep -qw avx2 /proc/cpuinfo; then
    echo "This check needs an AVX2 CPU; the patch only changes the AVX2 path." >&2
    exit 2
fi

echo "== Fetching pinned runtime $UPSTREAM_COMMIT"
if [[ ! -d "$src/.git" ]]; then
    git init -q "$src"
    git -C "$src" remote add origin "$UPSTREAM_URL"
fi
git -C "$src" fetch -q --depth 1 origin "$UPSTREAM_COMMIT"
git -C "$src" checkout -q --force --detach FETCH_HEAD
git -C "$src" clean -q -fd --exclude=build-cpu

echo "== Applying patches/ptq1-avx2.patch"
git -C "$src" apply --check "$repo_root/patches/ptq1-avx2.patch"
git -C "$src" apply "$repo_root/patches/ptq1-avx2.patch"

echo "== Building the CPU backend"
run_logged() {
    local log=$1
    shift
    "$@" > "$log" 2>&1 || { tail -n 40 "$log" >&2; exit 1; }
}
run_logged "$work_dir/configure.log" \
    cmake -S "$src" -B "$build" -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DGGML_CUDA=OFF -DLLAMA_CURL=OFF
run_logged "$work_dir/build.log" \
    cmake --build "$build" -j "$jobs" --target ggml-cpu ggml-base test-backend-ops

echo "== Generic vs AVX2 dot-product oracle"
c++ -O2 -std=c++17 -I "$src/ggml/include" -I "$src/ggml/src" -I "$src/ggml/src/ggml-cpu" \
    "$repo_root/checks/ptq1-avx2-dot-check.cpp" -L "$build/bin" -Wl,-rpath,"$build/bin" \
    -lggml-cpu -lggml-base -o "$work_dir/ptq1-avx2-dot-check"
"$work_dir/ptq1-avx2-dot-check"

echo "== ggml backend tests: MUL_MAT with PTQ1_0 weights"
"$build/bin/test-backend-ops" test -b CPU -o MUL_MAT -p ptq1_0 | tail -n 4

echo "== All checks passed"
