#include "ggml-cpu/quants.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

int main() {
    ggml_cpu_init();
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> values(-3.0f, 3.0f);
    constexpr int cases = 300;
    constexpr int sizes[] = {128, 256, 5120};
    int mismatches = 0;
    int nonzero = 0;

    for (int n : sizes) {
        std::vector<float> x(n), y(n);
        std::vector<block_ptq1_0> qx(n / QK_PTQ1_0);
        std::vector<block_q8_0> qy(n / QK8_0);
        for (int trial = 0; trial < cases; ++trial) {
            for (int i = 0; i < n; ++i) {
                x[i] = values(rng);
                y[i] = values(rng);
            }
            quantize_row_ptq1_0(x.data(), qx.data(), n);
            quantize_row_q8_0(y.data(), qy.data(), n);

            float reference = 0.0f;
            float candidate = 0.0f;
            ggml_vec_dot_ptq1_0_q8_0_generic(n, &reference, 0, qx.data(), 0, qy.data(), 0, 1);
            ggml_vec_dot_ptq1_0_q8_0(n, &candidate, 0, qx.data(), 0, qy.data(), 0, 1);
            nonzero += reference != 0.0f;
            if (trial == 0) {
                std::printf("sample n=%d x_scale=%u y_scale=%u output=%.9g\n", n, (unsigned) qx[0].d, (unsigned) qy[0].d, reference);
            }
            if (std::memcmp(&reference, &candidate, sizeof(float)) != 0) {
                ++mismatches;
                if (mismatches <= 5) {
                    std::printf("mismatch n=%d trial=%d reference=%.9g candidate=%.9g\n", n, trial, reference, candidate);
                }
            }
        }
        std::printf("n=%d cases=%d complete\n", n, cases);
    }

    std::printf("bitwise_mismatches=%d/%d\n", mismatches, cases * 3);
    std::printf("nonzero_outputs=%d/%d\n", nonzero, cases * 3);

    std::vector<float> edge_input(128, 1.0f);
    block_ptq1_0 edge_weight;
    block_q8_0 edge_activation[128 / QK8_0];
    quantize_row_ptq1_0(edge_input.data(), &edge_weight, 128);
    quantize_row_q8_0(edge_input.data(), edge_activation, 128);
    for (size_t pos = 0; pos < sizeof(edge_weight.qs); ++pos) {
        for (int packed = 0; packed <= 242; ++packed) {
            edge_weight.qs[pos] = static_cast<uint8_t>(packed);
            float reference = 0.0f, candidate = 0.0f;
            ggml_vec_dot_ptq1_0_q8_0_generic(128, &reference, 0, &edge_weight, 0, edge_activation, 0, 1);
            ggml_vec_dot_ptq1_0_q8_0(128, &candidate, 0, &edge_weight, 0, edge_activation, 0, 1);
            mismatches += std::memcmp(&reference, &candidate, sizeof(float)) != 0;
        }
    }
    std::printf("after_packed_byte_sweep_mismatches=%d\n", mismatches);

    constexpr int width = 5120;
    constexpr int rows = 4096;
    std::vector<float> input(width);
    std::vector<block_ptq1_0> weights(rows * width / QK_PTQ1_0);
    std::vector<block_q8_0> activation(width / QK8_0);
    for (float & v : input) v = values(rng);
    quantize_row_q8_0(input.data(), activation.data(), width);
    for (int row = 0; row < rows; ++row) {
        for (float & v : input) v = values(rng);
        quantize_row_ptq1_0(input.data(), weights.data() + row * width / QK_PTQ1_0, width);
    }
    using dot_fn = decltype(&ggml_vec_dot_ptq1_0_q8_0);
    auto time_dot = [&](dot_fn fn, const char * name) {
        for (int repeat = 0; repeat < 3; ++repeat) {
            float checksum = 0.0f;
            const auto begin = std::chrono::steady_clock::now();
            for (int row = 0; row < rows; ++row) {
                float output;
                fn(width, &output, 0, weights.data() + row * width / QK_PTQ1_0, 0, activation.data(), 0, 1);
                checksum += output;
            }
            const auto end = std::chrono::steady_clock::now();
            const double milliseconds = std::chrono::duration<double, std::milli>(end - begin).count();
            std::printf("variant=%s repeat=%d ms=%.3f checksum=%.9g\n", name, repeat, milliseconds, checksum);
        }
    };
    time_dot(ggml_vec_dot_ptq1_0_q8_0_generic, "generic");
    time_dot(ggml_vec_dot_ptq1_0_q8_0, "avx2");
    return mismatches == 0 && nonzero > cases * 2 ? 0 : 1;
}
