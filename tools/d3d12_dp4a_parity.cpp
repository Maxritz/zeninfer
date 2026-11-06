// DP4A decode parity: CPU double-precision reference vs the D3D12
// layer_decode_pipeline path (ZEN_D3D12_DP4A=0 and =1, toggled in-process
// between runs of the same synthetic layer). One layer, one position.
//
// PASS criteria: fp32 max-rel <= 1e-4, DP4A max-rel <= 2e-2 (whole-layer
// cumulative; a single matvec is ~4-8e-3). A broken matvec gives ~1.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

#include "zen/d3d12_backend.h"

namespace {

std::uint16_t float_to_half(float value);
std::uint16_t float_to_half(float value) {
    std::uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const std::uint32_t sign = (bits >> 16) & 0x8000u;
    int exponent = static_cast<int>((bits >> 23) & 0xFFu) - 127 + 15;
    std::uint32_t mantissa = bits & 0x7FFFFFu;
    if (exponent <= 0) {
        return static_cast<std::uint16_t>(sign);
    }
    if (exponent >= 31) {
        return static_cast<std::uint16_t>(sign | 0x7C00u);
    }
    double scaled = static_cast<double>(mantissa) / static_cast<double>(1u << 23);
    std::uint32_t half_mantissa = static_cast<std::uint32_t>(scaled * 1024.0 + 0.5) & 0x3FFu;
    return static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(exponent) << 10) |
                                      half_mantissa);
}

float half_to_float(std::uint16_t h) {
    const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000u) << 16;
    const std::uint32_t exp = (h >> 10) & 0x1Fu;
    const std::uint32_t mant = h & 0x3FFu;
    std::uint32_t bits;
    if (exp == 0) {
        bits = sign;
        if (mant) {
            std::uint32_t m = mant;
            int e = -1;
            do {
                m <<= 1;
                ++e;
            } while ((m & 0x400u) == 0);
            bits = sign | (static_cast<std::uint32_t>(127 - 15 - e) << 23) | ((m & 0x3FFu) << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 112u) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

// Raw Q8_0 weight blob for one matrix: rows x blocks x (scale 2B + 32 quants).
std::vector<std::uint8_t> make_weights(std::size_t rows, std::size_t cols, std::mt19937& rng) {
    const std::size_t bpr = cols / 32;
    std::vector<std::uint8_t> w(rows * bpr * 34);
    std::uniform_int_distribution<int> qd(-127, 127);
    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t b = 0; b < bpr; ++b) {
            std::uint8_t* block = w.data() + (r * bpr + b) * 34;
            const std::uint16_t half = float_to_half(0.02f);
            block[0] = static_cast<std::uint8_t>(half & 0xFFu);
            block[1] = static_cast<std::uint8_t>(half >> 8);
            for (std::size_t k = 0; k < 32; ++k) {
                block[2 + k] = static_cast<std::uint8_t>(static_cast<std::int8_t>(qd(rng)));
            }
        }
    }
    return w;
}

void cpu_matvec(const std::vector<std::uint8_t>& w, const std::vector<float>& x,
                std::vector<float>& y, std::size_t rows, std::size_t cols) {
    const std::size_t bpr = cols / 32;
    for (std::size_t r = 0; r < rows; ++r) {
        double acc = 0.0;
        for (std::size_t b = 0; b < bpr; ++b) {
            const std::uint8_t* block = w.data() + (r * bpr + b) * 34;
            const double scale = half_to_float(
                static_cast<std::uint16_t>(block[0] | (block[1] << 8)));
            double dot = 0.0;
            for (std::size_t k = 0; k < 32; ++k) {
                dot += static_cast<double>(static_cast<std::int8_t>(block[2 + k])) *
                       static_cast<double>(x[b * 32 + k]);
            }
            acc += scale * dot;
        }
        y[r] = static_cast<float>(acc);
    }
}

void cpu_rmsnorm(const std::vector<float>& x, const std::vector<float>& w,
                 std::vector<float>& y, float eps) {
    double ss = 0.0;
    for (float v : x) {
        ss += double(v) * double(v);
    }
    const double inv = 1.0 / std::sqrt(ss / double(x.size()) + double(eps));
    for (std::size_t i = 0; i < x.size(); ++i) {
        y[i] = static_cast<float>(double(x[i]) * inv * double(w[i]));
    }
}

}  // namespace

int main() {
    if (!zen::d3d12::init()) {
        std::printf("SKIP: no Direct3D 12.1 backend available\n");
        return 77;
    }
    constexpr std::size_t hidden = 128;
    constexpr std::size_t heads = 4;
    constexpr std::size_t kv_heads = 2;
    constexpr std::size_t head_dim = 16;
    constexpr std::size_t ffn_rows = 256;
    constexpr std::size_t pairs = 8;
    constexpr std::size_t context = 16;
    constexpr std::size_t q_width = heads * head_dim;    // 64
    constexpr std::size_t kv_width = kv_heads * head_dim;  // 32
    constexpr float epsilon = 1e-5f;
    constexpr std::size_t pos = 0;

    if (!zen::d3d12::kv_cache_configure(1, context, kv_heads, head_dim)) {
        std::printf("FAIL: kv_cache_configure\n");
        return 1;
    }

    std::mt19937 rng(777);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    const auto q_w = make_weights(q_width, hidden, rng);
    const auto k_w = make_weights(kv_width, hidden, rng);
    const auto v_w = make_weights(kv_width, hidden, rng);
    const auto o_w = make_weights(hidden, q_width, rng);
    const auto g_w = make_weights(ffn_rows, hidden, rng);
    const auto u_w = make_weights(ffn_rows, hidden, rng);
    const auto d_w = make_weights(hidden, ffn_rows, rng);

    std::vector<float> attn_norm_w(hidden, 1.0f), ffn_norm_w(hidden, 1.0f);
    std::vector<float> x(hidden);
    for (float& v : x) {
        v = dist(rng);
    }
    std::vector<float> cos_tab(pairs, 1.0f), sin_tab(pairs, 0.0f);

    // --- CPU double reference for the whole layer ---
    std::vector<float> h(hidden), qr(q_width), kr(kv_width), v(kv_width);
    cpu_rmsnorm(x, attn_norm_w, h, epsilon);
    std::vector<float> qv(q_width), kvv(kv_width);
    cpu_matvec(q_w, h, qv, q_width, hidden);
    cpu_matvec(k_w, h, kvv, kv_width, hidden);
    cpu_matvec(v_w, h, v, kv_width, hidden);
    // rope q
    qr = qv;
    for (std::size_t i = 0; i < pairs; ++i) {
        const float a = qr[i], b = qr[i + pairs];
        qr[i] = a * cos_tab[i] - b * sin_tab[i];
        qr[i + pairs] = a * sin_tab[i] + b * cos_tab[i];
    }
    kr = kvv;
    for (std::size_t i = 0; i < pairs; ++i) {
        const float a = kr[i], b = kr[i + pairs];
        kr[i] = a * cos_tab[i] - b * sin_tab[i];
        kr[i + pairs] = a * sin_tab[i] + b * cos_tab[i];
    }
    // attention (single visible position)
    std::vector<float> att(q_width, 0.0f);
    const float inv_scale = 1.0f / std::sqrt(float(head_dim));
    for (std::size_t hh = 0; hh < heads; ++hh) {
        const std::size_t kh = hh / (heads / kv_heads);
        double mx = -std::numeric_limits<double>::infinity();
        double sc = 0.0;
        for (std::size_t d = 0; d < head_dim; ++d) {
            sc += double(qr[hh * head_dim + d]) * double(kr[kh * head_dim + d]);
        }
        sc *= double(inv_scale);
        mx = sc;
        const double w1 = std::exp(sc - mx);
        for (std::size_t d = 0; d < head_dim; ++d) {
            att[hh * head_dim + d] = static_cast<float>(w1 * double(v[kh * head_dim + d]));
        }
    }
    std::vector<float> t0(hidden), h1(hidden);
    cpu_matvec(o_w, att, t0, hidden, q_width);
    for (std::size_t i = 0; i < hidden; ++i) {
        h1[i] = static_cast<float>(double(t0[i]) + double(x[i]));
    }
    std::vector<float> h2(hidden), gv(ffn_rows), uv(ffn_rows), fv(ffn_rows), dv(hidden), out_ref(hidden);
    cpu_rmsnorm(h1, ffn_norm_w, h2, epsilon);
    cpu_matvec(g_w, h2, gv, ffn_rows, hidden);
    cpu_matvec(u_w, h2, uv, ffn_rows, hidden);
    for (std::size_t i = 0; i < ffn_rows; ++i) {
        const double g = double(gv[i]);
        fv[i] = static_cast<float>(g / (1.0 + std::exp(-g)) * double(uv[i]));
    }
    cpu_matvec(d_w, fv, dv, hidden, ffn_rows);
    for (std::size_t i = 0; i < hidden; ++i) {
        out_ref[i] = static_cast<float>(double(h1[i]) + double(dv[i]));
    }

    auto run = [&](bool dp4a) {
        _putenv(dp4a ? "ZEN_D3D12_DP4A=1" : "ZEN_D3D12_DP4A=0");
        if (!zen::d3d12::kv_cache_configure(1, context, kv_heads, head_dim)) {
            std::printf("kv_cache_configure failed\n");
            return 1e9;
        }
        std::vector<float> out(hidden, 0.0f);
        zen::d3d12::LayerDecodeParams params;
        params.q_w = q_w.data();
        params.k_w = k_w.data();
        params.v_w = v_w.data();
        params.o_w = o_w.data();
        params.gate_w = g_w.data();
        params.up_w = u_w.data();
        params.down_w = d_w.data();
        params.attn_norm_w = attn_norm_w.data();
        params.ffn_norm_w = ffn_norm_w.data();
        params.cos = cos_tab.data();
        params.sin = sin_tab.data();
        params.pairs = pairs;
        params.layer = 0;
        params.pos = pos;
        params.first_pos = 0;
        params.rope_pairs_adjacent = 0;
        const bool ok = zen::d3d12::layer_decode_pipeline(&params, 1, x.data(), out.data(),
                                                          hidden, heads, kv_heads, head_dim,
                                                          ffn_rows, epsilon, nullptr);
        if (!ok) {
            std::printf("layer_decode_pipeline returned false (dp4a=%d)\n", int(dp4a));
            zen::d3d12::trace_report();
            return 1e9;
        }
        double mae = 0.0, mref = 0.0;
        for (std::size_t i = 0; i < hidden; ++i) {
            mae = std::max(mae, std::fabs(double(out[i]) - double(out_ref[i])));
            mref = std::max(mref, std::fabs(double(out_ref[i])));
        }
        std::printf("dp4a=%d max_abs=%.4e max_ref=%.4e rel=%.4e (worst idx rel)\n", int(dp4a), mae,
                    mref, mae / (mref + 1e-30));
        return mae / (mref + 1e-30);
    };

    const double r0 = run(false);
    const double r1 = run(true);
    // dp4a=0 must be at fp32 parity (~1e-6); dp4a=1 is approximate by design —
    // whole-layer cumulative rel error on the synthetic rig's wide dynamic
    // range sits at ~1.5e-2 (per-matvec ~4-8e-3), which still catches hard breakage
    // (a broken matvec gives ~1).
    const bool pass = (r0 <= 1e-4) && (r1 <= 2e-2);
    std::printf("%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
