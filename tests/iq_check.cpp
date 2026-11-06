// Cross-check the GGML Q8_K / TQ / IQ / PTQ decoders and CPU matrix paths
// against the independent Python reference produced by iq_reference.py.
#include "zen/quant.h"
#include "zen/kernels.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {
struct Record {
    std::string name;
    std::vector<std::uint8_t> block;
    std::vector<float> expected;
};

float activation(std::size_t i) { return float(int((i * 37 + 11) % 101) - 50) * 0.0137f; }
} // namespace

int main(int argc, char** argv) {
    if (argc < 2) { std::printf("usage: iq_check file.bin\n"); return 2; }
    std::ifstream in(argv[1], std::ios::binary);
    if (!in) { std::printf("cannot open %s\n", argv[1]); return 2; }
    std::uint32_t count = 0;
    in.read(reinterpret_cast<char*>(&count), 4);
    std::vector<Record> records;
    for (std::uint32_t i = 0; i < count; ++i) {
        char name[8]{};
        std::uint32_t size = 0;
        in.read(name, 8);
        in.read(reinterpret_cast<char*>(&size), 4);
        Record rec;
        rec.name = name;
        rec.block.resize(size);
        in.read(reinterpret_cast<char*>(rec.block.data()), size);
        rec.expected.resize(256);
        in.read(reinterpret_cast<char*>(rec.expected.data()), static_cast<std::streamsize>(256) * 4);
        records.push_back(std::move(rec));
    }

    int failures = 0, checks = 0;
    double worst_rel = 0.0;

    // 1) Decoder output against the independent reference.
    for (const auto& rec : records) {
        const auto* fmt = zen::quant::find(rec.name);
        if (!fmt || !fmt->implemented) { std::printf("missing format %s\n", rec.name.c_str()); ++failures; continue; }
        std::vector<float> actual(256);
        if (!zen::quant::dequantize(*fmt, rec.block.data(), 256, actual.data())) {
            std::printf("%s: decode failed\n", rec.name.c_str()); ++failures; continue;
        }
        float scale = 0.0f;
        for (float v : rec.expected) scale = std::max(scale, std::abs(v));
        if (scale <= 0.0f) scale = 1.0f;
        ++checks;
        for (std::size_t k = 0; k < 256; ++k) {
            const double rel = std::abs(double(actual[k]) - double(rec.expected[k])) / scale;
            if (rel > worst_rel) worst_rel = rel;
            if (rel > 1e-5) {
                std::printf("%s elem %zu: actual=%g expected=%g rel=%g\n",
                            rec.name.c_str(), k, actual[k], rec.expected[k], rel);
                ++failures;
                break;
            }
        }
    }

    // 2) Generic GEMV against a dot product of the reference values, serial
    //    and parallel dispatch.
    for (const auto& fmt_name : std::vector<std::string>{"Q8_K", "TQ1_0", "TQ2_0", "IQ4_XS",
                                                         "IQ2_XXS", "IQ2_XS", "IQ2_S",
                                                         "IQ3_XXS", "IQ3_S", "IQ1_S", "IQ1_M", "IQ4_NL", "PTQ1_0"}) {
        std::vector<const Record*> pool;
        for (const auto& rec : records) if (rec.name == fmt_name) pool.push_back(&rec);
        if (pool.empty()) { std::printf("%s: no reference records\n", fmt_name.c_str()); ++failures; continue; }
        const auto* fmt = zen::quant::find(fmt_name);
        const std::size_t rows = 8, columns = 256;
        const std::size_t row_bytes = zen::quant::row_bytes(*fmt, columns);
        std::vector<std::uint8_t> matrix(rows * row_bytes);
        std::vector<float> input(columns), output(rows), expected(rows);
        for (std::size_t i = 0; i < columns; ++i) input[i] = activation(i);
        for (std::size_t r = 0; r < rows; ++r) {
            const Record& rec = *pool[r % pool.size()];
            std::memcpy(matrix.data() + r * row_bytes, rec.block.data(), rec.block.size());
            double sum = 0.0;
            for (std::size_t k = 0; k < columns; ++k) sum += double(rec.expected[k]) * input[k];
            expected[r] = float(sum);
        }
        for (unsigned workers : {1u, 4u}) {
            if (!zen::quant::gemv(*fmt, matrix.data(), input.data(), output.data(), rows, columns, 1.0f, workers)) {
                std::printf("%s: gemv failed (workers=%u)\n", fmt_name.c_str(), workers); ++failures; continue;
            }
            ++checks;
            for (std::size_t r = 0; r < rows; ++r) {
                // Tolerance scaled by the accumulation magnitude, not the
                // (possibly cancelling) result: 256 float32 products summed
                // in float32 carry ~1e-7 relative error per term.
                double magnitude = 0.0;
                for (std::size_t k = 0; k < columns; ++k)
                    magnitude += std::abs(double(pool[r % pool.size()]->expected[k]) * input[k]);
                const double tol = 1e-5 * std::max(1.0, magnitude);
                const double diff = std::abs(double(output[r]) - double(expected[r]));
                if (diff > tol) {
                    std::printf("%s row %zu workers=%u: actual=%g expected=%g diff=%g tol=%g\n",
                                fmt_name.c_str(), r, workers, output[r], expected[r], diff, tol);
                    ++failures;
                }
            }
        }
    }

    // Independent decoded reference with multiple 256-element records per row,
    // unaligned weights, matrix/panel tails, varied token counts and workers.
    for(const auto& fmt_name:std::vector<std::string>{"IQ4_NL","IQ4_XS","IQ2_XXS","IQ2_XS","IQ2_S","IQ3_XXS","IQ3_S","IQ1_S","IQ1_M","PTQ1_0"}){
        std::vector<const Record*> pool;
        for(const auto& rec:records)if(rec.name==fmt_name)pool.push_back(&rec);
        const auto* fmt=zen::quant::find(fmt_name);
        if(!fmt||pool.empty()){++failures;continue;}
        for(std::size_t columns:{std::size_t(256),std::size_t(768),std::size_t(fmt->block_elements),std::size_t(fmt->block_elements)*3})for(std::size_t rows:{1u,5u,67u}){
            const auto rb=zen::quant::row_bytes(*fmt,columns);
            std::vector<std::uint8_t> matrix(rows*rb+1);
            std::vector<float> x(8*columns), output(8*rows), ref(rows*columns);
            for(std::size_t c=0;c<x.size();++c)x[c]=activation(c);
            for(std::size_t r=0;r<rows;++r)for(std::size_t pos=0;pos<columns;pos+=256){
                const auto& rec=*pool[(r*3+pos/256)%pool.size()];
                const auto chunk=std::min<std::size_t>(256,columns-pos);
                std::memcpy(matrix.data()+1+r*rb+zen::quant::row_bytes(*fmt,pos),rec.block.data(),zen::quant::row_bytes(*fmt,chunk));
                std::copy_n(rec.expected.begin(),chunk,ref.begin()+r*columns+pos);
            }
            for(std::size_t tokens:{1u,2u,3u,8u})for(unsigned workers:{1u,4u}){
                const bool ok=tokens==1?zen::quant::gemv(*fmt,matrix.data()+1,x.data(),output.data(),rows,columns,1,workers):
                    zen::quant::gemm(*fmt,matrix.data()+1,x.data(),output.data(),tokens,rows,columns,1,workers);
                ++checks;if(!ok){std::printf("%s matrix dispatch failed\n",fmt_name.c_str());++failures;continue;}
                for(std::size_t t=0;t<tokens;++t)for(std::size_t r=0;r<rows;++r){
                    double sum=0,mag=0;
                    for(std::size_t c=0;c<columns;++c){const double v=double(ref[r*columns+c])*x[t*columns+c];sum+=v;mag+=std::abs(v);}
                    if(!std::isfinite(output[t*rows+r])||std::abs(double(output[t*rows+r])-sum)>2e-5*std::max(1.0,mag)){
                        std::printf("%s matrix mismatch rows=%zu cols=%zu tokens=%zu workers=%u\n",fmt_name.c_str(),rows,columns,tokens,workers);
                        ++failures;goto matrix_failed;
                    }
                }
                matrix_failed:;
            }
#ifdef ZEN_HAS_AVX2_KERNELS
            if(zen::cpu_has_avx2_fma()){
                using Fn=bool(*)(const std::uint8_t*,const float*,float*,std::size_t,std::size_t,std::size_t,std::size_t);
                using E=zen::quant::Encoding;
                Fn fn=fmt->encoding==E::ptq1_0?zen::gemm_ptq1_0_kblk_avx2:
                    fmt->encoding==E::iq4_nl?zen::gemm_iq4_nl_kblk_avx2:
                    fmt->encoding==E::iq4_xs?zen::gemm_iq4_xs_kblk_avx2:
                    fmt->encoding==E::iq2_xxs?zen::gemm_iq2_xxs_kblk_avx2:
                    fmt->encoding==E::iq2_xs?zen::gemm_iq2_xs_kblk_avx2:
                    fmt->encoding==E::iq2_s?zen::gemm_iq2_s_kblk_avx2:
                    fmt->encoding==E::iq3_xxs?zen::gemm_iq3_xxs_kblk_avx2:
                    fmt->encoding==E::iq3_s?zen::gemm_iq3_s_kblk_avx2:
                    fmt->encoding==E::iq1_s?zen::gemm_iq1_s_kblk_avx2:
                    fmt->encoding==E::iq1_m?zen::gemm_iq1_m_kblk_avx2:nullptr;
                const auto stride=rows+3;std::vector<float> padded(3*stride,-777.f);
                if(!fn||!fn(matrix.data()+1,x.data(),padded.data(),3,rows,columns,stride)){++failures;}
                else for(std::size_t t=0;t<3;++t){
                    for(std::size_t r=0;r<rows;++r){
                        double sum=0,mag=0;for(std::size_t c=0;c<columns;++c){const double v=double(ref[r*columns+c])*x[t*columns+c];sum+=v;mag+=std::abs(v);}
                        if(!std::isfinite(padded[t*stride+r])||std::abs(double(padded[t*stride+r])-sum)>2e-5*std::max(1.0,mag)){std::printf("%s padded mismatch\n",fmt_name.c_str());++failures;}
                    }
                    for(std::size_t r=rows;r<stride;++r)if(padded[t*stride+r]!=-777.f){std::printf("%s wrote padding\n",fmt_name.c_str());++failures;}
                }
            }
#endif
        }
    }

    std::printf("iq_check: %d checks, %d failures, worst relative decoder diff %.3g\n",
                checks, failures, worst_rel);
    return failures ? 1 : 0;
}
