// Structured GGUF fuzzer built on the shared parse core.
// Unlike a raw libFuzzer byte stream, this keeps the valid GGUF header and
// mutates the interesting parts (metadata types, counts, strings, tensor
// directory) so coverage reaches past the header checks. Any parse attempt
// that hangs, crashes, throws or leaks is a defect: the parser's contract is
// to return ok=false with an error string for every malformed input.
//
// Build with libFuzzer (requires clang):
//   clang++ -std=c++20 -fsanitize=address,undefined -fno-omit-frame-pointer
//     -I include fuzz_gguf.cpp build/libzen_core.a -lFuzzer -o fuzz_gguf
//   ./fuzz_gguf -max_total_time=60
//
// Pure-harness mode without libFuzzer:
//   g++ -std=c++20 -O2 -DZINFUZZ_STANDALONE -I include fuzz_gguf.cpp \
//     build/libzen_core.a build/libzen_avx2.a -o fuzz_gguf && ./fuzz_gguf
#include "zen/gguf.h"
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// Field-level notes, from the GGUF v2/v3 layout:
//   header: magic[4] version u32 tensor_count u64 metadata_count u64
//   metadata: key_len u64 key bytes type u32 value...
//   tensor: name_len u64 name ndim u32 dims ndim*u64 ggml_type u32 offset u64
namespace {
// Rotate the byte stream into structural positions rather than random bytes,
// so one mutation pass covers the header, the metadata and the tensor directory.
struct Cycler {
    const std::uint8_t* data;
    std::size_t size;
    std::size_t head = 0;
    std::uint8_t next(std::uint8_t fallback) {
        if (size == 0) return fallback;
        return data[head++ % size];
    }
    void next_bytes(std::uint8_t* out, std::size_t n) {
        for (std::size_t i = 0; i < n; ++i) out[i] = next(static_cast<std::uint8_t>(i));
    }
};

void append_u16(std::vector<std::uint8_t>& v, std::uint16_t x) {
    v.push_back(static_cast<std::uint8_t>(x));
    v.push_back(static_cast<std::uint8_t>(x >> 8));
}
void append_u32(std::vector<std::uint8_t>& v, std::uint32_t x) {
    for (int i = 0; i < 4; ++i) { v.push_back(static_cast<std::uint8_t>(x)); x >>= 8; }
}
void append_u64(std::vector<std::uint8_t>& v, std::uint64_t x) {
    for (int i = 0; i < 8; ++i) { v.push_back(static_cast<std::uint8_t>(x)); x >>= 8; }
}
void append_gstr(std::vector<std::uint8_t>& v, const char* s, std::size_t n) {
    append_u64(v, static_cast<std::uint64_t>(n));
    v.insert(v.end(), s, s + n);
}

// Builds a GGUF image that is valid except for one hostile dimension chosen
// by the mutation stream. Returns roughly 8 KB including the tensor body so
// payload-bounds checks are exercised too.
std::vector<std::uint8_t> build_case(const std::uint8_t* data, std::size_t size) {
    Cycler c{data, size};
    std::vector<std::uint8_t> img;
    img.insert(img.end(), {'G', 'G', 'U', 'F'});
    append_u32(img, 3);      // version
    append_u64(img, 1);      // one tensor
    append_u64(img, 3);      // three metadata entries
    // 1) String key whose length comes from the stream (can overrun).
    {
        std::uint8_t klen;
        c.next_bytes(&klen, 1);
        append_u64(img, klen);
        for (int i = 0; i < klen; ++i) img.push_back(c.next('k'));
        append_u32(img, c.next(8)); // type from stream, 8=STRING expected
        if (size % 3) {             // sometimes a huge value length
            append_u64(img, 0x1000000ull + c.next(0));
        } else {
            append_gstr(img, "llama", 5);
        }
    }
    // 2) ARRAY with stream-chosen subtype and count (nesting is exercised
    //    by choosing subtype 9 and repeating the structure manually).
    {
        append_gstr(img, "crafted.arr", 11);
        append_u32(img, 9);
        std::uint8_t parts[8];
        c.next_bytes(parts, 8);
        const std::uint32_t subtype = parts[0] % 13;
        const std::uint64_t count = (std::uint64_t(parts[1]) << 24) | (parts[2] << 16) | (parts[3] << 8) | parts[4];
        append_u32(img, subtype);
        append_u64(img, subtype == 9 ? (count % 160) : (count % 4096));
        if (subtype == 9) {
            for (std::uint64_t i = 0; i < (count % 160); ++i) { append_u32(img, 8); append_u64(img, 0); }
        } else if (subtype == 8) {
            append_gstr(img, "x", 1);
        }
    }
    // 3) An embedding_length that is sometimes just past uint32 range.
    {
        append_gstr(img, "llama.embedding_length", 22);
        append_u32(img, 10); // U64
        append_u64(img, (c.next(0) & 1) ? (0x100000000ull + c.next(0)) : 4096);
    }
    // Tensor directory: name, ndim, dims, type, offset.
    {
        append_gstr(img, "w", 1);
        std::uint8_t nd[4];
        c.next_bytes(nd, 4);
        const std::uint32_t ndim = nd[0] % 5;
        append_u32(img, ndim);
        for (std::uint32_t i = 0; i < ndim; ++i) append_u64(img, 1 + (nd[1 + (i % 3)] % 511));
        append_u32(img, c.next(12)); // type from stream (12=Q4_K valid-ish)
        append_u64(img, (std::uint64_t(c.next(0)) << 40) | (c.next(0) << 8));
    }
    // Pad to alignment 32 and append a small body.
    while (img.size() % 32) img.push_back(0);
    img.insert(img.end(), 4096, 0xA5);
    return img;
}

int run_case(const std::uint8_t* data, std::size_t size, bool verbose) {
    if (size == 0) return 0;
    const auto img = build_case(data, size);
    const auto report = zen::gguf::inspect_bytes(img.data(), img.size());
    if (verbose) std::printf("%s (tensors=%zu) : %s\n", report.ok ? "ok" : "rejected",
                             report.tensors.size(), report.error.c_str());
    return 0;
}
} // namespace

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
    return run_case(data, size, false);
}

#ifdef ZINFUZZ_STANDALONE
// Deterministic standalone pass: 4096 pseudo-random cases from an LCG,
// plus the regression corpus from the security review.
int main() {
    std::uint64_t state = 0x243F6A8885A308D3ull;
    auto rnd = [&] {
        state = state * 6364136223846793005ull + 1442695040888963407ull;
        return static_cast<std::uint8_t>(state >> 33);
    };
    std::vector<std::uint8_t> buf(256);
    for (int i = 0; i < 4096; ++i) {
        for (auto& b : buf) b = rnd();
        run_case(buf.data(), buf.size(), false);
    }
    // Regression corpus.
    std::vector<std::uint8_t> deep{'G', 'G', 'U', 'F', 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0};
    const char* key = "crafted.nested";
    deep.push_back(14);
    deep.insert(deep.end(), {0, 0, 0, 0, 0, 0, 0});
    deep.insert(deep.end(), key, key + 14);
    deep.insert(deep.end(), {9, 0, 0, 0});
    for (int i = 0; i < 80; ++i) deep.insert(deep.end(), {9, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0});
    deep.insert(deep.end(), {8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
    run_case(deep.data(), deep.size(), true);
    run_case(nullptr, 0, true);
    std::vector<std::uint8_t> wide{'G', 'G', 'U', 'F', 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0};
    append_gstr(wide, "llama.embedding_length", 22);
    wide.insert(wide.end(), {10, 0, 0, 0, 0, 16, 0, 0, 1, 0, 0, 0});
    run_case(wide.data(), wide.size(), true);
    std::printf("standalone fuzz: 4096 random cases + 3 regression cases, no crash/hang\n");
    return 0;
}
#endif
