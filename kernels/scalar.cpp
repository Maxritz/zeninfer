#include "zen/kernels.h"

namespace zen {
void gemv_scalar(const float* a, const float* x, float* y,
                 std::size_t m, std::size_t k) {
    for (std::size_t i=0; i<m; ++i) {
        const float* row = a + i*k;
        float sum = 0.0f;
        for (std::size_t p=0; p<k; ++p) sum += row[p] * x[p];
        y[i] = sum;
    }
}

void gemm_scalar(const float* a, const float* b, float* c,
                 std::size_t m, std::size_t n, std::size_t k) {
    for (std::size_t i=0; i<m; ++i) {
        for (std::size_t j=0; j<n; ++j) {
            float sum = 0.0f;
            for (std::size_t p=0; p<k; ++p)
                sum += a[i*k+p] * b[p*n+j];
            c[i*n+j] = sum;
        }
    }
}
}
