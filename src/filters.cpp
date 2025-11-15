#include "pixfoundry/filters.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace pf {

// ------------------------------------------------------------
// 內部工具：索引邊界處理
// ------------------------------------------------------------
static inline int border_index(int i, int N, Border border) {
    if (N <= 0) return 0;

    switch (border) {
        case Border::Reflect: {
            if (N == 1) return 0;
            if (i < 0)      return -i - 1;
            if (i >= N)     return 2 * N - i - 1;
            return i;
        }
        case Border::Replicate: {
            if (i < 0)      return 0;
            if (i >= N)     return N - 1;
            return i;
        }
        case Border::Wrap: {
            int m = i % N;
            if (m < 0) m += N;
            return m;
        }
        default:
            // Constant 不應走到這裡，直接 clamp
            return std::clamp(i, 0, N - 1);
    }
}

// Constant 之外的 sample
static inline uint8_t sample_u8(const ImageU8& src, int y, int x, int c,
                                Border border, uint8_t border_value) 
{
    const int H = src.h(), W = src.w(), C = src.c();

    // Constant 直接判斷
    if (border == Border::Constant) {
        if (y < 0 || y >= H || x < 0 || x >= W) return border_value;
    }

    // 其他模式：套 border_index
    int yy = border_index(y, H, border);
    int xx = border_index(x, W, border);
    return src.data()[((yy * W + xx) * C) + c];
}

// ------------------------------------------------------------
// Kernel 工具
// ------------------------------------------------------------
std::vector<float> box_kernel1d(int ksize) {
    if (ksize < 3 || (ksize % 2 == 0)) {
        throw std::invalid_argument("box_kernel1d: ksize must be odd >= 3");
    }
    return std::vector<float>(ksize, 1.0f / ksize);
}

std::vector<float> gaussian_kernel1d(float sigma) {
    if (!(sigma > 0.f)) {
        throw std::invalid_argument("gaussian_kernel1d: sigma must be > 0");
    }

    int k = std::max(3, (int(std::ceil(6.f * sigma)) | 1));  // 強制 odd
    int R = k / 2;

    std::vector<float> kernel(k);
    float inv2s2 = 1.f / (2.f * sigma * sigma);
    float sum = 0.f;

    for (int i = -R; i <= R; ++i) {
        float v = std::exp(-(i * i) * inv2s2);
        kernel[i + R] = v;
        sum += v;
    }

    for (float& v : kernel) v /= sum;
    return kernel;
}

// ------------------------------------------------------------
//  Separable Convolution
// ------------------------------------------------------------
static ImageU8 convolve_separable_u8(const ImageU8& src,
                                     const std::vector<float>& k1d,
                                     Border border,
                                     uint8_t border_value)
{
    if (src.empty()) {
        throw std::invalid_argument("convolve: src empty");
    }

    const int H = src.h(), W = src.w(), C = src.c();
    const int K = (int)k1d.size();
    const int R = K / 2;

    std::vector<float> tmp((size_t)H * W * C, 0.f);
    ImageU8 dst(H, W, C);

    // ---- 水平 pass: src → tmp ----
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            for (int c = 0; c < C; ++c) {

                float sum = 0.f;
                for (int t = -R; t <= R; ++t) {
                    uint8_t v = sample_u8(src, y, x + t, c, border, border_value);
                    sum += k1d[t + R] * v;
                }

                tmp[((y * W + x) * C) + c] = sum;
            }
        }
    }

    // ---- 垂直 pass: tmp → dst ----
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            for (int c = 0; c < C; ++c) {

                float sum = 0.f;

                for (int t = -R; t <= R; ++t) {
                    int yy = y + t;
                    float v;

                    if (border == Border::Constant) {
                        if (yy < 0 || yy >= H) {
                            v = border_value;
                        } else {
                            v = tmp[((yy * W + x) * C) + c];
                        }
                    } else {
                        yy = border_index(yy, H, border);
                        v = tmp[((yy * W + x) * C) + c];
                    }

                    sum += k1d[t + R] * v;
                }

                float out = std::round(sum);
                out = std::clamp(out, 0.f, 255.f);
                dst.data()[((y * W + x) * C) + c] = (uint8_t)out;
            }
        }
    }

    return dst;
}

// ------------------------------------------------------------
//  Public API
// ------------------------------------------------------------
ImageU8 mean_filter(const ImageU8& src, int ksize,
                    Border border, Backend, uint8_t border_value)
{
    auto kernel = box_kernel1d(ksize);
    return convolve_separable_u8(src, kernel, border, border_value);
}

ImageU8 gaussian_filter(const ImageU8& src, float sigma,
                        Border border, Backend, uint8_t border_value)
{
    auto kernel = gaussian_kernel1d(sigma);
    return convolve_separable_u8(src, kernel, border, border_value);
}

} // namespace pf
