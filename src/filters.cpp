#include "pixfoundry/filters.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace pf {

// ============================================================
// 小工具：index 計算 & 邊界處理
// ============================================================

inline std::size_t linear_index(int y, int x, int c, int W, int C) {
    return (static_cast<std::size_t>(y) * W + x) * C + c;
}

// 將越界的 index 依照 Border 規則映射回合法範圍 [0, N-1]
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
        case Border::Constant:
        default:
            // Constant 不應走到這裡，這裡給個 clamp，實際上 Constant 會在 sample 裡處理
            return std::clamp(i, 0, N - 1);
    }
}

// 從 uint8 影像取樣（支援 Constant / Reflect / Replicate / Wrap）
static inline uint8_t sample_u8(const ImageU8& src,
                                int y, int x, int c,
                                Border border,
                                uint8_t border_value) {
    const int H = src.h(), W = src.w(), C = src.c();

    // Constant：只要越界就直接給 border_value
    if (border == Border::Constant) {
        if (y < 0 || y >= H || x < 0 || x >= W) {
            return border_value;
        }
    }

    // 其他模式：先用 border_index 做映射
    int yy = border_index(y, H, border);
    int xx = border_index(x, W, border);
    return src.data()[linear_index(yy, xx, c, W, C)];
}

// 從 float buffer（例如暫存的 tmp）取樣
static inline float sample_float(const std::vector<float>& buf,
                                 int y, int x, int c,
                                 int H, int W, int C,
                                 Border border,
                                 uint8_t border_value) {
    // Constant：越界直接回傳 border_value（轉成 float）
    if (border == Border::Constant) {
        if (y < 0 || y >= H || x < 0 || x >= W) {
            return static_cast<float>(border_value);
        }
    }

    int yy = border_index(y, H, border);
    int xx = border_index(x, W, border);
    return buf[linear_index(yy, xx, c, W, C)];
}

// ============================================================
// Kernel 工具
// ============================================================

std::vector<float> box_kernel1d(int ksize) {
    if (ksize < 3 || (ksize % 2 == 0)) {
        throw std::invalid_argument("box_kernel1d: ksize must be odd and >= 3");
    }

    std::vector<float> kernel(ksize, 1.0f / static_cast<float>(ksize));
    return kernel;
}

std::vector<float> gaussian_kernel1d(float sigma) {
    if (!(sigma > 0.f)) {
        throw std::invalid_argument("gaussian_kernel1d: sigma must be > 0");
    }

    // kernel 長度約為 6*sigma，並強制為奇數
    int k = std::max(3, (static_cast<int>(std::ceil(6.f * sigma)) | 1));
    int R = k / 2;

    std::vector<float> kernel(k);
    float inv2s2 = 1.f / (2.f * sigma * sigma);
    float sum = 0.f;

    for (int i = -R; i <= R; ++i) {
        float v = std::exp(-static_cast<float>(i * i) * inv2s2);
        kernel[i + R] = v;
        sum += v;
    }

    // 正規化使 sum = 1
    for (float& v : kernel) v /= sum;
    return kernel;
}

// ============================================================
// 可重用的 Separable Convolution（uint8 in / uint8 out）
// ============================================================

static ImageU8 convolve_separable_u8(const ImageU8& src,
                                     const std::vector<float>& k1d,
                                     Border border,
                                     uint8_t border_value) {
    if (src.empty()) {
        throw std::invalid_argument("convolve_separable_u8: src empty");
    }
    if (k1d.empty()) {
        throw std::invalid_argument("convolve_separable_u8: kernel empty");
    }

    const int H = src.h(), W = src.w(), C = src.c();
    const int K = static_cast<int>(k1d.size());
    const int R = K / 2;

    std::vector<float> tmp(static_cast<std::size_t>(H) * W * C, 0.f);
    ImageU8 dst(H, W, C);

    // ---- 水平 pass: src → tmp ----
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            for (int c = 0; c < C; ++c) {
                float sum = 0.f;
                for (int t = -R; t <= R; ++t) {
                    uint8_t v = sample_u8(src, y, x + t, c, border, border_value);
                    sum += k1d[t + R] * static_cast<float>(v);
                }
                tmp[linear_index(y, x, c, W, C)] = sum;
            }
        }
    }

    // ---- 垂直 pass: tmp → dst ----
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            for (int c = 0; c < C; ++c) {
                float sum = 0.f;
                for (int t = -R; t <= R; ++t) {
                    float v = sample_float(tmp, y + t, x, c, H, W, C, border, border_value);
                    sum += k1d[t + R] * v;
                }

                float out = std::round(sum);
                out = std::clamp(out, 0.f, 255.f);
                dst.data()[linear_index(y, x, c, W, C)] = static_cast<uint8_t>(out);
            }
        }
    }

    return dst;
}

// ============================================================
// Public API
// ============================================================

ImageU8 mean_filter(const ImageU8& src,
                    int ksize,
                    Border border,
                    Backend /*backend*/,
                    uint8_t border_value) {
    auto kernel = box_kernel1d(ksize);
    return convolve_separable_u8(src, kernel, border, border_value);
}

ImageU8 gaussian_filter(const ImageU8& src,
                        float sigma,
                        Border border,
                        Backend /*backend*/,
                        uint8_t border_value) {
    auto kernel = gaussian_kernel1d(sigma);
    return convolve_separable_u8(src, kernel, border, border_value);
}

ImageU8 median_filter(const ImageU8& src,
                      int ksize,
                      Border border,
                      Backend /*backend*/,
                      uint8_t border_value) {
    if (src.empty()) {
        throw std::invalid_argument("median_filter: src empty");
    }
    if (ksize < 3 || (ksize % 2 == 0)) {
        throw std::invalid_argument("median_filter: ksize must be odd and >= 3");
    }

    const int H = src.h(), W = src.w(), C = src.c();
    const int R = ksize / 2;

    ImageU8 dst(H, W, C);
    const int window_size = ksize * ksize;
    std::vector<uint8_t> window;
    window.reserve(static_cast<std::size_t>(window_size));

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            for (int c = 0; c < C; ++c) {

                window.clear();
                for (int dy = -R; dy <= R; ++dy) {
                    for (int dx = -R; dx <= R; ++dx) {
                        uint8_t v = sample_u8(src, y + dy, x + dx, c, border, border_value);
                        window.push_back(v);
                    }
                }

                // 取中位數：使用 nth_element, O(k^2)
                auto mid_it = window.begin() + window.size() / 2;
                std::nth_element(window.begin(), mid_it, window.end());
                uint8_t med = *mid_it;

                dst.data()[linear_index(y, x, c, W, C)] = med;
            }
        }
    }

    return dst;
}

ImageU8 bilateral_filter(const ImageU8& src,
                         int ksize,
                         float sigma_color,
                         float sigma_space,
                         Border border,
                         Backend /*backend*/,
                         uint8_t border_value) {
    if (src.empty()) {
        throw std::invalid_argument("bilateral_filter: src empty");
    }
    if (ksize < 3 || (ksize % 2 == 0)) {
        throw std::invalid_argument("bilateral_filter: ksize must be odd and >= 3");
    }
    if (!(sigma_color > 0.f) || !(sigma_space > 0.f)) {
        throw std::invalid_argument("bilateral_filter: sigma_color and sigma_space must be > 0");
    }

    const int H = src.h(), W = src.w(), C = src.c();
    const int R = ksize / 2;

    const float inv2_sigma_space2 = 1.0f / (2.0f * sigma_space * sigma_space);
    const float inv2_sigma_color2 = 1.0f / (2.0f * sigma_color * sigma_color);

    ImageU8 dst(H, W, C);

    // 預先算好空間權重（跟顏色無關，只跟 dx, dy 有關）
    std::vector<float> spatial_weight(static_cast<std::size_t>(ksize) * ksize);
    for (int dy = -R; dy <= R; ++dy) {
        for (int dx = -R; dx <= R; ++dx) {
            float dsq = static_cast<float>(dx * dx + dy * dy);
            float w = std::exp(-dsq * inv2_sigma_space2);
            spatial_weight[(dy + R) * ksize + (dx + R)] = w;
        }
    }

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            for (int c = 0; c < C; ++c) {

                uint8_t center_u8 = sample_u8(src, y, x, c, border, border_value);
                float center = static_cast<float>(center_u8);

                float norm = 0.f;
                float acc  = 0.f;

                for (int dy = -R; dy <= R; ++dy) {
                    for (int dx = -R; dx <= R; ++dx) {
                        uint8_t neigh_u8 = sample_u8(src, y + dy, x + dx, c, border, border_value);
                        float neigh = static_cast<float>(neigh_u8);

                        float dsq = static_cast<float>(dx * dx + dy * dy);
                        float spatial = spatial_weight[(dy + R) * ksize + (dx + R)];

                        float diff = neigh - center;
                        float range = std::exp(-(diff * diff) * inv2_sigma_color2);

                        float w = spatial * range;
                        norm += w;
                        acc  += w * neigh;
                    }
                }

                float out = (norm > 0.f) ? (acc / norm) : center;
                out = std::clamp(std::round(out), 0.f, 255.f);
                dst.data()[linear_index(y, x, c, W, C)] = static_cast<uint8_t>(out);
            }
        }
    }

    return dst;
}

} // namespace pf
