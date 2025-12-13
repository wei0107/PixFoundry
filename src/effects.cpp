#include "pixfoundry/effects.hpp"
#include "pixfoundry/color.hpp"
#include "pixfoundry/filters.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace pf {

static inline std::size_t idx(int y, int x, int c,
                              int W, int C) {
    return (static_cast<std::size_t>(y) * W + x) * C + c;
}

// =========================
//   single-thread 版本
// =========================

static ImageU8 sharpen_single(const ImageU8& src, float amount) {
    if (src.empty()) {
        throw std::invalid_argument("sharpen: empty image");
    }
    if (amount <= 0.0f) {
        // amount <= 0 → 直接回傳 copy
        const int H = src.h();
        const int W = src.w();
        const int C = src.c();
        ImageU8 dst(H, W, C);
        const std::size_t total = static_cast<std::size_t>(H) * W * C;
        std::copy(src.data(), src.data() + total, dst.data());
        return dst;
    }

    const int H = src.h();
    const int W = src.w();
    const int C = src.c();

    const uint8_t* in = src.data();
    ImageU8 dst(H, W, C);
    uint8_t* out = dst.data();

    // 3x3 銳化 kernel：center * (1+4*amount) - 四周 * amount
    // 等價於 base kernel [[0,-1,0],[-1,5,-1],[0,-1,0]] 的一般化
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            for (int c = 0; c < C; ++c) {
                // 取中心與四個鄰居（簡單 handling：超出邊界用最近點 clamping）
                auto clamp_xy = [&](int yy, int xx) {
                    yy = std::clamp(yy, 0, H - 1);
                    xx = std::clamp(xx, 0, W - 1);
                    return in[idx(yy, xx, c, W, C)];
                };

                float center = static_cast<float>(clamp_xy(y, x));
                float up     = static_cast<float>(clamp_xy(y - 1, x));
                float down   = static_cast<float>(clamp_xy(y + 1, x));
                float left   = static_cast<float>(clamp_xy(y, x - 1));
                float right  = static_cast<float>(clamp_xy(y, x + 1));

                float v = (1.0f + 4.0f * amount) * center
                          - amount * (up + down + left + right);

                v = std::round(v);
                v = std::clamp(v, 0.0f, 255.0f);
                out[idx(y, x, c, W, C)] = static_cast<uint8_t>(v);
            }
        }
    }

    return dst;
}

static ImageU8 emboss_single(const ImageU8& src, float strength) {
    if (src.empty()) {
        throw std::invalid_argument("emboss: empty image");
    }

    const int H = src.h();
    const int W = src.w();
    const int C = src.c();

    const uint8_t* in = src.data();
    ImageU8 dst(H, W, C);
    uint8_t* out = dst.data();

    // 3x3 emboss kernel（左下到右上的斜向）
    // [-2 -1 0
    //  -1  1 1
    //   0  1 2] * strength
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            for (int c = 0; c < C; ++c) {

                auto clamp_xy = [&](int yy, int xx) {
                    yy = std::clamp(yy, 0, H - 1);
                    xx = std::clamp(xx, 0, W - 1);
                    return static_cast<float>(in[idx(yy, xx, c, W, C)]);
                };

                float v =
                    -2.f * clamp_xy(y - 1, x - 1) +
                    -1.f * clamp_xy(y - 1, x    ) +
                    0.f  * clamp_xy(y - 1, x + 1) +
                    -1.f * clamp_xy(y,     x - 1) +
                    1.f  * clamp_xy(y,     x    ) +
                    1.f  * clamp_xy(y,     x + 1) +
                    0.f  * clamp_xy(y + 1, x - 1) +
                    1.f  * clamp_xy(y + 1, x    ) +
                    2.f  * clamp_xy(y + 1, x + 1);

                v *= strength;

                // 加 128 讓結果落在中間亮度附近，避免太多負數
                v = v + 128.0f;
                v = std::round(v);
                v = std::clamp(v, 0.0f, 255.0f);

                out[idx(y, x, c, W, C)] = static_cast<uint8_t>(v);
            }
        }
    }

    return dst;
}

static ImageU8 cartoonize_single(const ImageU8& src,
                                 float sigma_space,
                                 uint8_t edge_threshold) {
    if (src.empty()) {
        throw std::invalid_argument("cartoonize: empty image");
    }

    const int H = src.h();
    const int W = src.w();
    const int C = src.c();

    // 1. 平滑：先用 Gaussian 讓色塊變得比較平滑
    ImageU8 smooth = gaussian_filter(
        src,
        sigma_space,
        Border::Reflect,
        Backend::Single  // 先固定 single
    );

    // 2. 邊緣偵測：用灰階 + Sobel
    ImageU8 gray = to_grayscale(src, Backend::Single);
    const uint8_t* g_in = gray.data();

    ImageU8 edge_mask(H, W, 1);
    uint8_t* e_out = edge_mask.data();

    auto g_idx = [&](int y, int x) {
        return static_cast<std::size_t>(y) * W + x;
    };

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {

            auto clamp_g = [&](int yy, int xx) -> float {
                yy = std::clamp(yy, 0, H - 1);
                xx = std::clamp(xx, 0, W - 1);
                return static_cast<float>(g_in[g_idx(yy, xx)]);
            };

            float gx =
                -1.f * clamp_g(y - 1, x - 1) +
                 0.f * clamp_g(y - 1, x    ) +
                 1.f * clamp_g(y - 1, x + 1) +
                -2.f * clamp_g(y,     x - 1) +
                 0.f * clamp_g(y,     x    ) +
                 2.f * clamp_g(y,     x + 1) +
                -1.f * clamp_g(y + 1, x - 1) +
                 0.f * clamp_g(y + 1, x    ) +
                 1.f * clamp_g(y + 1, x + 1);

            float gy =
                -1.f * clamp_g(y - 1, x - 1) +
                -2.f * clamp_g(y - 1, x    ) +
                -1.f * clamp_g(y - 1, x + 1) +
                 1.f * clamp_g(y + 1, x - 1) +
                 2.f * clamp_g(y + 1, x    ) +
                 1.f * clamp_g(y + 1, x + 1);

            float mag = std::fabs(gx) + std::fabs(gy);

            // 邊緣 → 0 (黑)，非邊緣 → 255 (白)
            e_out[g_idx(y, x)] =
                (mag > static_cast<float>(edge_threshold)) ? 0 : 255;
        }
    }

    // 3. 顏色量化：讓色塊看起來更「卡通」
    uint8_t* s_data = smooth.data();
    const std::size_t total = static_cast<std::size_t>(H) * W * C;
    const int levels = 16;                 // 16 階
    const float step = 255.0f / (levels - 1);

    for (std::size_t i = 0; i < total; ++i) {
        float v = static_cast<float>(s_data[i]);
        int idx_level = static_cast<int>(std::round(v / step));
        v = std::clamp(idx_level * step, 0.0f, 255.0f);
        s_data[i] = static_cast<uint8_t>(v);
    }

    // 4. 把邊緣畫成黑色線條疊在平滑過的色塊上
    ImageU8 out_img(H, W, C);
    uint8_t* out = out_img.data();

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            uint8_t e = e_out[g_idx(y, x)];
            for (int c = 0; c < C; ++c) {
                uint8_t base = s_data[idx(y, x, c, W, C)];
                // 邊緣像素 -> 黑色，否則用量化後的平滑顏色
                const uint8_t edge_color = 20; // 調整這個值來控制邊緣亮度
                out[idx(y, x, c, W, C)] = (e == 0) ? edge_color : base;
            }
        }
    }

    return out_img;
}

// =========================
//   OpenMP 版本（若 PF_HAS_OPENMP）
// =========================

static ImageU8 sharpen_openmp(const ImageU8& src, float amount) {
    if (src.empty()) throw std::invalid_argument("sharpen: empty image");

    if (amount <= 0.0f) {
        const int H = src.h(), W = src.w(), C = src.c();
        ImageU8 dst(H, W, C);
        const std::size_t total = static_cast<std::size_t>(H) * W * C;
        std::copy(src.data(), src.data() + total, dst.data());
        return dst;
    }

    const int H = src.h(), W = src.w(), C = src.c();
    const uint8_t* in = src.data();
    ImageU8 dst(H, W, C);
    uint8_t* out = dst.data();

#ifdef PF_HAS_OPENMP
#pragma omp parallel for collapse(2)
#endif
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            for (int c = 0; c < C; ++c) {
                auto clamp_xy = [&](int yy, int xx) {
                    yy = std::clamp(yy, 0, H - 1);
                    xx = std::clamp(xx, 0, W - 1);
                    return in[idx(yy, xx, c, W, C)];
                };

                float center = static_cast<float>(clamp_xy(y, x));
                float up     = static_cast<float>(clamp_xy(y - 1, x));
                float down   = static_cast<float>(clamp_xy(y + 1, x));
                float left   = static_cast<float>(clamp_xy(y, x - 1));
                float right  = static_cast<float>(clamp_xy(y, x + 1));

                float v = (1.0f + 4.0f * amount) * center
                        - amount * (up + down + left + right);

                v = std::round(v);
                v = std::clamp(v, 0.0f, 255.0f);
                out[idx(y, x, c, W, C)] = static_cast<uint8_t>(v);
            }
        }
    }
    return dst;
}

static ImageU8 emboss_openmp(const ImageU8& src, float strength) {
    if (src.empty()) throw std::invalid_argument("emboss: empty image");

    const int H = src.h(), W = src.w(), C = src.c();
    const uint8_t* in = src.data();
    ImageU8 dst(H, W, C);
    uint8_t* out = dst.data();

#ifdef PF_HAS_OPENMP
#pragma omp parallel for collapse(2)
#endif
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            for (int c = 0; c < C; ++c) {
                auto clamp_xy = [&](int yy, int xx) -> float {
                    yy = std::clamp(yy, 0, H - 1);
                    xx = std::clamp(xx, 0, W - 1);
                    return static_cast<float>(in[idx(yy, xx, c, W, C)]);
                };

                float v =
                    -2.f * clamp_xy(y - 1, x - 1) +
                    -1.f * clamp_xy(y - 1, x    ) +
                     0.f * clamp_xy(y - 1, x + 1) +
                    -1.f * clamp_xy(y,     x - 1) +
                     1.f * clamp_xy(y,     x    ) +
                     1.f * clamp_xy(y,     x + 1) +
                     0.f * clamp_xy(y + 1, x - 1) +
                     1.f * clamp_xy(y + 1, x    ) +
                     2.f * clamp_xy(y + 1, x + 1);

                v *= strength;
                v = v + 128.0f;
                v = std::round(v);
                v = std::clamp(v, 0.0f, 255.0f);

                out[idx(y, x, c, W, C)] = static_cast<uint8_t>(v);
            }
        }
    }
    return dst;
}

static ImageU8 cartoonize_openmp(const ImageU8& src,
                                 float sigma_space,
                                 uint8_t edge_threshold) {
    if (src.empty()) throw std::invalid_argument("cartoonize: empty image");

    const int H = src.h(), W = src.w(), C = src.c();

    // 1) 平滑：這裡不要硬鎖 single，讓它跟著 OpenMP 走
    ImageU8 smooth = gaussian_filter(
        src,
        sigma_space,
        Border::Reflect,
        Backend::OpenMP
    );

    // 2) 邊緣偵測：灰階 + Sobel
    ImageU8 gray = to_grayscale(src, Backend::OpenMP);
    const uint8_t* g_in = gray.data();

    ImageU8 edge_mask(H, W, 1);
    uint8_t* e_out = edge_mask.data();

    auto g_idx = [&](int y, int x) {
        return static_cast<std::size_t>(y) * W + x;
    };

#ifdef PF_HAS_OPENMP
#pragma omp parallel for collapse(2)
#endif
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            auto clamp_g = [&](int yy, int xx) -> float {
                yy = std::clamp(yy, 0, H - 1);
                xx = std::clamp(xx, 0, W - 1);
                return static_cast<float>(g_in[g_idx(yy, xx)]);
            };

            float gx =
                -1.f * clamp_g(y - 1, x - 1) +
                 0.f * clamp_g(y - 1, x    ) +
                 1.f * clamp_g(y - 1, x + 1) +
                -2.f * clamp_g(y,     x - 1) +
                 0.f * clamp_g(y,     x    ) +
                 2.f * clamp_g(y,     x + 1) +
                -1.f * clamp_g(y + 1, x - 1) +
                 0.f * clamp_g(y + 1, x    ) +
                 1.f * clamp_g(y + 1, x + 1);

            float gy =
                -1.f * clamp_g(y - 1, x - 1) +
                -2.f * clamp_g(y - 1, x    ) +
                -1.f * clamp_g(y - 1, x + 1) +
                 1.f * clamp_g(y + 1, x - 1) +
                 2.f * clamp_g(y + 1, x    ) +
                 1.f * clamp_g(y + 1, x + 1);

            float mag = std::fabs(gx) + std::fabs(gy);
            e_out[g_idx(y, x)] =
                (mag > static_cast<float>(edge_threshold)) ? 0 : 255;
        }
    }

    // 3) 顏色量化
    uint8_t* s_data = smooth.data();
    const std::size_t total = static_cast<std::size_t>(H) * W * C;
    const int levels = 16;
    const float step = 255.0f / (levels - 1);

#ifdef PF_HAS_OPENMP
#pragma omp parallel for
#endif
    for (std::int64_t i = 0; i < static_cast<std::int64_t>(total); ++i) {
        float v = static_cast<float>(s_data[i]);
        int idx_level = static_cast<int>(std::round(v / step));
        v = std::clamp(idx_level * step, 0.0f, 255.0f);
        s_data[i] = static_cast<uint8_t>(v);
    }

    // 4) 疊邊緣
    ImageU8 out_img(H, W, C);
    uint8_t* out = out_img.data();
    const uint8_t edge_color = 20;

#ifdef PF_HAS_OPENMP
#pragma omp parallel for collapse(2)
#endif
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            uint8_t e = e_out[g_idx(y, x)];
            for (int c = 0; c < C; ++c) {
                uint8_t base = s_data[idx(y, x, c, W, C)];
                out[idx(y, x, c, W, C)] = (e == 0) ? edge_color : base;
            }
        }
    }

    return out_img;
}


// =========================
//   對外 API：帶 Backend
// =========================

ImageU8 sharpen(const ImageU8& src, float amount, Backend backend) {
    backend = normalize_backend(backend);

    if (backend == Backend::Auto) {
#ifdef PF_HAS_OPENMP
        backend = Backend::OpenMP;   // 先簡單：Auto => OpenMP
#else
        backend = Backend::Single;
#endif
    }

    switch (backend) {
    case Backend::OpenMP:
#ifdef PF_HAS_OPENMP
        return sharpen_openmp(src, amount);
#else
        return sharpen_single(src, amount);
#endif
    case Backend::Single:
    default:
        return sharpen_single(src, amount);
    }
}

ImageU8 emboss(const ImageU8& src, float strength, Backend backend) {
    backend = normalize_backend(backend);

    if (backend == Backend::Auto) {
#ifdef PF_HAS_OPENMP
        backend = Backend::OpenMP;
#else
        backend = Backend::Single;
#endif
    }

    switch (backend) {
    case Backend::OpenMP:
#ifdef PF_HAS_OPENMP
        return emboss_openmp(src, strength);
#else
        return emboss_single(src, strength);
#endif
    case Backend::Single:
    default:
        return emboss_single(src, strength);
    }
}

ImageU8 cartoonize(const ImageU8& src,
                   float sigma_space,
                   uint8_t edge_threshold,
                   Backend backend) {
    backend = normalize_backend(backend);

    if (backend == Backend::Auto) {
#ifdef PF_HAS_OPENMP
        backend = Backend::OpenMP;
#else
        backend = Backend::Single;
#endif
    }

    switch (backend) {
    case Backend::OpenMP:
#ifdef PF_HAS_OPENMP
        return cartoonize_openmp(src, sigma_space, edge_threshold);
#else
        return cartoonize_single(src, sigma_space, edge_threshold);
#endif
    case Backend::Single:
    default:
        return cartoonize_single(src, sigma_space, edge_threshold);
    }
}


} // namespace pf
