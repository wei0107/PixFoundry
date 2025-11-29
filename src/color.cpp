#include "pixfoundry/color.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace pf {

static inline std::size_t idx(int y, int x, int c,
                              int W, int C) {
    return (static_cast<std::size_t>(y) * W + x) * C + c;
}

// =========================
//   single-thread 版本實作
// =========================

static ImageU8 to_grayscale_single(const ImageU8& src) {
    if (src.empty()) {
        throw std::invalid_argument("to_grayscale: empty image");
    }

    const int H = src.h();
    const int W = src.w();
    const int C = src.c();

    const uint8_t* in = src.data();

    if (C == 1) {
        // 已經是灰階，手動複製一份（避免呼叫被 delete 的 copy ctor）
        ImageU8 dst(H, W, 1);
        uint8_t* out = dst.data();
        const std::size_t total = static_cast<std::size_t>(H) * W;
        std::copy(in, in + total, out);
        return dst;
    }

    ImageU8 dst(H, W, 1);
    uint8_t* out = dst.data();

    constexpr float wr = 0.299f;
    constexpr float wg = 0.587f;
    constexpr float wb = 0.114f;

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            std::size_t base = idx(y, x, 0, W, C);
            uint8_t r = in[base + 0];
            uint8_t g = in[base + 1];
            uint8_t b = in[base + 2];

            float v = wr * r + wg * g + wb * b;
            v = std::round(v);
            v = std::clamp(v, 0.0f, 255.0f);
            out[y * W + x] = static_cast<uint8_t>(v);
        }
    }

    return dst;
}

static ImageU8 invert_single(const ImageU8& src) {
    if (src.empty()) {
        throw std::invalid_argument("invert: empty image");
    }

    const int H = src.h();
    const int W = src.w();
    const int C = src.c();

    const uint8_t* in = src.data();
    ImageU8 dst(H, W, C);
    uint8_t* out = dst.data();

    const std::size_t total = static_cast<std::size_t>(H) * W * C;
    for (std::size_t i = 0; i < total; ++i) {
        out[i] = static_cast<uint8_t>(255 - in[i]);
    }

    return dst;
}

static ImageU8 sepia_single(const ImageU8& src) {
    if (src.empty()) {
        throw std::invalid_argument("sepia: empty image");
    }

    const int H = src.h();
    const int W = src.w();
    const int C = src.c();

    if (C != 3) {
        throw std::invalid_argument("sepia: expects 3-channel RGB image");
    }

    const uint8_t* in = src.data();
    ImageU8 dst(H, W, 3);
    uint8_t* out = dst.data();

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            std::size_t base = idx(y, x, 0, W, 3);

            float r = static_cast<float>(in[base + 0]);
            float g = static_cast<float>(in[base + 1]);
            float b = static_cast<float>(in[base + 2]);

            float tr = 0.393f * r + 0.769f * g + 0.189f * b;
            float tg = 0.349f * r + 0.686f * g + 0.168f * b;
            float tb = 0.272f * r + 0.534f * g + 0.131f * b;

            tr = std::clamp(std::round(tr), 0.0f, 255.0f);
            tg = std::clamp(std::round(tg), 0.0f, 255.0f);
            tb = std::clamp(std::round(tb), 0.0f, 255.0f);

            out[base + 0] = static_cast<uint8_t>(tr);
            out[base + 1] = static_cast<uint8_t>(tg);
            out[base + 2] = static_cast<uint8_t>(tb);
        }
    }

    return dst;
}

static ImageU8 adjust_brightness_contrast_single(const ImageU8& src,
                                                 float alpha,
                                                 float beta) {
    if (src.empty()) {
        throw std::invalid_argument("adjust_brightness_contrast: empty image");
    }

    const int H = src.h();
    const int W = src.w();
    const int C = src.c();

    const uint8_t* in = src.data();
    ImageU8 dst(H, W, C);
    uint8_t* out = dst.data();

    const std::size_t total = static_cast<std::size_t>(H) * W * C;
    for (std::size_t i = 0; i < total; ++i) {
        float v = alpha * static_cast<float>(in[i]) + beta;
        v = std::clamp(std::round(v), 0.0f, 255.0f);
        out[i] = static_cast<uint8_t>(v);
    }

    return dst;
}

static ImageU8 gamma_correct_single(const ImageU8& src,
                                    float gamma) {
    if (src.empty()) {
        throw std::invalid_argument("gamma_correct: empty image");
    }
    if (!(gamma > 0.0f)) {
        throw std::invalid_argument("gamma_correct: gamma must be > 0");
    }

    const int H = src.h();
    const int W = src.w();
    const int C = src.c();

    const uint8_t* in = src.data();
    ImageU8 dst(H, W, C);
    uint8_t* out = dst.data();

    // 查表加速
    float inv = 1.0f / 255.0f;
    uint8_t lut[256];
    for (int i = 0; i < 256; ++i) {
        float x = static_cast<float>(i) * inv;
        float y = std::pow(x, gamma);
        float v = std::clamp(std::round(y * 255.0f), 0.0f, 255.0f);
        lut[i] = static_cast<uint8_t>(v);
    }

    const std::size_t total = static_cast<std::size_t>(H) * W * C;
    for (std::size_t i = 0; i < total; ++i) {
        out[i] = lut[in[i]];
    }

    return dst;
}

// =========================
//   對外 API：帶 Backend
// =========================

ImageU8 to_grayscale(const ImageU8& src,
                     Backend backend) {
    switch (backend) {
    case Backend::Single:
    case Backend::Auto:
    default:
        // 目前 Auto 一律走 single，之後可以在這裡選 OpenMP
        return to_grayscale_single(src);
    }
}

ImageU8 invert(const ImageU8& src,
               Backend backend) {
    switch (backend) {
    case Backend::Single:
    case Backend::Auto:
    default:
        return invert_single(src);
    }
}

ImageU8 sepia(const ImageU8& src,
              Backend backend) {
    switch (backend) {
    case Backend::Single:
    case Backend::Auto:
    default:
        return sepia_single(src);
    }
}

ImageU8 adjust_brightness_contrast(const ImageU8& src,
                                   float alpha,
                                   float beta,
                                   Backend backend) {
    switch (backend) {
    case Backend::Single:
    case Backend::Auto:
    default:
        return adjust_brightness_contrast_single(src, alpha, beta);
    }
}

ImageU8 gamma_correct(const ImageU8& src,
                      float gamma,
                      Backend backend) {
    switch (backend) {
    case Backend::Single:
    case Backend::Auto:
    default:
        return gamma_correct_single(src, gamma);
    }
}

} // namespace pf
