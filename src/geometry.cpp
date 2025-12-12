#include "pixfoundry/geometry.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace pf {

static inline std::size_t idx(int y, int x, int c,
                              int W, int C) {
    return (static_cast<std::size_t>(y) * W + x) * C + c;
}

// ======================
//  Bilinear resize (single)
// ======================
static ImageU8 resize_bilinear_single(const ImageU8& src,
                                      int new_h,
                                      int new_w) {
    if (src.empty()) {
        throw std::invalid_argument("resize: empty image");
    }
    if (new_h <= 0 || new_w <= 0) {
        throw std::invalid_argument("resize: invalid new size");
    }

    const int H = src.h();
    const int W = src.w();
    const int C = src.c();

    const uint8_t* in = src.data();
    ImageU8 dst(new_h, new_w, C);
    uint8_t* out = dst.data();

    const float scale_y = static_cast<float>(H) / static_cast<float>(new_h);
    const float scale_x = static_cast<float>(W) / static_cast<float>(new_w);

    for (int y = 0; y < new_h; ++y) {
        float sy = (y + 0.5f) * scale_y - 0.5f;
        int   y0 = static_cast<int>(std::floor(sy));
        float fy = sy - y0;
        int   y1 = y0 + 1;

        y0 = std::clamp(y0, 0, H - 1);
        y1 = std::clamp(y1, 0, H - 1);

        for (int x = 0; x < new_w; ++x) {
            float sx = (x + 0.5f) * scale_x - 0.5f;
            int   x0 = static_cast<int>(std::floor(sx));
            float fx = sx - x0;
            int   x1 = x0 + 1;

            x0 = std::clamp(x0, 0, W - 1);
            x1 = std::clamp(x1, 0, W - 1);

            for (int c = 0; c < C; ++c) {
                float v00 = static_cast<float>(in[idx(y0, x0, c, W, C)]);
                float v10 = static_cast<float>(in[idx(y0, x1, c, W, C)]);
                float v01 = static_cast<float>(in[idx(y1, x0, c, W, C)]);
                float v11 = static_cast<float>(in[idx(y1, x1, c, W, C)]);

                float v0 = v00 + (v10 - v00) * fx;
                float v1 = v01 + (v11 - v01) * fx;
                float v  = v0  + (v1  - v0)  * fy;

                v = std::round(v);
                v = std::clamp(v, 0.0f, 255.0f);
                out[idx(y, x, c, new_w, C)] = static_cast<uint8_t>(v);
            }
        }
    }

    return dst;
}

// ======================
//  Flip (single)
// ======================
static ImageU8 flip_horizontal_single(const ImageU8& src) {
    if (src.empty()) throw std::invalid_argument("flip_horizontal: empty image");

    const int H = src.h();
    const int W = src.w();
    const int C = src.c();
    const uint8_t* in = src.data();

    ImageU8 dst(H, W, C);
    uint8_t* out = dst.data();

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            int sx = W - 1 - x;
            for (int c = 0; c < C; ++c) {
                out[idx(y, x, c, W, C)] = in[idx(y, sx, c, W, C)];
            }
        }
    }

    return dst;
}

static ImageU8 flip_vertical_single(const ImageU8& src) {
    if (src.empty()) throw std::invalid_argument("flip_vertical: empty image");

    const int H = src.h();
    const int W = src.w();
    const int C = src.c();
    const uint8_t* in = src.data();

    ImageU8 dst(H, W, C);
    uint8_t* out = dst.data();

    for (int y = 0; y < H; ++y) {
        int sy = H - 1 - y;
        for (int x = 0; x < W; ++x) {
            for (int c = 0; c < C; ++c) {
                out[idx(y, x, c, W, C)] = in[idx(sy, x, c, W, C)];
            }
        }
    }

    return dst;
}

// ======================
//  Crop (single)
// ======================
static ImageU8 crop_single(const ImageU8& src,
                           int y0,
                           int x0,
                           int h,
                           int w) {
    if (src.empty()) throw std::invalid_argument("crop: empty image");
    if (h <= 0 || w <= 0) throw std::invalid_argument("crop: invalid size");

    const int H = src.h();
    const int W = src.w();
    const int C = src.c();
    const uint8_t* in = src.data();

    // clamp 範圍
    int y1 = std::clamp(y0, 0, H);
    int x1 = std::clamp(x0, 0, W);
    int y2 = std::clamp(y0 + h, 0, H);
    int x2 = std::clamp(x0 + w, 0, W);

    int out_h = std::max(0, y2 - y1);
    int out_w = std::max(0, x2 - x1);

    ImageU8 dst(out_h, out_w, C);
    uint8_t* out = dst.data();

    for (int y = 0; y < out_h; ++y) {
        for (int x = 0; x < out_w; ++x) {
            int sy = y1 + y;
            int sx = x1 + x;
            for (int c = 0; c < C; ++c) {
                out[idx(y, x, c, out_w, C)] = in[idx(sy, sx, c, W, C)];
            }
        }
    }

    return dst;
}

// ======================
//  Rotate (single, keep same size)
// ======================
static ImageU8 rotate_single(const ImageU8& src,
                             float angle_deg) {
    if (src.empty()) throw std::invalid_argument("rotate: empty image");

    const int H = src.h();
    const int W = src.w();
    const int C = src.c();
    const uint8_t* in = src.data();

    ImageU8 dst(H, W, C);
    uint8_t* out = dst.data();

    const float pi = std::acos(-1.0f);
    float rad = angle_deg * pi / 180.0f;
    float cos_t = std::cos(rad);
    float sin_t = std::sin(rad);

    float cx = (W - 1) * 0.5f;
    float cy = (H - 1) * 0.5f;

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            // 目的座標 (x, y) 對應回原圖座標 (sx, sy)
            float dx = x - cx;
            float dy = y - cy;

            float sx =  cos_t * dx + sin_t * dy + cx;
            float sy = -sin_t * dx + cos_t * dy + cy;

            int x0 = static_cast<int>(std::floor(sx));
            int y0 = static_cast<int>(std::floor(sy));
            float fx = sx - x0;
            float fy = sy - y0;

            // 在邊界外 → 填黑
            if (x0 < 0 || x0 >= W ||
                y0 < 0 || y0 >= H) {
                for (int c = 0; c < C; ++c) {
                    out[idx(y, x, c, W, C)] = 0;
                }
                continue;
            }

            int x1 = std::clamp(x0 + 1, 0, W - 1);
            int y1 = std::clamp(y0 + 1, 0, H - 1);

            for (int c = 0; c < C; ++c) {
                float v00 = static_cast<float>(in[idx(y0, x0, c, W, C)]);
                float v10 = static_cast<float>(in[idx(y0, x1, c, W, C)]);
                float v01 = static_cast<float>(in[idx(y1, x0, c, W, C)]);
                float v11 = static_cast<float>(in[idx(y1, x1, c, W, C)]);

                float v0 = v00 + (v10 - v00) * fx;
                float v1 = v01 + (v11 - v01) * fx;
                float v  = v0  + (v1  - v0)  * fy;

                v = std::round(v);
                v = std::clamp(v, 0.0f, 255.0f);
                out[idx(y, x, c, W, C)] = static_cast<uint8_t>(v);
            }
        }
    }

    return dst;
}

// ======================
//  Public APIs with Backend
// ======================

ImageU8 resize(const ImageU8& src,
               int new_h,
               int new_w,
               Backend backend) {
    switch (backend) {
    case Backend::Single:
    case Backend::Auto:
    default:
        return resize_bilinear_single(src, new_h, new_w);
    }
}

ImageU8 flip_horizontal(const ImageU8& src,
                        Backend backend) {
    switch (backend) {
    case Backend::Single:
    case Backend::Auto:
    default:
        return flip_horizontal_single(src);
    }
}

ImageU8 flip_vertical(const ImageU8& src,
                      Backend backend) {
    switch (backend) {
    case Backend::Single:
    case Backend::Auto:
    default:
        return flip_vertical_single(src);
    }
}

ImageU8 crop(const ImageU8& src,
             int y,
             int x,
             int h,
             int w,
             Backend backend) {
    switch (backend) {
    case Backend::Single:
    case Backend::Auto:
    default:
        return crop_single(src, y, x, h, w);
    }
}

ImageU8 rotate(const ImageU8& src,
               float angle_deg,
               Backend backend) {
    switch (backend) {
    case Backend::Single:
    case Backend::Auto:
    default:
        return rotate_single(src, angle_deg);
    }
}

} // namespace pf
