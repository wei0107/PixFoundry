#include "pixfoundry/filters.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace pf {

static inline int reflect(int i, int N) {
    // 反射邊界: ..., 2,1,0 | 0,1,2,...,N-1 | N-1,N-2,...
    if (i < 0) return -i - 1;
    if (i >= N) return 2 * N - i - 1;
    return i;
}

std::vector<float> box_kernel1d(int k) {
    if (k < 3 || (k % 2 == 0)) throw std::invalid_argument("box_kernel1d: k must be odd >=3");
    return std::vector<float>(k, 1.0f / k);
}

std::vector<float> gaussian_kernel1d(float sigma) {
    if (!(sigma > 0.f)) throw std::invalid_argument("gaussian_kernel1d: sigma must be > 0");
    // kernel size: ceil(6*sigma)|odd
    int k = std::max(3, (int)std::ceil(6.f * sigma) | 1);
    std::vector<float> k1(k);
    int r = k / 2;
    float inv2s2 = 1.f / (2.f * sigma * sigma);
    float sum = 0.f;
    for (int i = -r; i <= r; ++i) {
        float v = std::exp(- (i * i) * inv2s2);
        k1[i + r] = v;
        sum += v;
    }
    for (float& v : k1) v /= sum;
    return k1;
}

static ImageU8 convolve_separable_u8(const ImageU8& src,
                                     const std::vector<float>& k1d)
{
    if (src.empty()) throw std::invalid_argument("convolve: empty src");
    const int H = src.h(), W = src.w(), C = src.c();
    const int K = (int)k1d.size(), R = K / 2;

    // 中間緩衝: 水平結果 (float)
    std::vector<float> tmp((size_t)H * W * C, 0.f);
    ImageU8 dst(H, W, C);

    // 1) 水平
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            for (int c = 0; c < C; ++c) {
                float sum = 0.f;
                for (int t = -R; t <= R; ++t) {
                    int xx = reflect(x + t, W);
                    sum += k1d[t + R] * src.data()[( (y * W + xx) * C ) + c];
                }
                tmp[( (y * W + x) * C ) + c] = sum;
            }
        }
    }

    // 2) 垂直
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            for (int c = 0; c < C; ++c) {
                float sum = 0.f;
                for (int t = -R; t <= R; ++t) {
                    int yy = reflect(y + t, H);
                    sum += k1d[t + R] * tmp[( (yy * W + x) * C ) + c];
                }
                // 轉回 uint8_t with clamp & round
                float v = std::round(sum);
                v = std::min(255.f, std::max(0.f, v));
                dst.data()[( (y * W + x) * C ) + c] = (uint8_t)v;
            }
        }
    }

    return dst;
}

ImageU8 mean_filter(const ImageU8& src, int ksize) {
    auto k = box_kernel1d(ksize);
    return convolve_separable_u8(src, k);
}

ImageU8 gaussian_filter(const ImageU8& src, float sigma) {
    auto k = gaussian_kernel1d(sigma);
    return convolve_separable_u8(src, k);
}

} // namespace pf
