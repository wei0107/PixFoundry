#pragma once

#include <cstdint>
#include <vector>
#include "pixfoundry/image.hpp"

namespace pf {

// ------------------------------------------------------------
// 邊界模式
// ------------------------------------------------------------
enum class Border {
    Reflect,
    Replicate,
    Wrap,
    Constant
};

// ------------------------------------------------------------
// 後端（目前實作只用到 Single，但 bindings 有 "auto"）
// ------------------------------------------------------------
enum class Backend {
    Auto,    // 給 Python 的 "auto" 用（目前你實作裡可以當成 Single 用）
    Single,  // 單執行緒
    // 之後如果要加 OpenMP 可以多加一個 OpenMP
};

// 這個型別已經在 image.hpp 裡定義了：class ImageU8 {...};
using std::uint8_t;

// ------------------------------------------------------------
// 濾波 API（對應你現在 filters.cpp 裡的定義）
// ------------------------------------------------------------

// 平均濾波（均值濾波）
// ksize: 奇數且 >= 3
ImageU8 mean_filter(const ImageU8& src,
                    int ksize,
                    Border border = Border::Reflect,
                    Backend backend = Backend::Single,
                    uint8_t border_value = 0);

// Gaussian 濾波
// sigma: > 0
ImageU8 gaussian_filter(const ImageU8& src,
                        float sigma,
                        Border border = Border::Reflect,
                        Backend backend = Backend::Single,
                        uint8_t border_value = 0);

// 中值濾波（鹽胡椒雜訊）
ImageU8 median_filter(const ImageU8& src,
                      int ksize,
                      Border border = Border::Reflect,
                      Backend backend = Backend::Single,
                      uint8_t border_value = 0);

// 雙邊濾波
ImageU8 bilateral_filter(const ImageU8& src,
                         int ksize,
                         float sigma_color,
                         float sigma_space,
                         Border border = Border::Reflect,
                         Backend backend = Backend::Single,
                         uint8_t border_value = 0);

// ------------------------------------------------------------
// Kernel utilities
// ------------------------------------------------------------

// box kernel: k 個 1/k
std::vector<float> box_kernel1d(int ksize);

// gaussian kernel: sum = 1
std::vector<float> gaussian_kernel1d(float sigma);

} // namespace pf
