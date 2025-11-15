#pragma once
#include <cstdint>
#include <vector>
#include "pixfoundry/image.hpp"

namespace pf {

// 邊界模式
enum class Border {
    Reflect,
    Replicate,
    Wrap,
    Constant
};

// 後端（目前僅 Single）
enum class Backend { Auto, Single };

// ---- API：直接用預設參數即可兼容舊版 ----

// Mean (Box) filter
ImageU8 mean_filter(const ImageU8& src, int ksize,
                    Border border = Border::Reflect,
                    Backend backend = Backend::Single,
                    uint8_t border_value = 0);

// Gaussian filter
ImageU8 gaussian_filter(const ImageU8& src, float sigma,
                        Border border = Border::Reflect,
                        Backend backend = Backend::Single,
                        uint8_t border_value = 0);

// ---- Kernel utilities ----

// box kernel: k 個 1/k
std::vector<float> box_kernel1d(int ksize);

// gaussian kernel: sum = 1
std::vector<float> gaussian_kernel1d(float sigma);

} // namespace pf
