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

// Week3: Median filter（中值濾波）
ImageU8 median_filter(const ImageU8& src, int ksize,
                      Border border = Border::Reflect,
                      Backend backend = Backend::Single,
                      uint8_t border_value = 0);

// Week3: Bilateral filter（雙邊濾波）
// ksize：核心大小（必須為奇數，例如 3/5/7）
// sigma_color：顏色差異的標準差（range Gaussian）
// sigma_space：空間距離的標準差（spatial Gaussian）
ImageU8 bilateral_filter(const ImageU8& src,
                         int ksize,
                         float sigma_color,
                         float sigma_space,
                         Border border = Border::Reflect,
                         Backend backend = Backend::Single,
                         uint8_t border_value = 0);
                         
// ---- Kernel utilities ----

// box kernel: k 個 1/k
std::vector<float> box_kernel1d(int ksize);

// gaussian kernel: sum = 1
std::vector<float> gaussian_kernel1d(float sigma);

std::vector<float> box_kernel1d(int ksize);
std::vector<float> gaussian_kernel1d(float sigma);

} // namespace pf
