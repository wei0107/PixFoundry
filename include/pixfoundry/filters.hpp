#pragma once
#include <cstdint>
#include <vector>
#include "pixfoundry/image.hpp"

namespace pf {

// 平均濾波：ksize 必須是奇數 >= 3
ImageU8 mean_filter(const ImageU8& src, int ksize);

// Gaussian 濾波：sigma > 0
ImageU8 gaussian_filter(const ImageU8& src, float sigma);

// （可選）內部工具：供單元測試 / 其他濾波重用
std::vector<float> gaussian_kernel1d(float sigma); // 正規化為 1
std::vector<float> box_kernel1d(int ksize);        // 長度 k，全 1/k

} // namespace pf
