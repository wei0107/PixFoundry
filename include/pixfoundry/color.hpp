#pragma once

#include <cstdint>
#include "image.hpp"
#include "filters.hpp"  // 為了拿到 pf::Backend 定義

namespace pf {

using std::uint8_t;

// 轉灰階：輸入 3-channel，輸出 1-channel
ImageU8 to_grayscale(const ImageU8& src,
                     Backend backend = Backend::Auto);

// 負片效果：v -> 255 - v
ImageU8 invert(const ImageU8& src,
               Backend backend = Backend::Auto);

// 懷舊色調（簡單 sepia）
ImageU8 sepia(const ImageU8& src,
              Backend backend = Backend::Auto);

// 亮度 / 對比調整：new = alpha * old + beta
ImageU8 adjust_brightness_contrast(const ImageU8& src,
                                   float alpha,   // 對比，1.0 保持不變
                                   float beta,    // 亮度偏移，加減
                                   Backend backend = Backend::Auto);

// Gamma 校正：new = 255 * (old / 255)^gamma
ImageU8 gamma_correct(const ImageU8& src,
                      float gamma,
                      Backend backend = Backend::Auto);

} // namespace pf
