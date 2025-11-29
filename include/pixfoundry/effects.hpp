#pragma once

#include <cstdint>
#include "image.hpp"
#include "filters.hpp"  // 拿 Backend 定義

namespace pf {

using std::uint8_t;

// 銳化：簡單 unsharp mask / 銳化 kernel，amount 控制強度
ImageU8 sharpen(const ImageU8& src,
                float amount = 1.0f,
                Backend backend = Backend::Auto);

// 浮雕效果：製造立體感，strength 控制強度
ImageU8 emboss(const ImageU8& src,
               float strength = 1.0f,
               Backend backend = Backend::Auto);

// 簡易卡通化：平滑 + 邊緣線條
ImageU8 cartoonize(const ImageU8& src,
                   float sigma_space = 2.0f,      // 平滑用 Gaussian sigma
                   uint8_t edge_threshold = 40,    // Sobel 邊緣門檻
                   Backend backend = Backend::Auto);

} // namespace pf
