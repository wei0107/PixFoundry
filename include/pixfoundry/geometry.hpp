#pragma once

#include <cstdint>
#include "pixfoundry/image.hpp"
#include "pixfoundry/filters.hpp"  // 為了取得 Backend enum

namespace pf {

// resize：使用 bilinear，輸出 new_h x new_w
ImageU8 resize(const ImageU8& src,
               int new_h,
               int new_w,
               Backend backend = Backend::Auto);

// rotate：以中心為旋轉軸，輸出尺寸跟原圖一樣
ImageU8 rotate(const ImageU8& src,
               float angle_deg,
               Backend backend = Backend::Auto);

// flip：水平或垂直翻轉
ImageU8 flip_horizontal(const ImageU8& src,
                        Backend backend = Backend::Auto);

ImageU8 flip_vertical(const ImageU8& src,
                      Backend backend = Backend::Auto);

// crop：從 (y, x) 開始取 h x w 區域，超出會自動 clamp
ImageU8 crop(const ImageU8& src,
             int y,
             int x,
             int h,
             int w,
             Backend backend = Backend::Auto);

} // namespace pf
