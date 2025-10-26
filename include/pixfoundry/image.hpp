#pragma once
#include <vector>
#include <cstdint>
#include <string>
#include <stdexcept>

namespace pf {

class ImageU8 {
public:
    ImageU8() = default;
    ImageU8(int h, int w, int c)
        : h_(h), w_(w), c_(c), data_(static_cast<size_t>(h) * w * c, 0) {
        if (h <= 0 || w <= 0 || (c != 1 && c != 3))
            throw std::invalid_argument("ImageU8: invalid shape");
    }

    uint8_t* data() noexcept { return data_.data(); }
    const uint8_t* data() const noexcept { return data_.data(); }

    int height() const noexcept { return h_; }
    int width()  const noexcept { return w_; }
    int channels() const noexcept { return c_; }

private:
    int h_ = 0, w_ = 0, c_ = 0;
    std::vector<uint8_t> data_;
};

// 通用 I/O：只走 stb
ImageU8 load_image(const std::string& path);
void    save_image(const std::string& path, const ImageU8& img, int jpeg_quality = 90);

} // namespace pf
