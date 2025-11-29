#pragma once
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

namespace pf {

class ImageU8 {
public:
    ImageU8() = default;

    // 擁有新配置（自有）的影像
    ImageU8(int h, int w, int c)
        : h_(h), w_(w), c_(c)
    {
        if (h <= 0 || w <= 0 || (c != 1 && c != 3))
            throw std::invalid_argument("ImageU8: invalid shape");
        const size_t n = static_cast<size_t>(h_) * w_ * c_;
        // 自己配置，shared_ptr 確保生命週期
        data_ = std::shared_ptr<uint8_t[]>(new uint8_t[n](), std::default_delete<uint8_t[]>());
    }

    // 共享外部緩衝區（零拷貝）
    ImageU8(int h, int w, int c, std::shared_ptr<uint8_t[]> external)
        : h_(h), w_(w), c_(c), data_(std::move(external))
    {
        if (!data_) throw std::invalid_argument("ImageU8: null external buffer");
        if (h <= 0 || w <= 0 || (c != 1 && c != 3))
            throw std::invalid_argument("ImageU8: invalid shape");
    }

    // 不允許複製（避免意外深拷）
    ImageU8(const ImageU8&)            = delete;
    ImageU8& operator=(const ImageU8&) = delete;

    // 允許移動
    ImageU8(ImageU8&&)            = default;
    ImageU8& operator=(ImageU8&&) = default;

    int  h() const { return h_; }
    int  w() const { return w_; }
    int  c() const { return c_; }
    bool empty() const { return !data_; }

    uint8_t*       data()       { return data_.get(); }
    const uint8_t* data() const { return data_.get(); }

    // 暴露 shared_ptr 以便 pybind11 綁定時延長生命週期
    const std::shared_ptr<uint8_t[]>& shared() const { return data_; }

private:
    size_t h_ = 0, w_ = 0, c_ = 1;
    std::shared_ptr<uint8_t[]> data_;
};

    ImageU8 load_image_u8(const std::string& path);
    void    save_image_u8(const std::string& path,
                        const uint8_t* data, int h, int w, int c);

} // namespace pf
