#include "pixfoundry/image.hpp"
#include <string>
#include <stdexcept>

// stb
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

namespace pf {

// 低階：回傳 ImageU8（內含 shared_ptr 指向 stb 的 buffer）
// 零拷貝：不再做 std::copy，直接共享 stb 配置的像素
ImageU8 load_image_u8(const std::string& path)
{
    int w = 0, h = 0, ch_in = 0;

    // 我們要求輸出 1 或 3 通道，避免回來再轉通道造成再次配置
    // 若原圖是 1 通道 → 要 1；否則 → 要 3
    // 但 stbi_load 的第五參數是「要求的輸出通道數」，不會回傳原圖通道數
    // 因此我們先用 stbi_info 得知原始通道數，再決定 desired_c。
    int x_dummy=0,y_dummy=0,comp_dummy=0;
    int ok = stbi_info(path.c_str(), &x_dummy, &y_dummy, &comp_dummy);
    if (!ok) throw std::runtime_error("stb_image: failed to stbi_info " + path);

    const int desired_c = (comp_dummy == 1) ? 1 : 3;

    stbi_uc* raw = stbi_load(path.c_str(), &w, &h, &ch_in, desired_c);
    if (!raw) throw std::runtime_error("stb_image: failed to load " + path);

    // 將 raw 交給 shared_ptr 管理，deleter 使用 stbi_image_free
    std::shared_ptr<uint8_t[]> sp(
        reinterpret_cast<uint8_t*>(raw),
        [](uint8_t* p){ stbi_image_free(p); }
    );

    return ImageU8(h, w, desired_c, std::move(sp));
}


// 簡單存檔工具：支援 .png / .jpg
// 期望輸入為 HxW 或 HxWx3 的 uint8_t 緩衝區
// alpha/2ch 不處理（若 numpy 來的是 RGBA，請在 bindings 端先轉 3ch）
void save_image_u8(const std::string& path,
                   const uint8_t* data, int h, int w, int c)
{
    if (!data || h <= 0 || w <= 0 || (c != 1 && c != 3))
        throw std::invalid_argument("save_image: invalid input");

    const auto ends_with = [](const std::string& s, const char* suf){
        const size_t n = std::char_traits<char>::length(suf);
        return s.size() >= n && s.compare(s.size()-n, n, suf) == 0;
    };

    if (ends_with(path, ".png") || ends_with(path, ".PNG"))
    {
        int stride = w * c; // bytes per row
        if (!stbi_write_png(path.c_str(), w, h, c, data, stride))
            throw std::runtime_error("stb_image_write: failed to write png " + path);
    }
    else if (ends_with(path, ".jpg") || ends_with(path, ".jpeg")
          || ends_with(path, ".JPG") || ends_with(path, ".JPEG"))
    {
        int quality = 95;
        if (!stbi_write_jpg(path.c_str(), w, h, c, data, quality))
            throw std::runtime_error("stb_image_write: failed to write jpg " + path);
    }
    else {
        throw std::runtime_error("save_image: unsupported extension (use .png/.jpg): " + path);
    }
}

} // namespace pf
