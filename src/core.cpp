#include "pixfoundry/image.hpp"
#include <algorithm>
#include <stdexcept>
#include <string>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

namespace pf {

ImageU8 load_image(const std::string& path) {
    int w = 0, h = 0, ch = 0;
    stbi_uc* raw = stbi_load(path.c_str(), &w, &h, &ch, 0);
    if (!raw) throw std::runtime_error("stb_image: failed to load " + path);

    int out_c = (ch == 1) ? 1 : 3;
    ImageU8 img(h, w, out_c);
    const size_t N = static_cast<size_t>(h) * w;

    if (ch == 1) {
        std::copy(raw, raw + N, img.data());
    } else if (ch == 2) {
        for (size_t i = 0; i < N; ++i) img.data()[i] = raw[i*2];
    } else if (ch == 3) {
        std::copy(raw, raw + N*3, img.data());
    } else if (ch == 4) {
        auto* dst = img.data();
        for (size_t i = 0; i < N; ++i) {
            dst[i*3+0] = raw[i*4+0];
            dst[i*3+1] = raw[i*4+1];
            dst[i*3+2] = raw[i*4+2];
        }
    }
    stbi_image_free(raw);
    return img;
}

void save_image(const std::string& path, const ImageU8& img, int jpeg_quality) {
    const int w = img.width();
    const int h = img.height();
    const int c = img.channels();
    const uint8_t* data = img.data();

    auto lower = [](std::string s) {
        std::transform(s.begin(), s.end(), s.begin(), ::tolower);
        return s;
    };
    std::string ext = lower(path.substr(path.find_last_of('.')+1));

    int ok = 0;
    if (ext == "png") {
        ok = stbi_write_png(path.c_str(), w, h, c, data, w*c);
    } else if (ext == "jpg" || ext == "jpeg") {
        ok = stbi_write_jpg(path.c_str(), w, h, c, data, jpeg_quality);
    } else if (ext == "bmp") {
        ok = stbi_write_bmp(path.c_str(), w, h, c, data);
    } else if (ext == "tga") {
        ok = stbi_write_tga(path.c_str(), w, h, c, data);
    } else {
        // 預設 PNG
        ok = stbi_write_png(path.c_str(), w, h, c, data, w*c);
    }
    if (!ok) throw std::runtime_error("stb_image_write: failed to save " + path);
}

} // namespace pf
