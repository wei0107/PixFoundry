#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cstring>
#include "pixfoundry/image.hpp"

namespace py = pybind11;
using pf::ImageU8;

static py::array_t<uint8_t> to_numpy(const ImageU8& img) {
    const int h = img.height(), w = img.width(), c = img.channels();
    if (c == 1) {
        py::array_t<uint8_t> arr({h, w});
        std::memcpy(arr.mutable_data(), img.data(), static_cast<size_t>(h) * w);
        return arr;
    } else {
        py::array_t<uint8_t> arr({h, w, 3});
        std::memcpy(arr.mutable_data(), img.data(), static_cast<size_t>(h) * w * 3);
        return arr;
    }
}

static ImageU8 from_numpy(const py::array& arr) {
    if (arr.dtype().kind() != 'u' || arr.dtype().itemsize() != 1)
        throw std::runtime_error("Expect uint8 numpy array");

    if (arr.ndim() == 2) {
        const int h = static_cast<int>(arr.shape(0));
        const int w = static_cast<int>(arr.shape(1));
        ImageU8 img(h, w, 1);
        auto info = arr.request();
        const auto* src = static_cast<const uint8_t*>(info.ptr);
        std::copy(src, src + static_cast<size_t>(h) * w, img.data());
        return img;
    } else if (arr.ndim() == 3 && arr.shape(2) == 3) {
        const int h = static_cast<int>(arr.shape(0));
        const int w = static_cast<int>(arr.shape(1));
        ImageU8 img(h, w, 3);
        auto info = arr.request();
        const auto* src = static_cast<const uint8_t*>(info.ptr);
        std::copy(src, src + static_cast<size_t>(h) * w * 3, img.data());
        return img;
    }
    throw std::runtime_error("Expect shape (H,W) or (H,W,3) uint8 array");
}

PYBIND11_MODULE(_core, m) {
    m.doc() = "PixFoundry: image IO via stb_image/stb_image_write";

    m.def("load_image",
          [](const std::string& path) {
              return to_numpy(pf::load_image(path));
          },
          py::arg("path"),
          "Load image (JPG/PNG/BMP/PSD/GIF/HDR/PIC/PNM) via stb.");

    m.def("save_image",
          [](const std::string& path, const py::array& arr, int quality) {
              pf::save_image(path, from_numpy(arr), quality);
          },
          py::arg("path"), py::arg("array"), py::arg("quality") = 90,
          "Save image (PNG/JPG/BMP/TGA) via stb. 'quality' is used for JPG.");
}
