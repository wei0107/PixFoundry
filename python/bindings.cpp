#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "pixfoundry/image.hpp"
#include "pixfoundry/filters.hpp"
#include "pixfoundry/color.hpp"
#include "pixfoundry/effects.hpp"
#include "pixfoundry/geometry.hpp"
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace pfpy {

using pf::ImageU8;
using pf::Border;
using pf::Backend;

// ------------------------------------------------------------
// 共用：檢查 numpy array (uint8, C-contiguous, HxW or HxWxC)
// ------------------------------------------------------------
struct ShapeInfo {
    int h;
    int w;
    int c;
};

static ShapeInfo check_uint8_hw_or_hwc(const py::buffer_info& info) {
    if (info.ndim != 2 && info.ndim != 3) {
        throw std::runtime_error("expected HxW or HxWxC uint8 array");
    }
    if (info.itemsize != 1) {
        throw std::runtime_error("expected dtype=uint8");
    }

    const int h = static_cast<int>(info.shape[0]);
    const int w = static_cast<int>(info.shape[1]);
    const int c = (info.ndim == 3) ? static_cast<int>(info.shape[2]) : 1;

    if (c != 1 && c != 3) {
        throw std::runtime_error("expected 1 or 3 channels");
    }

    // C-contiguous 檢查
    if (info.ndim == 2) {
        // H x W: strides = [W, 1]
        if (!(info.strides[0] == static_cast<ssize_t>(w) &&
              info.strides[1] == 1)) {
            throw std::runtime_error("expected C-contiguous array (HxW)");
        }
    } else {
        // H x W x C: strides = [W*C, C, 1]
        if (!(info.strides[0] == static_cast<ssize_t>(w * c) &&
              info.strides[1] == static_cast<ssize_t>(c) &&
              info.strides[2] == 1)) {
            throw std::runtime_error("expected C-contiguous array (HxWxC)");
        }
    }

    return {h, w, c};
}

// ------------------------------------------------------------
// numpy.ndarray -> ImageU8（零拷貝）
// ------------------------------------------------------------
static ImageU8 numpy_to_imageu8_zero_copy(const py::array& array) {
    py::buffer_info info = array.request();
    auto shape = check_uint8_hw_or_hwc(info);

    auto* ptr = static_cast<uint8_t*>(info.ptr);

    // 持有原始 numpy 陣列，確保其生命週期 >= ImageU8
    py::object owner = array;

    // shared_ptr 不負責釋放 ptr，僅透過捕獲 owner 延長生命週期
    std::shared_ptr<uint8_t[]> sp(ptr, [owner](uint8_t*) mutable {
        // 不 delete ptr，numpy 擁有這塊記憶體
        // owner 的生命週期由 shared_ptr 的複本管理
    });

    return ImageU8(shape.h, shape.w, shape.c, std::move(sp));
}

// ------------------------------------------------------------
// ImageU8 -> numpy.ndarray（零拷貝）
// ------------------------------------------------------------
static py::array imageu8_to_numpy(const ImageU8& img) {
    if (img.empty()) {
        throw std::runtime_error("Image is empty");
    }

    const int h = img.h();
    const int w = img.w();
    const int c = img.c();

    std::vector<ssize_t> shape;
    std::vector<ssize_t> strides;

    if (c == 1) {
        shape   = {h, w};
        strides = {static_cast<ssize_t>(w), 1};
    } else {
        shape   = {h, w, c};
        strides = {static_cast<ssize_t>(w * c),
                   static_cast<ssize_t>(c),
                   1};
    }

    // 建立 shared_ptr 副本放進 capsule，讓 numpy 管理一份 ref 計數
    auto* sp_copy = new std::shared_ptr<uint8_t[]>(img.shared());
    py::capsule base(sp_copy, [](void* p) {
        delete reinterpret_cast<std::shared_ptr<uint8_t[]>*>(p);
    });

    return py::array(
        py::dtype::of<uint8_t>(),
        shape,
        strides,
        img.shared().get(),  // data pointer
        base                 // base object to keep memory alive
    );
}

// ------------------------------------------------------------
// 檔案 I/O 包裝（load/save 本身也零拷貝）
// ------------------------------------------------------------
static py::array load_image_py(const std::string& path) {
    ImageU8 im = pf::load_image_u8(path);
    return imageu8_to_numpy(im);
}

static void save_image_py(const std::string& path, const py::array& array) {
    py::buffer_info info = array.request();
    auto shape = check_uint8_hw_or_hwc(info);

    const auto* ptr = static_cast<const uint8_t*>(info.ptr);
    pf::save_image_u8(path, ptr, shape.h, shape.w, shape.c);
}

// ------------------------------------------------------------
// 文字參數 → enum
// ------------------------------------------------------------
static Border parse_border(const std::string& s) {
    if (s == "reflect")   return Border::Reflect;
    if (s == "replicate") return Border::Replicate;
    if (s == "wrap")      return Border::Wrap;
    if (s == "constant")  return Border::Constant;
    throw std::runtime_error("border must be one of: reflect, replicate, wrap, constant");
}

static Backend parse_backend(const std::string& s) {
    if (s == "auto") {
    #ifdef PF_HAS_OPENMP
        return Backend::OpenMP;
    #else
        return Backend::Single;
    #endif
    }
    if (s == "single") return Backend::Single;
    if (s == "openmp" || s == "omp") return Backend::OpenMP;
    throw std::runtime_error("backend must be one of: auto, single");
}

// ------------------------------------------------------------
// 共用 wrap：把 numpy 轉 ImageU8 → 呼叫 C++ filter → 再轉回 numpy
// ------------------------------------------------------------
template <typename FilterFunc, typename ParamT>
static py::array wrap_filter(
    const py::array& src,
    ParamT param,
    const std::string& backend_str,
    const std::string& border_str,
    uint8_t border_value,
    FilterFunc&& func)
{
    ImageU8 in  = numpy_to_imageu8_zero_copy(src);  // 這裡是零拷貝
    Border  b   = parse_border(border_str);
    Backend be  = parse_backend(backend_str);
    ImageU8 out = func(in, param, b, be, border_value);
    return imageu8_to_numpy(out);                   // 這裡也是零拷貝
}

} // namespace pfpy

// ------------------------------------------------------------
// pybind11 module
// ------------------------------------------------------------
PYBIND11_MODULE(_core, m) {
    using namespace pfpy;

    m.doc() = "PixFoundry core (zero-copy image IO + filters)";

    // Image IO
    m.def("load_image", &load_image_py,
          py::arg("path"),
          "Load image as numpy.ndarray (uint8, HxW or HxWx3) with zero-copy.");

    m.def("save_image", &save_image_py,
          py::arg("path"), py::arg("img"),
          "Save numpy.ndarray (uint8, HxW or HxWx3) to file (.png/.jpg).");

    // mean_filter
    m.def("mean_filter",
          [](const py::array& src,
             int ksize,
             const std::string& backend,
             const std::string& border,
             uint8_t border_value)
          {
              return wrap_filter(
                  src, ksize, backend, border, border_value,
                  [](const ImageU8& in, int k,
                     Border b, Backend be, uint8_t bv) {
                      return pf::mean_filter(in, k, b, be, bv);
                  });
          },
          py::arg("img"),
          py::arg("ksize"),
          py::arg("backend") = "auto",
          py::arg("border") = "reflect",
          py::arg("border_value") = 0,
          "Mean (box) filter with selectable backend/border.");

    // gaussian_filter
    m.def("gaussian_filter",
          [](const py::array& src,
             float sigma,
             const std::string& backend,
             const std::string& border,
             uint8_t border_value)
          {
              return wrap_filter(
                  src, sigma, backend, border, border_value,
                  [](const ImageU8& in, float s,
                     Border b, Backend be, uint8_t bv) {
                      return pf::gaussian_filter(in, s, b, be, bv);
                  });
          },
          py::arg("img"),
          py::arg("sigma"),
          py::arg("backend") = "auto",
          py::arg("border")  = "reflect",
          py::arg("border_value") = 0,
          "Gaussian filter with selectable backend/border.");

    // median_filter
    m.def("median_filter",
          [](const py::array& src,
             int ksize,
             const std::string& backend,
             const std::string& border,
             uint8_t border_value)
          {
              return wrap_filter(
                  src, ksize, backend, border, border_value,
                  [](const ImageU8& in, int k,
                     Border b, Backend be, uint8_t bv) {
                      return pf::median_filter(in, k, b, be, bv);
                  });
          },
          py::arg("img"),
          py::arg("ksize"),
          py::arg("backend") = "auto",
          py::arg("border")  = "reflect",
          py::arg("border_value") = 0,
          "Median filter with selectable backend/border.");

    // bilateral_filter（參數比較多，就不套 wrap_filter 模板了）
    m.def("bilateral_filter",
          [](const py::array& src,
             int ksize,
             float sigma_color,
             float sigma_space,
             const std::string& backend,
             const std::string& border,
             uint8_t border_value)
          {
              ImageU8 in  = numpy_to_imageu8_zero_copy(src);  // zero-copy in
              Border  b   = parse_border(border);
              Backend be  = parse_backend(backend);
              ImageU8 out = pf::bilateral_filter(in, ksize,
                                                 sigma_color, sigma_space,
                                                 b, be, border_value);
              return imageu8_to_numpy(out);                   // zero-copy out
          },
          py::arg("img"),
          py::arg("ksize"),
          py::arg("sigma_color"),
          py::arg("sigma_space"),
          py::arg("backend") = "auto",
          py::arg("border")  = "reflect",
          py::arg("border_value") = 0,
          "Bilateral filter with selectable backend/border.");


    // -------------------- Color & tone (Week4) --------------------
    m.def(
        "to_grayscale",
        [](const py::array& src,
           const std::string& backend) {
            ImageU8 in  = numpy_to_imageu8_zero_copy(src);
            Backend be  = parse_backend(backend);
            ImageU8 out = pf::to_grayscale(in, be);
            return imageu8_to_numpy(out);
        },
        py::arg("img"),
        py::arg("backend") = "auto",
        "Convert RGB image to grayscale (returns HxW array)."
    );

    m.def(
        "invert",
        [](const py::array& src,
           const std::string& backend) {
            ImageU8 in  = numpy_to_imageu8_zero_copy(src);
            Backend be  = parse_backend(backend);
            ImageU8 out = pf::invert(in, be);
            return imageu8_to_numpy(out);
        },
        py::arg("img"),
        py::arg("backend") = "auto",
        "Invert pixel values: v -> 255 - v."
    );

    m.def(
        "sepia",
        [](const py::array& src,
           const std::string& backend) {
            ImageU8 in  = numpy_to_imageu8_zero_copy(src);
            Backend be  = parse_backend(backend);
            ImageU8 out = pf::sepia(in, be);
            return imageu8_to_numpy(out);
        },
        py::arg("img"),
        py::arg("backend") = "auto",
        "Apply sepia tone effect (RGB only)."
    );

    m.def(
        "adjust_brightness_contrast",
        [](const py::array& src,
           float alpha,
           float beta,
           const std::string& backend) {
            ImageU8 in  = numpy_to_imageu8_zero_copy(src);
            Backend be  = parse_backend(backend);
            ImageU8 out = pf::adjust_brightness_contrast(in, alpha, beta, be);
            return imageu8_to_numpy(out);
        },
        py::arg("img"),
        py::arg("alpha"),
        py::arg("beta"),
        py::arg("backend") = "auto",
        "Adjust brightness and contrast: new = alpha * old + beta."
    );

    m.def(
        "gamma_correct",
        [](const py::array& src,
           float gamma,
           const std::string& backend) {
            ImageU8 in  = numpy_to_imageu8_zero_copy(src);
            Backend be  = parse_backend(backend);
            ImageU8 out = pf::gamma_correct(in, gamma, be);
            return imageu8_to_numpy(out);
        },
        py::arg("img"),
        py::arg("gamma"),
        py::arg("backend") = "auto",
        "Gamma correction: new = 255 * (old/255)^gamma."
    );

    // -------------------- Effects (Week5) --------------------
    m.def(
        "sharpen",
        [](const py::array& src,
           float amount,
           const std::string& backend) {
            using namespace pfpy;
            ImageU8 in  = numpy_to_imageu8_zero_copy(src);
            Backend be  = parse_backend(backend);
            ImageU8 out = pf::sharpen(in, amount, be);
            return imageu8_to_numpy(out);
        },
        py::arg("img"),
        py::arg("amount") = 1.0f,
        py::arg("backend") = "auto",
        "Sharpen the image with a simple 3x3 kernel."
    );

    m.def(
        "emboss",
        [](const py::array& src,
           float strength,
           const std::string& backend) {
            using namespace pfpy;
            ImageU8 in  = numpy_to_imageu8_zero_copy(src);
            Backend be  = parse_backend(backend);
            ImageU8 out = pf::emboss(in, strength, be);
            return imageu8_to_numpy(out);
        },
        py::arg("img"),
        py::arg("strength") = 1.0f,
        py::arg("backend") = "auto",
        "Emboss effect to give a relief-style shading."
    );

    m.def(
        "cartoonize",
        [](const py::array& src,
           float sigma_space,
           int edge_threshold,
           const std::string& backend) {
            using namespace pfpy;
            ImageU8 in  = numpy_to_imageu8_zero_copy(src);
            Backend be  = parse_backend(backend);
            ImageU8 out = pf::cartoonize(
                in,
                sigma_space,
                static_cast<std::uint8_t>(edge_threshold),
                be
            );
            return imageu8_to_numpy(out);
        },
        py::arg("img"),
        py::arg("sigma_space") = 2.0f,
        py::arg("edge_threshold") = 40,
        py::arg("backend") = "auto",
        "Simple cartoon effect: smooth + edge lines + color quantization."
    );

    // ------------------------------------------------------------
    // Geometry (Week6)
    // ------------------------------------------------------------
    m.def(
        "resize",
        [](const py::array& src,
           int new_h,
           int new_w,
           const std::string& backend) {
            using namespace pfpy;
            ImageU8 in  = numpy_to_imageu8_zero_copy(src);
            Backend be  = parse_backend(backend);
            ImageU8 out = pf::resize(in, new_h, new_w, be);
            return imageu8_to_numpy(out);
        },
        py::arg("img"),
        py::arg("height"),
        py::arg("width"),
        py::arg("backend") = "auto",
        "Resize image to (height, width) using bilinear interpolation."
    );

    m.def(
        "flip_horizontal",
        [](const py::array& src,
           const std::string& backend) {
            using namespace pfpy;
            ImageU8 in  = numpy_to_imageu8_zero_copy(src);
            Backend be  = parse_backend(backend);
            ImageU8 out = pf::flip_horizontal(in, be);
            return imageu8_to_numpy(out);
        },
        py::arg("img"),
        py::arg("backend") = "auto",
        "Flip image horizontally."
    );

    m.def(
        "flip_vertical",
        [](const py::array& src,
           const std::string& backend) {
            using namespace pfpy;
            ImageU8 in  = numpy_to_imageu8_zero_copy(src);
            Backend be  = parse_backend(backend);
            ImageU8 out = pf::flip_vertical(in, be);
            return imageu8_to_numpy(out);
        },
        py::arg("img"),
        py::arg("backend") = "auto",
        "Flip image vertically."
    );

    m.def(
        "crop",
        [](const py::array& src,
           int y,
           int x,
           int h,
           int w,
           const std::string& backend) {
            using namespace pfpy;
            ImageU8 in  = numpy_to_imageu8_zero_copy(src);
            Backend be  = parse_backend(backend);
            ImageU8 out = pf::crop(in, y, x, h, w, be);
            return imageu8_to_numpy(out);
        },
        py::arg("img"),
        py::arg("y"),
        py::arg("x"),
        py::arg("height"),
        py::arg("width"),
        py::arg("backend") = "auto",
        "Crop a (height, width) region starting from (y, x)."
    );

    m.def(
        "rotate",
        [](const py::array& src,
           float angle_deg,
           const std::string& backend) {
            using namespace pfpy;
            ImageU8 in  = numpy_to_imageu8_zero_copy(src);
            Backend be  = parse_backend(backend);
            ImageU8 out = pf::rotate(in, angle_deg, be);
            return imageu8_to_numpy(out);
        },
        py::arg("img"),
        py::arg("angle_deg"),
        py::arg("backend") = "auto",
        "Rotate image by angle_deg (center-based), output size same as input."
    );
    // arr (numpy) -> ImageU8 (zero-copy) -> numpy (zero-copy)
    m.def("_debug_zerocopy_roundtrip_u8", [](py::array arr) {
        // 你已經有這兩個 helper：numpy_to_imageu8_zero_copy / imageu8_to_numpy
        ImageU8 img = numpy_to_imageu8_zero_copy(arr);
        return imageu8_to_numpy(img);
    });
}
