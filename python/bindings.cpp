#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "pixfoundry/image.hpp"
#include "pixfoundry/filters.hpp"
#include <vector>
#include <cstdint> 

namespace py = pybind11;
using namespace pf;

// 將 numpy.ndarray (uint8, C-contiguous, HxW 或 HxWx3) 以零拷貝轉為 ImageU8
static ImageU8 numpy_to_imageu8_zero_copy(const py::array& array)
{
    py::buffer_info info = array.request();

    if (info.ndim != 2 && info.ndim != 3)
        throw std::runtime_error("expected HxW or HxWxC array");

    if (info.itemsize != 1)
        throw std::runtime_error("expected dtype=uint8");

    int h = static_cast<int>(info.shape[0]);
    int w = static_cast<int>(info.shape[1]);
    int c = (info.ndim == 3) ? static_cast<int>(info.shape[2]) : 1;

    if (c != 1 && c != 3)
        throw std::runtime_error("expected 1 or 3 channels");

    // 僅接受 C 連續（零拷貝前提）
    if (info.ndim == 2) {
        if (!(info.strides[0] == static_cast<ssize_t>(w) && info.strides[1] == 1))
            throw std::runtime_error("expected C-contiguous array (HxW)");
    } else {
        if (!(info.strides[0] == static_cast<ssize_t>(w * c) &&
              info.strides[1] == static_cast<ssize_t>(c) &&
              info.strides[2] == 1))
            throw std::runtime_error("expected C-contiguous array (HxWxC)");
    }

    // 取得資料指標
    auto* ptr = static_cast<uint8_t*>(info.ptr);

    // 以 py::object 持有原始 numpy 陣列，確保其生命週期 >= ImageU8
    py::object owner = array;  // 增加一個 Python 引用

    // 建立 shared_ptr 指向 numpy 的記憶體；deleter 不釋放資料，只捕獲 owner
    // 當 shared_ptr 被銷毀時，lambda 也會被銷毀，owner 的析構才會 dec_ref
    std::shared_ptr<uint8_t[]> sp(ptr, [owner](uint8_t*) mutable {
        // 不釋放 ptr（numpy 擁有）；僅靠 owner 的生命期維持記憶體
        // 空 body 即可；owner 在此 lambda 析構時自動 dec_ref
    });

    return ImageU8(h, w, c, std::move(sp));
}

// 將 ImageU8 零拷貝包成 numpy.ndarray（附帶 capsule 以延長 shared_ptr 生命週期）
static py::array imageu8_to_numpy(const ImageU8& img)
{
    const int h = img.h();
    const int w = img.w();
    const int c = img.c();

    if (img.empty())
        throw std::runtime_error("Image is empty");

    // 形狀與步幅（C 連續）
    std::vector<ssize_t> shape;
    std::vector<ssize_t> strides;

    if (c == 1) {
        shape   = { h, w };
        strides = { static_cast<ssize_t>(w), 1 };
    } else {
        shape   = { h, w, c };
        strides = { static_cast<ssize_t>(w * c), static_cast<ssize_t>(c), 1 };
    }

    // 建一個 capsule，裡面放一個 new 出來的 shared_ptr 副本：
    // 當 numpy 物件被 GC 掉時，capsule 會被析構，shared_ptr 計數-1，
    // 原本在 C++ 的 shared_ptr 就能一起管理生命週期（零拷貝）
    auto sp_copy = new std::shared_ptr<uint8_t[]>(img.shared());
    py::capsule base(sp_copy, [](void *p){
        delete reinterpret_cast<std::shared_ptr<uint8_t[]>*>(p);
    });

    return py::array(
        py::dtype::of<uint8_t>(),
        shape,
        strides,
        img.shared().get(), // data pointer
        base                 // base to keep memory alive
    );
}

static py::array load_image_py(const std::string& path)
{
    ImageU8 im = load_image_u8(path);  // 零拷貝從 stb → ImageU8
    return imageu8_to_numpy(im);       // 再零拷貝包成 numpy
}

static void save_image_py(const std::string& path, py::array array)
{
    // 接受 HxW 或 HxWxC（C 要是 1 或 3）
    py::buffer_info info = array.request();
    if (info.ndim != 2 && info.ndim != 3)
        throw std::runtime_error("save_image expects HxW or HxWxC array");

    if (info.itemsize != 1)
        throw std::runtime_error("save_image expects dtype=uint8");

    int h = static_cast<int>(info.shape[0]);
    int w = static_cast<int>(info.shape[1]);
    int c = (info.ndim == 3) ? static_cast<int>(info.shape[2]) : 1;

    if (c != 1 && c != 3)
        throw std::runtime_error("save_image expects 1 or 3 channels");

    // 檢查是否 C 連續（簡單檢查）
    if (info.ndim == 2) {
        // HxW: strides = [W, 1]
        if (!(info.strides[0] == static_cast<ssize_t>(w) && info.strides[1] == 1))
            throw std::runtime_error("save_image expects a contiguous array");
    } else {
        // HxWxC: strides = [W*C, C, 1]
        if (!(info.strides[0] == static_cast<ssize_t>(w * c) &&
              info.strides[1] == static_cast<ssize_t>(c) &&
              info.strides[2] == 1))
            throw std::runtime_error("save_image expects a contiguous array");
    }

    auto* ptr = static_cast<const uint8_t*>(info.ptr);
    save_image_u8(path, ptr, h, w, c);
}

PYBIND11_MODULE(_core, m) {
    m.doc() = "PixFoundry core (zero-copy IO)";

    m.def("load_image", &load_image_py,
          "Load image as numpy.ndarray (uint8, HxW or HxWx3) with zero-copy.");

    m.def("save_image", &save_image_py,
          "Save numpy.ndarray (uint8, HxW or HxWx3) to file (.png/.jpg).");
    m.def("mean_filter",
        [](py::array src, int ksize) {
            // 零拷貝取得輸入
            ImageU8 in = numpy_to_imageu8_zero_copy(src);
            auto out = pf::mean_filter(in, ksize);
            return imageu8_to_numpy(out);  // 輸出仍然零拷貝
        },
        "Mean blur with odd ksize >=3");

    m.def("gaussian_filter",
        [](py::array src, float sigma) {
            ImageU8 in = numpy_to_imageu8_zero_copy(src);  // 零拷貝輸入
            auto out = pf::gaussian_filter(in, sigma);
            return imageu8_to_numpy(out);                  // 零拷貝輸出
        },
        "Gaussian blur with sigma > 0");
}
