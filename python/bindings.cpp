#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "pixfoundry/image.hpp"
#include <vector>
#include <cstdint> 

namespace py = pybind11;
using namespace pf;

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
}
