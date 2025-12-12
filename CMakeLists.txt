cmake_minimum_required(VERSION 3.18)
project(pixfoundry LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# pybind11
include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG v2.12.0
)
FetchContent_MakeAvailable(pybind11)

pybind11_add_module(_core
  python/bindings.cpp
  src/core.cpp
  src/filters.cpp
  src/color.cpp
  src/effects.cpp
  src/geometry.cpp
)

target_include_directories(_core
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/stb
)

# 讓 scikit-build-core 把 .so 安裝到 pixfoundry/ 底下
install(TARGETS _core LIBRARY DESTINATION pixfoundry)