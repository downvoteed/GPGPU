cmake_minimum_required(VERSION 3.0.0)
project(
    gpgpu
    VERSION 0.1.0
    DESCRIPTION "GPGPU"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

subdirs(src)

