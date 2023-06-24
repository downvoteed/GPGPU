cmake_minimum_required(VERSION 3.21)

project(
    gpgpu
    VERSION 0.1.0
    DESCRIPTION "GPGPU"
    LANGUAGES CXX
)

if(POLICY CMP0079)
  cmake_policy(SET CMP0079 NEW)
endif()

if (NOT APPLE)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 14)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT MSVC AND NOT CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror -pedantic -Ofast")
elseif(WINDOWS)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4 -Ofast")
endif()

subdirs(src)
