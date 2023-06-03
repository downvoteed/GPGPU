cmake_minimum_required(VERSION 3.12)
project(
    gpgpu
    VERSION 0.1.0
    DESCRIPTION "GPGPU"
    LANGUAGES CXX
)

find_package(OpenCV QUIET)

if(NOT OpenCV_FOUND)
    include(FetchContent)

    message(STATUS "OpenCV not found in the system, fetching and building...")

    FetchContent_Declare(
        opencv
        GIT_REPOSITORY https://github.com/opencv/opencv.git
        GIT_TAG        master
    )

    set(BUILD_LIST "core;")  # customize this list according to your needs
    set(BUILD_DOCS OFF CACHE BOOL "" FORCE)  # disable building of documentation
    set(BUILD_opencv_dnn OFF CACHE BOOL "" FORCE)  # disable building of DNN module

    FetchContent_MakeAvailable(opencv)

else()
    message(STATUS "Found OpenCV in the system")
endif()


if(POLICY CMP0079)
  cmake_policy(SET CMP0079 NEW)
endif()


find_package(Threads REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Boost COMPONENTS program_options REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIR})

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)

subdirs(src)
