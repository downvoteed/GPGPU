cmake_minimum_required(VERSION 3.21)

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

if (NOT APPLE)
    find_package(CUDA REQUIRED)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 14)
endif()

find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED COMPONENTS program_options log thread)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIRS})
include_directories(include)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)

set (CMAKE_CXX_FLAGS "-Ofast")
if(NOT MVSC)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror -pedantic")
else()
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
endif()

subdirs(src)
