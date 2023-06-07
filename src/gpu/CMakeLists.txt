add_executable(gpu ${CPU_SRC} ${CPU_HEADERS})

set(GPU_SRC "kernel.cu")
target_sources(gpu PRIVATE ${GPU_SRC})

if (NOT APPLE)
    set_property(TARGET gpu PROPERTY CUDA_ARCHITECTURES native)
    find_package(CUDAToolkit REQUIRED)
endif()

find_package(OpenCV REQUIRED)
target_link_libraries(gpu PRIVATE ${CUDA_LIBRARIES} ${OpenCV_LIBS})
include_directories(${CUDA_INCLUDE_DIRS} ${OPENCV_INCLUDE_DIRS})
