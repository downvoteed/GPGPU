add_executable(gpu ${CPU_SRC} ${CPU_HEADERS})

set(GPU_SRC "kernel.cu")
target_sources(gpu PRIVATE ${GPU_SRC})

set_property(TARGET gpu PROPERTY CUDA_ARCHITECTURES native)

find_package(CUDAToolkit REQUIRED)
target_link_libraries(gpu PRIVATE ${CUDA_LIBRARIES})
