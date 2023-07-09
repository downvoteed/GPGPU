add_executable(gpu ${CPU_SRC} ${CPU_HEADERS})

set(GPU_SRC "kernel.cu" "frame-helper.cu" "segmentation-helper.cu")
target_sources(gpu PRIVATE ${GPU_SRC})

find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED COMPONENTS program_options log thread)


if (NOT APPLE)
    set_property(TARGET gpu PROPERTY CUDA_ARCHITECTURES
    # all "recent" sm_compute
	60 61 62 70 72 75 80 86 
    )
    find_package(CUDAToolkit REQUIRED)
endif()

#optimiziation flags
target_compile_options(gpu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       --use_fast_math
                       >)



target_link_libraries(gpu PRIVATE ${CUDA_LIBRARIES} ${OpenCV_LIBS} ${Boost_LIBRARIES})
target_include_directories(gpu PUBLIC ${OpenCV_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})
target_include_directories(gpu PUBLIC ../../include)
