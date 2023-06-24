file(GLOB CPU_SRC
    "*.cc"
)

file(GLOB CPU_HEADERS
    "*.hh"
)

add_executable(cpu ${CPU_SRC} ${CPU_HEADERS})

find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED COMPONENTS program_options log thread)

target_link_libraries(cpu ${OpenCV_LIBS} ${Boost_LIBRARIES})
target_include_directories(cpu PUBLIC ${OpenCV_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})
target_include_directories(cpu PUBLIC ../../include)
