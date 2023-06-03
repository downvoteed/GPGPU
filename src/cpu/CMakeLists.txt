file(GLOB CPU_SRC
    "*.cc"
)

file(GLOB CPU_HEADERS
    "*.hh"
)

add_executable(cpu ${CPU_SRC} ${CPU_HEADERS})
target_link_libraries(cpu ${OpenCV_LIBS} ${Boost_LIBRARIES})
target_link_libraries(cpu Threads::Threads)

target_include_directories(cpu PUBLIC ../../include)
