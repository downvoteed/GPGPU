
file(GLOB STATS_SRC
    "*.cc"
)

target_sources (cpu PRIVATE ${STATS_SRC} ${STATS_HEADERS})
target_link_libraries(cpu ${OpenCV_LIBS} ${Boost_LIBRARIES})

target_include_directories(cpu PUBLIC ../../include)
