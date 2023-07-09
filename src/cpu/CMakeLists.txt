set(CPU_SRC
	"color-helper.cc"
	"frame-helper.cc"
	"segmentation-helper.cc"
	"video-processor.cc"
	"texture-helper.cc"
	"webcam-processor.cc"
	"logger.cc"
	"main.cc"
)

add_executable(cpu ${CPU_SRC})

find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED COMPONENTS program_options log thread)

target_link_libraries(cpu ${OpenCV_LIBS} ${Boost_LIBRARIES})
target_include_directories(cpu PUBLIC ${OpenCV_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})
target_include_directories(cpu PUBLIC ../../include)
