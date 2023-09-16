cmake_minimum_required(VERSION 3.14.0 FATAL_ERROR)
project(padim)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")


find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)
find_package(TorchVision REQUIRED)

set(SAMPLE_SOURCES train.cpp)
set(TARGET_NAME train)

set(SAMPLE_DEP_LIBS
    nvinfer
    nvonnxparser
)

include_directories("./commons")

add_executable(${TARGET_NAME} ${SAMPLE_SOURCES})
target_link_libraries(${TARGET_NAME} ${TORCH_LIBRARIES})
target_link_libraries(${TARGET_NAME} ${OpenCV_LIBS})
target_link_libraries(${TARGET_NAME} TorchVision::TorchVision)
target_link_libraries(${TARGET_NAME} ${SAMPLE_DEP_LIBS})

set_property(TARGET ${TARGET_NAME} PROPERTY CXX_STANDARD 17)