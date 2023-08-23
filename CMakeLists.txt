cmake_minimum_required(VERSION 3.14.0 FATAL_ERROR)
project(learningDL)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")


find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)
find_package(TorchVision REQUIRED)
message(STATUS "    include path:${OpenCV_INCLUDE_DIRS}")
set(SAMPLES_LIST 
unet  
)

foreach(f ${SAMPLES_LIST})
    add_subdirectory(${f})
endforeach(f SAMPLES_LIST)