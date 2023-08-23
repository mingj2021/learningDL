find_package(yaml-cpp REQUIRED)

set(SAMPLE_SOURCES train.cpp)
set(TARGET_NAME train)

add_executable(${TARGET_NAME} ${SAMPLE_SOURCES})
target_link_libraries(${TARGET_NAME} ${TORCH_LIBRARIES})
target_link_libraries(${TARGET_NAME} ${OpenCV_LIBS})
target_link_libraries(${TARGET_NAME} yaml-cpp)
target_link_libraries(${TARGET_NAME} TorchVision::TorchVision)
# target_link_libraries(${TARGET_NAME} ${SAMPLE_DEP_LIBS})

set_property(TARGET ${TARGET_NAME} PROPERTY CXX_STANDARD 17)


set(SAMPLE_SOURCES detect.cpp)
set(TARGET_NAME detect)

add_executable(${TARGET_NAME} ${SAMPLE_SOURCES})
target_link_libraries(${TARGET_NAME} ${TORCH_LIBRARIES})
target_link_libraries(${TARGET_NAME} ${OpenCV_LIBS})
target_link_libraries(${TARGET_NAME} yaml-cpp)
target_link_libraries(${TARGET_NAME} TorchVision::TorchVision)
# target_link_libraries(${TARGET_NAME} ${SAMPLE_DEP_LIBS})

set_property(TARGET ${TARGET_NAME} PROPERTY CXX_STANDARD 17)


set(SAMPLE_SOURCES val.cpp)
set(TARGET_NAME val)

add_executable(${TARGET_NAME} ${SAMPLE_SOURCES})
target_link_libraries(${TARGET_NAME} ${TORCH_LIBRARIES})
target_link_libraries(${TARGET_NAME} ${OpenCV_LIBS})
target_link_libraries(${TARGET_NAME} yaml-cpp)
target_link_libraries(${TARGET_NAME} TorchVision::TorchVision)
# target_link_libraries(${TARGET_NAME} ${SAMPLE_DEP_LIBS})

set_property(TARGET ${TARGET_NAME} PROPERTY CXX_STANDARD 17)