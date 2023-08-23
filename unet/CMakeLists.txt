# set(SAMPLE_SOURCES train.cpp)
# set(TARGET_NAME train)

# add_executable(${TARGET_NAME} ${SAMPLE_SOURCES})
# target_link_libraries(${TARGET_NAME} ${TORCH_LIBRARIES})
# target_link_libraries(${TARGET_NAME} ${OpenCV_LIBS})
# target_link_libraries(${TARGET_NAME} TorchVision::TorchVision)

# set_property(TARGET ${TARGET_NAME} PROPERTY CXX_STANDARD 17)

# set(SAMPLE_SOURCES test.cpp)
# set(TARGET_NAME test)

# add_executable(${TARGET_NAME} ${SAMPLE_SOURCES})
# target_link_libraries(${TARGET_NAME} ${TORCH_LIBRARIES})
# target_link_libraries(${TARGET_NAME} ${OpenCV_LIBS})
# target_link_libraries(${TARGET_NAME} TorchVision::TorchVision)

# set_property(TARGET ${TARGET_NAME} PROPERTY CXX_STANDARD 17)


set(SAMPLE_SOURCES mult_class_train.cpp)
set(TARGET_NAME mult_class_train)

add_executable(${TARGET_NAME} ${SAMPLE_SOURCES})
target_link_libraries(${TARGET_NAME} ${TORCH_LIBRARIES})
target_link_libraries(${TARGET_NAME} ${OpenCV_LIBS})
target_link_libraries(${TARGET_NAME} TorchVision::TorchVision)

set_property(TARGET ${TARGET_NAME} PROPERTY CXX_STANDARD 17)


set(SAMPLE_SOURCES mult_class_test.cpp)
set(TARGET_NAME mult_class_test)

add_executable(${TARGET_NAME} ${SAMPLE_SOURCES})
target_link_libraries(${TARGET_NAME} ${TORCH_LIBRARIES})
target_link_libraries(${TARGET_NAME} ${OpenCV_LIBS})
target_link_libraries(${TARGET_NAME} TorchVision::TorchVision)

set_property(TARGET ${TARGET_NAME} PROPERTY CXX_STANDARD 17)