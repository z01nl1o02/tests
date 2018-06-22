cmake_minimum_required(VERSION 2.6)

project(mxnetcpp)



find_package(OPENCV REQUIRED)

include_directories(. ${OPENCV_INCLUDE_DIR})

aux_source_directory(. SRC_LIST)

find_library(LIBMXNET libmxnet.lib mxnet/)

add_executable(${PROJECT_NAME} ${SRC_LIST})

target_link_libraries(${PROJECT_NAME}
${OPENCV_LIBRARIES} ${LIBMXNET})