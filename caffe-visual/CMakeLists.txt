cmake_minimum_required(VERSION 2.6)

project(demo)

message(STATUS "export CAFFEROOT=caffe dir")

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Caffe REQUIRED)

include_directories(. ${CUDA_INCLUDE_DIRS} 
	${OpenCV_INCLUDE_DIRS} 
	${Caffe_INCLUDE_DIRS}   )
aux_source_directory(. SRC_LIST)


message(STATUS ${Caffe_INCLUDE_DIRS})
message(STATUS ${Caffe_LIBS})

find_library(LIB_BOOST_SYSTEM libboost_system.so /usr/lib/x86_64-linux-gnu/)
find_library(LIB_GLOG libglog.so /usr/lib/x86_64-linux-gnu/)


add_executable(${PROJECT_NAME} ${SRC_LIST})
target_link_libraries(${PROJECT_NAME} 
	${Caffe_LIBS} 
        ${LIB_BOOST_SYSTEM}
        ${LIB_GLOG}
	${CUDA_LIBRARIES} 
	${OpenCV_LIBS})

