cmake_minimum_required(VERSION 2.6)

project(calc)

include_directories(.)

aux_source_directory(. SRC_LIST)

add_library(${PROJECT_NAME} SHARED ${SRC_LIST})

