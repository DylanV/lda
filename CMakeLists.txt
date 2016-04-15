cmake_minimum_required(VERSION 3.5)
project(lda)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(dir ${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${dir}/cpp/build")

set(src ${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory("${PROJECT_SOURCE_DIR}/cpp/src")
add_executable(lda ${SOURCE} ${HEADERS})
