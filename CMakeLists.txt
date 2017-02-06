cmake_minimum_required(VERSION 3.4)
project(lda)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O2")

# if you're compiling with cygwin on windows and the python wrapper is failing
# add the flags -static-libgcc -static-libstdc++. This blows up the executable size
# but should solve the linking issues. Otherwise figure out what's not linking and add
# that to your PATH.

set(dir ${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${dir}/cpp/build")

add_subdirectory("${PROJECT_SOURCE_DIR}/cpp/src")
include_directories("${PROJECT_SOURCE_DIR}/cpp/src")

add_executable(lda ${PROJECT_SOURCE_DIR}/cpp/main.cpp)
target_link_libraries(lda lib)

###########
# Testing #
###########

add_subdirectory("${PROJECT_SOURCE_DIR}/cpp/ext/gtest")
enable_testing()

include_directories(${GTEST_INCLUDE_DIRS})

add_subdirectory("${PROJECT_SOURCE_DIR}/cpp/test")
include_directories("${PROJECT_SOURCE_DIR}/cpp/test")

add_executable(testBasic ${TEST_FILES})
add_dependencies(testBasic googletest)

target_link_libraries(testBasic lib)
target_link_libraries(testBasic ${GTEST_LIBS_DIR}/libgtest.a ${GTEST_LIBS_DIR}/libgtest_main.a)
target_link_libraries(testBasic ${CMAKE_THREAD_LIBS_INIT})

add_test(test1 testBasic)