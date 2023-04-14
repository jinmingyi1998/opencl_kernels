cmake_minimum_required(VERSION 3.16)

project(opencl_demo)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fopenmp -lm")
find_library(OpenCL name OpenCL PATHS "/usr/lib/x86_64-linux-gnu")
message("${OpenCL}")
find_package(OpenMP REQUIRED)

# third party
include(FetchContent)
find_package(GTest 1.13.0)
if(GTest_FOUND)
    message("found googletest ${googletest}")
else()
    message(
        "could not find googletest, clone from https://github.com/google/googletest.git v1.13.0"
    )
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.13.0)
    FetchContent_MakeAvailable(googletest)
endif()
find_package(gflags 2.2.2)
if(gflags_FOUND)
    message("found gflags ${gflags}")
else()
    message(
        "not found gflags, clone from github: https://github.com/gflags/gflags.git v2.2.2"
    )
    FetchContent_Declare(
        gflags
        GIT_REPOSITORY https://github.com/gflags/gflags.git
        GIT_TAG v2.2.2)
    FetchContent_MakeAvailable(gflags)
endif()
find_package(glog 0.6.0)
if(glog_FOUND)
    message("found glog ${glog}")
else()
    message(
        "not found glog, clone from https://github.com/google/glog.git v0.6.0")
    FetchContent_Declare(
        glog
        GIT_REPOSITORY https://github.com/google/glog.git
        GIT_TAG v0.6.0)
    FetchContent_MakeAvailable(glog)
endif()

# FetchContent_Declare( yaml-cpp GIT_REPOSITORY
# git@github.com:jbeder/yaml-cpp.git GIT_TAG yaml-cpp-0.7.0 )

option(DISABLE_BUILD_TIMER "disable timer" off)
option(DISABLE_BUILD_TEST "disable tests" off)
option(DISABLE_BUILD_BENCHMARK "disable benchnmark" off)

if(NOT DISABLE_TIMER)
    add_definitions(-DENABLE_TIMER)
endif()

# glob source files
aux_source_directory(csrc CSRC_DIR)
aux_source_directory(csrc/utils CSRC_UTILS_DIR)
list(APPEND CSRC_DIR ${CSRC_UTILS_DIR})
add_library(
    ocl_demo_base STATIC
    csrc/ocl_manager.cc csrc/kernel_wrapper.cc csrc/utils/buffer_wrapper.cc
    csrc/utils/image_wrapper.cc)
target_link_libraries(ocl_demo_base PUBLIC OpenCL glog::glog gflags)

add_library(ocl_demo INTERFACE)
target_link_libraries(ocl_demo INTERFACE ocl_demo_base)

macro(add_op name)
    add_library(ocl_op_${name} STATIC csrc/operator/${name}.cc)
    target_link_libraries(ocl_op_${name} PUBLIC ocl_demo_base)
    target_link_libraries(ocl_demo INTERFACE ocl_op_${name})
endmacro()

add_op(binary_op)
add_op(buffer_image_converter)

if(NOT DISABLE_BUILD_TEST OR NOT DISABLE_BUILD_BENCHMARK)
    add_library(ocl_helper STATIC helper/helper.cc helper/timer.cc)
    target_link_libraries(ocl_helper PRIVATE ocl_demo)
    target_include_directories(ocl_helper PRIVATE ${CMAKE_SOURCE_DIR})
    if(NOT DISABLE_BUILD_TEST)
        add_subdirectory(tests)
    endif()
    if(NOT DISABLE_BUILD_BENCHMARK)
        add_subdirectory(benchmark)
    endif()
endif()
