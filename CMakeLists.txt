cmake_minimum_required(VERSION 3.16)

project(opencl_demo)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 11)

find_library(OpenCL name OpenCL PATHS "/usr/lib/x86_64-linux-gnu")
message("${OpenCL}")
find_package(OpenMP REQUIRED)

# third party
include(FetchContent)
FetchContent_Declare(
        googletest
        GIT_REPOSITORY git@github.com:google/googletest.git
        GIT_TAG v1.13.0
)
FetchContent_Declare(
        gflags
        GIT_REPOSITORY git@github.com:gflags/gflags.git
        GIT_TAG v2.2.2
)
FetchContent_Declare(
        glog
        GIT_REPOSITORY git@github.com:google/glog.git
        GIT_TAG v0.6.0
)
FetchContent_Declare(
        yamlcpp
        GIT_REPOSITORY git@github.com:jbeder/yaml-cpp.git
        GIT_TAG yaml-cpp-0.7.0
)

FetchContent_MakeAvailable(
        googletest
        gflags
        glog
#        yamlcpp
)

# glob source files
aux_source_directory(csrc CSRC_DIR)
aux_source_directory(csrc/utils CSRC_UTILS_DIR)
list(APPEND CSRC_DIR ${CSRC_UTILS_DIR})

add_library(ocl_demo STATIC ${CSRC_DIR})

target_link_libraries(ocl_demo PUBLIC
        OpenCL
        glog::glog
        gflags::gflags
#        yaml-cpp
        )

add_subdirectory(tests)
