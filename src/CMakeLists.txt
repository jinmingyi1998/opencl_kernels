cmake_minimum_required(VERSION 3.16)
project(oclk)

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Os")
find_library(OpenCL name OpenCL PATHS "/usr/lib/x86_64-linux-gnu")
message(STATUS "${OpenCL}")

include(FetchContent)
set(FETCH_LIST "")
#FetchContent_Declare(
#    googletest
#    GIT_REPOSITORY https://github.com/google/googletest.git
#    GIT_TAG v1.13.0
#    FIND_PACKAGE_ARGS NAMES GTest 1.13.0)
#list(APPEND FETCH_LIST googletest)
FetchContent_Declare(
        gflags
        GIT_REPOSITORY https://github.com/gflags/gflags.git
        GIT_TAG v2.2.2
        FIND_PACKAGE_ARGS NAMES gflags 2.2.2)
list(APPEND FETCH_LIST gflags)

FetchContent_Declare(
        glog
        GIT_REPOSITORY https://github.com/google/glog.git
        GIT_TAG v0.6.0
        FIND_PACKAGE_ARGS NAMES glog 0.6.0)
list(APPEND FETCH_LIST glog)

message(STATUS "FetchContent_MakeAvailable ${FETCH_LIST}")
FetchContent_MakeAvailable(${FETCH_LIST})

find_package(OpenCV QUIET)
if (OpenCV_FOUND)
    message(STATUS "OpenCV library: ${OpenCV_INSTALL_PATH}")
    message(STATUS "    version: ${OpenCV_VERSION}")
    message(STATUS "    libraries: ${OpenCV_LIBS}")
    message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
    include_directories(${OpenCV_INCLUDE_DIRS})
else ()
    message(WARNING "OpenCV not found! You may set -DOpenCV_DIR to cmake")
endif ()

message(STATUS "FOUND glog ${glog_VERSION} ${glog_DIR}")
message(STATUS "FOUND gflags ${gflags_VERSION} ${gflags_DIR}")
message(STATUS "FOUND GTest ${googletest_VERSION} ${googletest_DIR}")

set(CSRC_FILE runner.cc timer.cc kernel.cc common.cc)
add_library(ocl_base SHARED ${CSRC_FILE})
target_link_libraries(ocl_base PUBLIC
        OpenCL
        glog::glog
        ${GFLAGS_LIBRARIES})
if (OpenCV_FOUND)
    target_link_libraries(ocl_base PUBLIC ${OpenCV_LIBS})
endif ()
