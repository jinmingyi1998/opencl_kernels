cmake_minimum_required(VERSION 3.16)
project(oclk)

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Os")

find_library(OpenCL name OpenCL)

include(FetchContent)
set(FETCH_LIST "")

FetchContent_Declare(
        glog
        GIT_REPOSITORY https://github.com/google/glog.git
        GIT_TAG v0.6.0
        FIND_PACKAGE_ARGS NAMES glog 0.6.0 COMPONENTS glog::glog)
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
message(STATUS "FOUND GTest ${googletest_VERSION} ${googletest_DIR}")

include_directories(${glog_SOURCE_DIR}/src ${glog_BINARY_DIR})
set(CSRC_FILE runner.cc timer.cc kernel.cc common.cc)
add_library(ocl_base SHARED ${CSRC_FILE})
target_link_libraries(ocl_base PUBLIC
        libOpenCL.so
        glog::glog)
if (OpenCV_FOUND)
    target_link_libraries(ocl_base PUBLIC ${OpenCV_LIBS})
endif ()
