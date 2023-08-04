cmake_minimum_required(VERSION 3.5...3.27)

find_package(glog 0.6.0 QUIET)
#find_package(OpenCV QUIET)

if (glog_FOUND)
    message(STATUS "FOUND glog ${glog_VERSION} ${glog_DIR}")
    add_definitions(-DWITH_GLOG=1)
else ()
    message(STATUS "not found glog, recommended use glog")
endif ()
pybind11_add_module(oclk_C oclk_pyapi.cc
        common.cc
        kernel.cc
        runner.cc
        timer.cc)
target_link_libraries(oclk_C PUBLIC OpenCL)
if (glog_FOUND)
    target_link_libraries(oclk_C PUBLIC
            glog::glog)
endif ()
