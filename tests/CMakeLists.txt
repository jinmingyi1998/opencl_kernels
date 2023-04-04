cmake_minimum_required(VERSION 3.16)

aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} TEST_SRC_FILES)
add_executable(opencl_kernel_test
        ${TEST_SRC_FILES})

message("CMAKE_CACHEFILE_DIR= ${FETCHCONTENT_BASE_DIR}")
target_include_directories(opencl_kernel_test
        PUBLIC
        ${CMAKE_SOURCE_DIR}/csrc
        )
target_link_libraries(opencl_kernel_test
        PUBLIC
        ocl_demo
        glog::glog
        gflags::gflags
        GTest::gtest_main
#        yaml-cpp
        )