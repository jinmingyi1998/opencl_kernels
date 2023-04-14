cmake_minimum_required(VERSION 3.16)

set(TEST_SRC "${CMAKE_CURRENT_SOURCE_DIR}/test_main.cc")
macro(add_test name)
    list(APPEND TEST_SRC ${CMAKE_CURRENT_SOURCE_DIR}/test_${name}.cc)
endmacro()

add_test(binary_op)
add_test(image_buffer)
add_test(ctx)

add_executable(opencl_kernel_test ${TEST_SRC})
target_include_directories(opencl_kernel_test PUBLIC ${CMAKE_SOURCE_DIR}/csrc)
target_link_libraries(opencl_kernel_test PUBLIC ocl_demo ocl_helper GTest::gtest_main)
