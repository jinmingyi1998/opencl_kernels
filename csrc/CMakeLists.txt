cmake_minimum_required(VERSION 3.5...3.27)

pybind11_add_module(
    oclk_C
    oclk_pyapi.cc
    common.cc
    kernel.cc
    runner.cc
    timer.cc)

target_link_libraries(oclk_C PUBLIC OpenCL)
