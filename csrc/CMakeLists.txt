cmake_minimum_required(VERSION 3.24)

include_directories(${PROJECT_SOURCE_DIR}/src)

pybind11_add_module(oclk_C oclk_pyapi.cc)

target_link_libraries(oclk_C PUBLIC ocl_base)