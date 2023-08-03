cmake_minimum_required(VERSION 3.24)
project(oclk_python LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)

add_definitions(-DOCLK_VERSION_INFO=${OCLK_VERSION_INFO})

include_directories(oclk/third_party/OpenCL/include)
link_directories(oclk/third_party/OpenCL/lib/${CMAKE_SYSTEM_PROCESSOR})

find_package(Python 3.6 COMPONENTS Interpreter Development REQUIRED)
message(STATUS "${Python_INTERPRETER_ID} ${Python_VERSION}")
message(STATUS "    libraries: ${Python_LIBRARY_DIRS}")
message(STATUS "    includes: ${Python_INCLUDE_DIRS}")

add_subdirectory(pybind11)
add_subdirectory(src)
add_subdirectory(csrc)
