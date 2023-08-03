cmake_minimum_required(VERSION 3.24)
project(oclk_python LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 14)

set(Python_ROOT_DIR /home/jimmy/miniconda3/bin)
find_package(Python 3.10 COMPONENTS Interpreter Development REQUIRED)
message(STATUS "${Python_INTERPRETER_ID} ${Python_VERSION}")
message(STATUS "    libraries: ${Python_LIBRARY_DIRS}")
message(STATUS "    includes: ${Python_INCLUDE_DIRS}")

add_subdirectory(pybind11)
add_subdirectory(src)
add_subdirectory(csrc)
