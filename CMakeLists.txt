cmake_minimum_required(VERSION 3.5...3.27)
project(oclk_python LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Os -static-libstdc++ -static-libgcc")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Os -static-libgcc")

add_definitions(-DOCLK_VERSION_INFO=${OCLK_VERSION_INFO})

include_directories(oclk/third_party/OpenCL/include)
link_directories(oclk/third_party/OpenCL/lib/${CMAKE_SYSTEM_PROCESSOR})

include_directories(include)

# set Python_ROOT_DIR to specify the python
find_package(
    Python 3.6
    COMPONENTS Interpreter Development.Module
    REQUIRED)
message(STATUS "${Python_INTERPRETER_ID} ${Python_VERSION}")
message(STATUS "    libraries: ${Python_LIBRARY_DIRS}")
message(STATUS "    includes: ${Python_INCLUDE_DIRS}")

add_subdirectory(pybind11)
add_subdirectory(csrc)
