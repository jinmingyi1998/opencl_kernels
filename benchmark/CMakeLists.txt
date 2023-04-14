macro(add_bench name)
    add_executable(bench_${name} ${CMAKE_CURRENT_SOURCE_DIR}/bench_${name}.cc)
    target_include_directories(bench_${name} PUBLIC ${CMAKE_SOURCE_DIR}/csrc
                                                    ${CMAKE_SOURCE_DIR}/include
            ${CMAKE_SOURCE_DIR})
    target_link_libraries(bench_${name} PUBLIC ocl_demo ocl_helper)
endmacro(add_bench)

add_bench(binary_op)
