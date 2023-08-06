#ifndef OPENCL_KERNELS_OCLK_PYAPI_H
#define OPENCL_KERNELS_OCLK_PYAPI_H
//
// Created by jimmy on 23-7-28.
//
#include <iostream>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string.h>

#include "runner.h"
#include "spdlog/logger.h"

#ifndef OCLK_VERSION_INFO
#define OCLK_VERSION_INFO 0.0.0
#endif // OCLK_VERSION_INFO
#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)
const std::string module_version = MACRO_STRINGIFY(OCLK_VERSION_INFO);

unsigned long Init();
unsigned long LoadKernel(std::string &kernel_filename,
                         std::string &kernel_name,
                         std::string &compile_option_string);

inline void add_array_arg(std::string &arg_name,
                          pybind11::array &arr,
                          std::vector<oclk::ArgWrapper> &constants);

pybind11::list parse_args(pybind11::list arg_dicts,
                          std::vector<oclk::ArgWrapper> &args);

inline void add_array_arg(std::string &arg_name,
                          pybind11::array &arr,
                          std::vector<oclk::ArgWrapper> &constants);

pybind11::list parse_args(pybind11::list arg_dicts,
                          std::vector<oclk::ArgWrapper> &args);

/**
 * @param kwargs example input:
 *     {
 *         "kernel_name" str : "YOUR_KERNEL_NAME"
 *         "global_worksize" List[int]: [1123,132],
 *         "local_worksize" List[int]: [1,1],
 *         "input" List[Dict[str,Union[int,float,np.array]]] : [
 *             {'name':'name1', 'value' : value1} ,
 *             {'name':'name2', 'value' : value1} ,
 *             {'name':'name3', 'value' : value1} ,
 *             {'name':'name4', 'value' : value2}
 *         ],
 *         "output" List[str]: ["name3","name4"]
 *         "wait" bool : false (by default)
 *         "timer" :{
 *             "enable" : false,
 *             "warmup" : 0,
 *             "repeat" : 1,
 *             "name"   : "TIMER_NAME"
 *         }
 *     }
 * @return
 */
pybind11::list run_impl(pybind11::kwargs &kwargs);

PYBIND11_MODULE(oclk_C, m) {
    m.doc() =
        "OCLK(OpenCL Kernel) runner Python api"; // optional module docstring
    m.def("init", &Init, "");
    m.def("load_kernel", &LoadKernel, "");
    m.def("run", &run_impl, "run_impl_float");
    m.attr("__version__") = module_version;
}
#endif // OPENCL_KERNELS_OCLK_PYAPI_H
