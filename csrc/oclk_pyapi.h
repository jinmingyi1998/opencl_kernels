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

namespace py = pybind11;

#ifndef OCLK_VERSION_INFO
#define OCLK_VERSION_INFO 0.0.0
#endif // OCLK_VERSION_INFO
#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

#ifndef OCLK_KERNEL_ARG_CUSTOM_ARRAY_SIZE
#define OCLK_KERNEL_ARG_CUSTOM_ARRAY_SIZE 256
#endif

const std::string module_version = MACRO_STRINGIFY(OCLK_VERSION_INFO);

class RunnerReturn {
public:
    RunnerReturn() = default;

public:
    oclk::TimerResult timer_result = oclk::no_result;
    py::list results;
};

unsigned long Init();

unsigned long LoadKernel(std::string &kernel_filename,
                         std::string &kernel_name,
                         std::string &compile_option_string);

unsigned long ReleaseKernel(std::string &kernel_name);

inline void add_array_arg(std::string &arg_name,
                          py::array &arr,
                          std::vector<oclk::ArgWrapper> &constants);

py::list parse_args(py::list arg_dicts, std::vector<oclk::ArgWrapper> &args);

inline void add_array_arg(std::string &arg_name,
                          py::array &arr,
                          std::vector<oclk::ArgWrapper> &constants);

py::list parse_args(py::list arg_dicts, std::vector<oclk::ArgWrapper> &args);

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
RunnerReturn run_impl(py::kwargs &kwargs);

PYBIND11_MODULE(oclk_C, m) {
    m.doc() =
        "OCLK(OpenCL Kernel) runner Python api"; // optional module docstring
    m.def("init", &Init);
    m.def("load_kernel", &LoadKernel);
    m.def("release_kernel", &ReleaseKernel);
    m.def("run", &run_impl);
    m.def("clear_timer",&oclk::TimeMonitor::Clear);
    m.attr("__version__") = module_version;
    m.def("device_info",&oclk::GetDeviceName);

    py::class_<oclk::TimerResult>(m, "TimerResult")
        .def(py::init<const std::string &>())
        .def(py::init<const std::string &, long, double, double, double>())
        .def("__str__", &oclk::TimerResult::ToString)
        .def_readwrite("name", &oclk::TimerResult::name)
        .def_readwrite("cnt", &oclk::TimerResult::cnt)
        .def_readwrite("avg", &oclk::TimerResult::avg)
        .def_readwrite("stdev", &oclk::TimerResult::stdev)
        .def_readwrite("total", &oclk::TimerResult::total);

    py::class_<RunnerReturn>(m, "RunnerReturn")
        .def(py::init<>())
        .def_readwrite("timer_result", &RunnerReturn::timer_result)
        .def_readwrite("results", &RunnerReturn::results);
}

#endif // OPENCL_KERNELS_OCLK_PYAPI_H
