//
// Created by jimmy on 23-7-28.
//
#include <iostream>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string.h>

#include "runner.h"
namespace py = pybind11;

std::shared_ptr<oclk::CLRunner> runner;
unsigned long Init() {
    oclk::ocl_instance.init();
    runner = std::make_shared<oclk::CLRunner>(&oclk::ocl_instance);
    return 0;
}
unsigned long LoadKernel(std::string &kernel_filename,
                         std::string &kernel_name,
                         std::string &compile_option_string) {
    oclk::OCLENV *env = &oclk::ocl_instance;
    LOG(INFO) << "Compiling kernel file" << kernel_filename << " kernel name "
              << kernel_name << " compile option " << compile_option_string;
    auto k = oclk::LoadKernel(env->context,
                              env->device_id,
                              kernel_filename,
                              compile_option_string,
                              "",
                              kernel_name);
    runner->AddKernel(kernel_name, k);
    return 0;
}

inline void add_array_arg(std::string arg_name,
                          py::array &arr,
                          std::vector<oclk::ArgWrapper> &constants) {
    oclk::OCLENV *env = &oclk::ocl_instance;
    size_t array_size = arr.size();
    auto mem = oclk::CreateBuffer(env->context, array_size, arr.itemsize());
    oclk::write_data_to_buffer(
        env->command_queue, mem, (void *)arr.data(), arr.nbytes());
    constants.emplace_back(arg_name, mem);
}
/**
 * parse arg from kwargs dict, meanwhile modify args vector
 * @param arg_dict
 * @param args
 * @return py::list of all pyobject, order same as args
 */
py::list parse_args(py::dict arg_dict, std::vector<oclk::ArgWrapper> &args) {
    auto iter     = arg_dict.begin();
    auto end_iter = arg_dict.end();
    py::list arg_list;
    while (iter != end_iter) {
        arg_list.append(iter->second);
        LOG(INFO) << "Parse arg: " << iter->first
                  << " type:" << iter->second.get_type();
        if (py::isinstance<py::array>(iter->second)) {
            py::array arr = iter->second.cast<py::array>();
            LOG(INFO) << "    array dtype: " << arr.dtype()
                      << " size: " << arr.size()
                      << " data size: " << arr.nbytes() << " Bytes";
            add_array_arg((iter->first).cast<std::string>(), arr, args);
        } else if (py::isinstance<py::int_>(iter->second) ||
                   py::isinstance<py::float_>(iter->second)) {
            if (py::isinstance<py::int_>(iter->second)) {
                long v = iter->second.cast<long>();
                args.emplace_back((iter->first).cast<std::string>(), v);
            } else {
                float v = iter->second.cast<float>();
                args.emplace_back(iter->first.cast<std::string>(), v);
            }
        } else {
            std::stringstream err_msg("error: unknown type, only support int, "
                                      "float, np.array, but got");
            err_msg << iter->first << " type: " << iter->second.get_type();
            LOG(ERROR) << err_msg.str();
            return arg_list;
        }
        iter++;
    }
}
/**
 * @param kwargs example input:
 *     {
 *         "kernel_name" str : "YOUR_KERNEL_NAME"
 *         "global_worksize" List[int]: [1123,132],
 *         "local_worksize" List[int]: [1,1],
 *         "input" Dict[str,Union[int,float,np.array]] : {
 *             "name1" : value1 ,
 *             "name2" : value1 ,
 *             "name3" : value1 ,
 *             "name4" : value2
 *         },
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
py::list run_impl(py::kwargs &kwargs) {
    oclk::OCLENV *env = &oclk::ocl_instance;
    std::vector<oclk::ArgWrapper> kernel_args;
    auto in_arg_list =
        parse_args(kwargs[py::str("input")].cast<py::dict>(), kernel_args);
    auto global_work_size_pylist =
        kwargs[py::str("global_work_size")].cast<py::list>();
    auto local_work_size_pylist =
        kwargs[py::str("local_work_size")].cast<py::list>();
    std::vector<size_t> global_work_size, local_work_size;
    for (int i = 0; i < global_work_size_pylist.size(); i++) {
        global_work_size.push_back(global_work_size_pylist[i].cast<size_t>());
        local_work_size.push_back(local_work_size_pylist[i].cast<size_t>());
    }
    auto ws = oclk::GetWorkSize(global_work_size, local_work_size);
    for (auto &c : kernel_args) {
        LOG(INFO) << "Arg name: " << c.name << " size: " << c.bytes.size();
    }
    [&]() {
        std::stringstream ss;
        ss << "local_work_size= { ";
        for (auto &v : ws.second) {
            ss << v << " ";
        }
        ss << "} global_work_size= { ";
        for (auto &v : ws.first) {
            ss << v << " ";
        }
        ss << "}";
        LOG(INFO) << ss.str();
    }();
    auto kernel_name    = kwargs[py::str("kernel_name")].cast<std::string>();
    auto timer_arg_dict = kwargs[py::str("timer")].cast<py::dict>();
    oclk::TimerArgs timer_args(
        timer_arg_dict[py::str("enable")].cast<bool>(),
        timer_arg_dict[py::str("warmup")].cast<unsigned long>(),
        timer_arg_dict[py::str("repeat")].cast<unsigned long>(),
        timer_arg_dict[py::str("name")].cast<std::string>());
    runner->RunKernel(kernel_name,
                      kernel_args,
                      ws.first.size(),
                      ws.first.data(),
                      ws.second.data(),
                      kwargs[py::str("wait")].cast<bool>(),
                      timer_args);

    auto out_arg_list = kwargs[py::str("output")].cast<py::list>();

    for (auto &s : out_arg_list) {
        auto arg_name = s.cast<std::string>();
        for (int i = 0; i < kernel_args.size(); i++) {
            auto &c = kernel_args[i];
            if (c.name == arg_name) {
                cl_mem mem;
                memcpy(&mem, c.bytes.data(), c.bytes.size());
                py::array arr = in_arg_list[i];
                oclk::read_data_from_buffer(
                    env->command_queue, mem, arr.mutable_data(), arr.nbytes());
                break;
            }
        }
    }
    return out_arg_list;
}

PYBIND11_MODULE(oclk_C, m) {
    m.doc() =
        "OCLK(OpenCL Kernel) runner Python api"; // optional module docstring
    m.def("init", &Init, "");
    m.def("load_kernel", &LoadKernel, "");
    m.def("run", &run_impl, "run_impl_float");
}