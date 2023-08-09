//
// Created by jimmy on 23-7-28.
//
#include "oclk_pyapi.h"

namespace py = pybind11;

std::shared_ptr<oclk::CLRunner> runner;

unsigned long Init() {
    int err = oclk::ocl_instance.init();
    if (err != 0) {
        spdlog::critical("init failed , error code: {}", err);
    } else {
        spdlog::info("init success");
    }
    runner = std::make_shared<oclk::CLRunner>(&oclk::ocl_instance);
    return err;
}

unsigned long LoadKernel(std::string &kernel_filename,
                         std::string &kernel_name,
                         std::string &compile_option_string) {
    oclk::OCLENV *env = &oclk::ocl_instance;
    auto k            = oclk::LoadKernel(env->context,
                              env->device_id,
                              kernel_filename,
                              compile_option_string,
                              "",
                              kernel_name);
    runner->AddKernel(kernel_name, k);
    return 0;
}

inline void add_array_arg(std::string &arg_name,
                          py::array &arr,
                          std::vector<oclk::ArgWrapper> &args) {
    oclk::OCLENV *env = &oclk::ocl_instance;
    size_t array_size = arr.size();
    auto mem = oclk::CreateBuffer(env->context, array_size, arr.itemsize());
    oclk::write_data_to_buffer(
        env->command_queue, mem, (void *)arr.data(), arr.nbytes());
    args.emplace_back(arg_name, mem);
}

/**
 * parse arg from kwargs dict, meanwhile modify args vector
 * @param arg_dict list of dict
 * @param args
 * @return py::list of all pyobject, order same as args
 */
py::list parse_args(py::list arg_dicts, std::vector<oclk::ArgWrapper> &args) {
    for (int i = 0; i < arg_dicts.size(); i++) {
        auto arg_dict    = arg_dicts[i].cast<py::dict>();
        std::string name = arg_dict["name"].cast<std::string>();
        auto v           = arg_dict["value"];
        spdlog::debug("Parsing arg: [{:>10}] type: {}",
                      name,
                      v.get_type().cast<py::str>().cast<std::string>());
        if (py::isinstance<py::array>(v)) {
            py::array arr = v.cast<py::array>();
            spdlog::debug("    ndarray [dtype: {:>10}] [size: {:8d}] [data "
                          "size: {}]",
                          arr.dtype().cast<py::str>().cast<std::string>(),
                          arr.size(),
                          oclk::human_readable_bytesize(arr.nbytes()));
            add_array_arg(name, arr, args);
        } else if (py::isinstance<py::int_>(v) ||
                   py::isinstance<py::float_>(v)) {
            if (arg_dict.contains("type")) {
                std::string type_str = arg_dict["type"].cast<std::string>();
                spdlog::debug("        type string: {}", type_str);
                if (type_str == "float") {
                    float vv = v.cast<float>();
                    args.emplace_back(name, vv);
                } else if (type_str == "double") {
                    double vv = v.cast<double>();
                    args.emplace_back(name, vv);
                } else if (type_str == "int" || type_str == "unsigned int") {
                    unsigned int vv = v.cast<unsigned int>();
                    args.emplace_back(name, vv);
                } else if (type_str == "long" || type_str == "unsigned long") {
                    unsigned long vv = v.cast<long>();
                    args.emplace_back(name, vv);
                } else {
                    spdlog::error("Unknown type {}", type_str);
                }
            } else {
                if (py::isinstance<py::int_>(v)) {
                    long v_int = v.cast<long>();
                    args.emplace_back(name, v_int);
                    spdlog::debug("    constant arg: {}", v_int);
                } else {
                    float v_float = v.cast<float>();
                    args.emplace_back(name, v_float);
                    spdlog::debug("    constant arg: {}", v_float);
                }
            }
        } else {
            spdlog::error("error: unknown type, only support int, "
                          "float, np.array, but got {}",
                          v.get_type().cast<py::str>().cast<std::string>());
            return arg_dicts;
        }
    }
    return arg_dicts;
}

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
RunnerReturn run_impl(py::kwargs &kwargs) {
    oclk::OCLENV *env = &oclk::ocl_instance;
    std::vector<oclk::ArgWrapper> kernel_args;
    parse_args(kwargs["input"].cast<py::list>(), kernel_args);
    /** arg_list is list of dict:
     * [
     *      {"name":"something0", "value":12312, "type": "float"},
     *      {"name":"something1", "value":45.45},
     *      {"name":"something2", "value":np.array([1,2,3],dtype=np.float32)}
     * ]
     * TODO: add "type"
     */
    auto global_work_size_pylist =
        kwargs[py::str("global_work_size")].cast<py::list>();
    auto local_work_size_pylist =
        kwargs[py::str("local_work_size")].cast<py::list>();
    std::vector<long> global_work_size, local_work_size;
    for (int i = 0; i < global_work_size_pylist.size(); i++) {
        global_work_size.push_back(global_work_size_pylist[i].cast<long>());
        local_work_size.push_back(local_work_size_pylist[i].cast<long>());
    }
    auto ws = oclk::GetWorkSize(global_work_size, local_work_size);
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
        spdlog::debug(ss.str());
    }();
    auto kernel_name    = kwargs[py::str("kernel_name")].cast<std::string>();
    auto timer_arg_dict = kwargs[py::str("timer")].cast<py::dict>();
    oclk::TimerArgs timer_args(
        timer_arg_dict[py::str("enable")].cast<bool>(),
        timer_arg_dict[py::str("warmup")].cast<unsigned long>(),
        timer_arg_dict[py::str("repeat")].cast<unsigned long>(),
        timer_arg_dict[py::str("name")].cast<std::string>());
    auto runner_return = std::make_unique<oclk::CLRunnerReturnWrapper_>();
    runner->RunKernel(kernel_name,
                      kernel_args,
                      ws.first.size(),
                      ws.first.data(),
                      ws.second.data(),
                      kwargs[py::str("wait")].cast<bool>(),
                      timer_args,
                      runner_return.get());

    auto out_arg_list = kwargs[py::str("output")].cast<py::list>();
    auto in_arg_list  = kwargs["input"].cast<py::list>();
    py::list result_arrays;
    for (auto &s : out_arg_list) {
        auto arg_name = s.cast<std::string>();
        for (int i = 0; i < kernel_args.size(); i++) {
            auto &c = kernel_args[i];
            if (c.name == arg_name) {
                cl_mem mem;
                memcpy(&mem, c.bytes.data(), c.bytes.size());
                py::array arr = (in_arg_list[i].cast<py::dict>())["value"]
                                    .cast<py::array>();
                spdlog::debug("read arg: [{:>10}] [size: {}] [data size: {}]",
                              arg_name,
                              arr.size(),
                              oclk::human_readable_bytesize(arr.nbytes()));
                oclk::read_data_from_buffer(
                    env->command_queue, mem, arr.mutable_data(), arr.nbytes());
                result_arrays.append(arr);
                break;
            }
        }
    }
    RunnerReturn result;
    result.timer_result = runner_return->timer_result;
    result.results      = result_arrays;
    oclk::release_allocated_gpumem();
    return result;
}

unsigned long ReleaseKernel(std::string &kernel_name) {
    return runner->ReleaseKernel(kernel_name);
}
