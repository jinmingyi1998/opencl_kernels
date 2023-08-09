//
// Created by jimmy on 23-7-28.
//
#include "runner.h"
#include "opencl_error_code.h"
namespace oclk {
CLRunner::CLRunner(OCLENV *env)
    : context(env->context)
    , device_id(env->device_id)
    , command_queue(env->command_queue) { }

void CLRunner::AddKernel(const std::string &kernel_name, cl_kernel kernel) {
    if (kernel_lists.find(kernel_name) != kernel_lists.end()) {
        spdlog::critical("kernel name must not be duplicated {}", kernel_name);
    }
    kernel_lists[kernel_name] = kernel;
}

void CLRunner::RunKernel(const std::string &kernel_name,
                         std::vector<ArgWrapper> &constants,
                         size_t dim,
                         long *global_work_size,
                         long *local_work_size,
                         bool wait,
                         TimerArgs timer_args) {
    auto kernel = this->kernel_lists.at(kernel_name);
    int arg_idx = 0;
    for (auto &c : constants) {
        int _err =
            clSetKernelArg(kernel, arg_idx++, c.bytes.size(), c.bytes.data());
        if (_err != CL_SUCCESS) {
            spdlog::error("set arg {} failed, err: {} {}",
                          arg_idx,
                          _err,
                          getErrorString(_err));
        }
    }
    if (local_work_size[0] == -1) {
        local_work_size = nullptr;
    }
    auto run_kernel_fn = [&]() {
        int _err = clEnqueueNDRangeKernel(command_queue,
                                          kernel,
                                          dim,
                                          nullptr,
                                          (size_t *)global_work_size,
                                          (size_t *)local_work_size,
                                          0,
                                          nullptr,
                                          nullptr);
        if (_err != CL_SUCCESS) {
            spdlog::error(
                "clEnqueueNDRangeKernel failed, kernel_name: {} err: {} msg:{}",
                kernel_name,
                _err,
                getErrorString(_err));
            return _err;
        }
        return _err;
    };
    if (timer_args.isEnable()) {
        int err;
        for (int i = 0; i < timer_args.getWarmup(); i++) {
            ASSERT_PRINT((err = run_kernel_fn(), err == 0),
                         getErrorString(err));
        }
        TIMER_KERNEL_BLOCK_REPEAT(
            timer_args.getTimerName(), timer_args.getRepeat(), command_queue, {
                ASSERT_PRINT((err = run_kernel_fn(), err == 0),
                             getErrorString(err));
            });
        TimeMonitor::ShowTimer(timer_args.getTimerName());
    } else {
        int err;
        ASSERT_PRINT((err = run_kernel_fn(), err == 0), getErrorString(err));
        if (wait) {
            clFlush(command_queue);
            clFinish(command_queue);
        }
    }
    return;
}

int CLRunner::ReleaseKernel(const std::string &kernel_name) {
    auto kernel = this->kernel_lists.at(kernel_name);
    int err     = oclk::ReleaseKernel(kernel);
    if (err == 0) {
        this->kernel_lists.erase(kernel_name);
    }
    return err;
}
} // namespace oclk