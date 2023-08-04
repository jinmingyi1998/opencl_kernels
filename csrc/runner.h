//
// Created by jimmy on 23-7-28.
//
#ifndef CLKERNELBENCH_SRC_RUNNER_H
#define CLKERNELBENCH_SRC_RUNNER_H
#include "common.h"
#include "kernel.h"
#include "timer.h"
namespace oclk {
class CLRunner {
public:
    CLRunner(OCLENV *env);
    void AddKernel(const std::string &kernel_name, cl_kernel kernel);
    void RunKernel(const std::string &kernel_name,
                   std::vector<ArgWrapper> &constants,
                   size_t dim,
                   size_t *global_work_size,
                   size_t *local_work_size,
                   bool wait            = true,
                   TimerArgs timer_args = disabled_timer_arg) {
        auto kernel = this->kernel_lists.at(kernel_name);
        int arg_idx = 0;
        for (auto &c : constants) {
            CHECK_RTN_PRINT_ERR_NO_RETURN(
                clSetKernelArg(
                    kernel, arg_idx++, c.bytes.size(), c.bytes.data()),
                "set constant arg failed ")
                << arg_idx;
        }
        auto run_kernel_fn = [&]() {
            int _err = clEnqueueNDRangeKernel(command_queue,
                                              kernel,
                                              dim,
                                              nullptr,
                                              global_work_size,
                                              local_work_size,
                                              0,
                                              nullptr,
                                              nullptr);
            CHECK_RTN_PRINT_ERR_NO_RETURN(_err, "clEnqueueNDRangeKernel failed")
                << "\nkernel_name" << kernel_name;
        };
        if (timer_args.isEnable()) {
            for (int i = 0; i < timer_args.getWarmup(); i++) {
                run_kernel_fn();
            }
            TIMER_KERNEL_BLOCK_REPEAT(timer_args.getTimerName(),
                                      timer_args.getRepeat(),
                                      command_queue,
                                      { run_kernel_fn(); });
            TimeMonitor::ShowAll();
            TimeMonitor::Clear();
        } else {
            run_kernel_fn();
            if (wait) {
                clFlush(command_queue);
                clFinish(command_queue);
            }
        }
    }

private:
    cl_context context;
    cl_device_id device_id;
    cl_command_queue command_queue;
    std::map<std::string, cl_kernel> kernel_lists;
};
} // namespace oclk
#endif // CLKERNELBENCH_SRC_RUNNER_H
