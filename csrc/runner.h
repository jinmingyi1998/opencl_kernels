//
// Created by jimmy on 23-7-28.
//
#ifndef CLKERNELBENCH_SRC_RUNNER_H
#define CLKERNELBENCH_SRC_RUNNER_H
#include "common.h"
#include "kernel.h"
#include "timer.h"
namespace oclk {
/**
 * Put anything runner returned in this class
 */
class CLRunnerReturnWrapper_ {
public:
    TimerResult timer_result = no_result;
};
typedef CLRunnerReturnWrapper_ *CLRunnerReturnWrapper;

class CLRunner {
public:
    CLRunner(OCLENV *env);
    void AddKernel(const std::string &kernel_name, cl_kernel kernel);
    void RunKernel(const std::string &kernel_name,
                   std::vector<ArgWrapper> &constants,
                   size_t dim,
                   long *global_work_size,
                   long *local_work_size,
                   bool wait                            = true,
                   TimerArgs timer_args                 = disabled_timer_arg,
                   CLRunnerReturnWrapper return_wrapper = nullptr);
    int ReleaseKernel(const std::string &kernel_name);

private:
    cl_context context;
    cl_device_id device_id;
    cl_command_queue command_queue;
    std::map<std::string, cl_kernel> kernel_lists;
};
} // namespace oclk
#endif // CLKERNELBENCH_SRC_RUNNER_H
