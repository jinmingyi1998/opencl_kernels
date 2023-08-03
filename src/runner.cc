//
// Created by jimmy on 23-7-28.
//
#include "runner.h"
namespace oclk {
CLRunner::CLRunner(OCLENV *env)
    : context(env->context)
    , device_id(env->device_id)
    , command_queue(env->command_queue) { }

void CLRunner::AddKernel(const std::string &kernel_name, cl_kernel kernel) {
    if (kernel_lists.find(kernel_name) != kernel_lists.end()) {
        LOG(ERROR) << "kernel name must not be duplicated";
    }
    kernel_lists[kernel_name] = kernel;
}
} // namespace oclk