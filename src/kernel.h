//
// Created by jimmy on 23-4-24.
//

#ifndef CLKERNELBENCH_KERNEL_H
#define CLKERNELBENCH_KERNEL_H
#include <CL/cl.h>
#include <string>
#include <vector>

#include "common.h"
namespace oclk {
cl_program CreateProgram_(const cl_context &ctx,
                          const cl_device_id &device,
                          const std::string &filename,
                          const std::string &compile_options,
                          const std::string &link_options);

cl_kernel LoadKernel(cl_context context,
                     cl_device_id deviceId,
                     const std::string &program_source_file,
                     const std::string &program_compile_options,
                     const std::string &program_link_options,
                     const std::string &kernel_name);

std::vector<cl_kernel>
LoadKernel(cl_context context,
           cl_device_id deviceId,
           const std::string &program_source_file,
           const std::string &program_compile_options,
           const std::string &program_link_options,
           const std::vector<std::string> &kernel_name_list);
} // namespace oclk
#endif // CLKERNELBENCH_KERNEL_H
