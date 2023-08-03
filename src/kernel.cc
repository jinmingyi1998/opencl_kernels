//
// Created by jimmy on 23-4-24.
//

#include "kernel.h"
#include <fstream>
std::string readFile(const std::string &filename) {
    std::ifstream file_stream(filename);
    LOG_ASSERT(file_stream.is_open())
        << " Error: Failed to open program file! " << filename;
    std::stringstream buffer;
    buffer << file_stream.rdbuf();
    return buffer.str();
}
cl_program oclk::CreateProgram_(const cl_context &ctx,
                                const cl_device_id &device,
                                const std::string &filename,
                                const std::string &compile_options,
                                const std::string &link_options) {
    VLOG(3) << "filename: " << filename
            << "\tcompile_options: " << compile_options
            << "\tlink_options: " << link_options;
    auto program_str           = readFile(filename);
    const char *program_source = program_str.c_str();
    CHECK_NE(program_source, nullptr) << "failed to read file " << filename;
    cl_int err;
    cl_program program = clCreateProgramWithSource(
        ctx, 1, (const char **)&program_source, nullptr, &err);
    CHECK_CL_SUCCESS(err, "clCreateProgram failed");
    CHECK_NE(program, nullptr) << "clCreateProgram failed, program is NULL";
    err = clCompileProgram(program,
                           1,
                           &device,
                           compile_options.c_str(),
                           0,
                           nullptr,
                           nullptr,
                           nullptr,
                           nullptr);
    if (err != CL_SUCCESS) {
        LOG(ERROR) << "clCompileProgram failed";
        static const size_t LOG_SIZE = 2048;
        char log[LOG_SIZE];
        log[0] = 0;
        err    = clGetProgramBuildInfo(
            program, device, CL_PROGRAM_BUILD_LOG, LOG_SIZE, log, nullptr);
        if (log[0] != 0) {
            VLOG(1) << "Program Build Log: " << log;
        }
        CHECK_CL_SUCCESS(err, "compile error");
    }

    cl_program linked_program = clLinkProgram(ctx,
                                              1,
                                              &device,
                                              link_options.c_str(),
                                              1,
                                              &program,
                                              nullptr,
                                              nullptr,
                                              &err);
    CHECK_CL_SUCCESS(err, "Link Program failed");
    return linked_program;
}
cl_kernel oclk::LoadKernel(cl_context context,
                           cl_device_id deviceId,
                           const std::string &program_source_file,
                           const std::string &program_compile_options,
                           const std::string &program_link_options,
                           const std::string &kernel_name) {
    VLOG(2) << "Loading kernel: " << kernel_name
            << " surce file:" << program_source_file;
    cl_program program = CreateProgram_(context,
                                        deviceId,
                                        program_source_file,
                                        program_compile_options,
                                        program_link_options);
    CHECK_NE(program, nullptr) << "program create failed";
    std::string _real_kernel_name =
        kernel_name.substr(0, kernel_name.find('/'));
    int err;
    cl_kernel kernel = clCreateKernel(program, _real_kernel_name.c_str(), &err);
    CHECK_CL_SUCCESS(err, "failed to create kernel");
    CHECK_NE(kernel, nullptr) << "kernel create failed";
    VLOG(2) << "Loaded kernel: " << kernel_name;
    return kernel;
}
std::vector<cl_kernel>
oclk::LoadKernel(cl_context context,
                 cl_device_id deviceId,
                 const std::string &program_source_file,
                 const std::string &program_compile_options,
                 const std::string &program_link_options,
                 const std::vector<std::string> &kernel_name_list) {
    cl_program program = CreateProgram_(context,
                                        deviceId,
                                        program_source_file,
                                        program_compile_options,
                                        program_link_options);
    CHECK_NE(program, nullptr) << "program create failed";
    std::vector<cl_kernel> kernel_list;
    for (auto &kernel_name : kernel_name_list) {
        VLOG(2) << "Loading kernel: " << kernel_name
                << " surce file:" << program_source_file;
        std::string _real_kernel_name =
            kernel_name.substr(0, kernel_name.find('/'));
        int err;
        cl_kernel kernel =
            clCreateKernel(program, _real_kernel_name.c_str(), &err);
        CHECK_CL_SUCCESS(err, "failed to create kernel");
        CHECK_NE(kernel, nullptr) << "kernel create failed";
        VLOG(2) << "Loaded kernel: " << kernel_name;
        kernel_list.push_back(kernel);
    }
    return kernel_list;
}