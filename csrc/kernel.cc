//
// Created by jimmy on 23-4-24.
//

#include "kernel.h"
#include "opencl_error_code.h"
#include <fstream>
namespace oclk {
std::string readFile(const std::string &filename) {
    std::ifstream file_stream(filename);
    if (!file_stream.is_open()) {
        spdlog::critical(" Error: Failed to open program file! {}", filename);
        return "";
    }

    std::stringstream buffer;
    buffer << file_stream.rdbuf();
    return buffer.str();
}
cl_program CreateProgram_(const cl_context &ctx,
                          const cl_device_id &device,
                          const std::string &filename,
                          const std::string &compile_options,
                          const std::string &link_options) {
    spdlog::debug("filename: {}\tcompile_options: {}\tlink_options: {}",
                  filename,
                  compile_options,
                  link_options);
    auto program_str           = readFile(filename);
    const char *program_source = program_str.c_str();
    if (program_source == nullptr) {
        spdlog::critical("failed to read file {}", filename);
        ASSERT_PRINT(program_source != nullptr, "Failed to read file");
    }
    cl_int err;
    cl_program program = clCreateProgramWithSource(
        ctx, 1, (const char **)&program_source, nullptr, &err);

    CL_CHECK_RTN(err, "clCreateProgram failed");
    if (program == nullptr) {
        spdlog::critical("clCreateProgram failed, program is NULL");
        return nullptr;
    }
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
        spdlog::critical(
            "clCompileProgram failed! err: {} {}", err, getErrorString(err));
        static const size_t LOG_SIZE = 2048;
        char log[LOG_SIZE];
        log[0] = 0;
        err    = clGetProgramBuildInfo(
            program, device, CL_PROGRAM_BUILD_LOG, LOG_SIZE, log, nullptr);
        if (log[0] != 0) {
            spdlog::info("Program Build Log: {}", log);
        }
        return nullptr;
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
    if (err != CL_SUCCESS) {
        CL_CHECK_RTN(err, "Link Program failed");
        ASSERT_PRINT(false, "Link Program failed");
    }
    return linked_program;
}
cl_kernel LoadKernel(cl_context context,
                     cl_device_id deviceId,
                     const std::string &program_source_file,
                     const std::string &program_compile_options,
                     const std::string &program_link_options,
                     const std::string &kernel_name) {
    cl_program program = CreateProgram_(context,
                                        deviceId,
                                        program_source_file,
                                        program_compile_options,
                                        program_link_options);
    ASSERT_PRINT(program != nullptr, "program create failed");
    std::string _real_kernel_name =
        kernel_name.substr(0, kernel_name.find('/'));
    int err;
    cl_kernel kernel = clCreateKernel(program, _real_kernel_name.c_str(), &err);
    if (err != CL_SUCCESS || kernel == nullptr) {
        spdlog::error("error {} {}", err, getErrorString(err));
        ASSERT_PRINT((err != CL_SUCCESS && kernel != nullptr),
                     "failed to create kernel");
    }
    return kernel;
}
std::vector<cl_kernel>
LoadKernel(cl_context context,
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
    ASSERT_PRINT(program != nullptr, "program create failed");
    std::vector<cl_kernel> kernel_list;
    for (auto &kernel_name : kernel_name_list) {
        std::string _real_kernel_name =
            kernel_name.substr(0, kernel_name.find('/'));
        int err;
        cl_kernel kernel =
            clCreateKernel(program, _real_kernel_name.c_str(), &err);
        ASSERT_PRINT(err == 0 && kernel != nullptr, "kernel create failed");
        kernel_list.push_back(kernel);
    }
    return kernel_list;
}

int ReleaseKernel(cl_kernel kernel) {
    int err = clReleaseKernel(kernel);
    if (err != CL_SUCCESS) {
        spdlog::critical("clReleaseKernel failed, err: {}", err);
    }
    return err;
}
} // namespace oclk