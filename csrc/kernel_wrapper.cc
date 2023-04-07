//
// Created by jimmy on 23-3-30.
//

#include "kernel_wrapper.h"
#include "ocl_demo.h"
#include <fstream>
#include <iostream>
#include <type_traits>
namespace oclk {
KernelWrapper::KernelWrapper(OclManager *manager_ptr) {
    m_ocl_manager_ = manager_ptr;
}
KernelWrapper::~KernelWrapper() = default;

cl_command_queue &KernelWrapper::GetCommandQueue() {
    return m_ocl_manager_->getCommandQueue();
}

std::string readFile(const std::string &filename) {
    std::ifstream file_stream(filename);
    CHECK_EQ(file_stream.is_open(), true)
        << "Error: Failed to open program file!";
    std::stringstream buffer;
    buffer << file_stream.rdbuf();
    return buffer.str();
}
cl_program KernelWrapper::CreateProgram_(const cl_context &ctx,
                                         const cl_device_id &device,
                                         const std::string &filename,
                                         const std::string &compile_options,
                                         const std::string &link_options) {
    auto program_str           = readFile(filename);
    const char *program_source = program_str.c_str();
    if (nullptr == program_source) {
        LOG(INFO) << "failed to read file " << filename;
        return nullptr;
    }
    cl_int err;
    cl_program program = clCreateProgramWithSource(
        ctx, 1, (const char **)&program_source, nullptr, &err);
    CHECK_CL_SUCCESS(err, "clCreateProgram failed");
    if (nullptr == program) {
        LOG(INFO) << "clCreateProgram failed, program is NULL";
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
    CHECK_CL_SUCCESS(err, "clCompileProgram failed");

    static const size_t LOG_SIZE = 2048;
    char log[LOG_SIZE];
    log[0] = 0;
    err    = clGetProgramBuildInfo(
        program, device, CL_PROGRAM_BUILD_LOG, LOG_SIZE, log, nullptr);
    if (log[0] != 0) {
        LOG(INFO) << "Program Build Log: " << log;
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

cl_kernel KernelWrapper::CreateKernel_(cl_program const &program,
                                       const std::string &kernel_name) {
    cl_int err;
    LOG(INFO) << kernel_name;
    cl_kernel kernel = clCreateKernel(program, kernel_name.c_str(), &err);

    CHECK_CL_SUCCESS(err, "failed to create kernel");
    return kernel;
}

int KernelWrapper::LoadKernel(const std::string &program_source_file,
                              const std::string &program_compile_options,
                              const std::string &program_link_options,
                              std::vector<std::string> &kernel_names) {
    cl_program program =
        this->CreateProgram_(this->m_ocl_manager_->getContext(),
                             this->m_ocl_manager_->getDeviceId(),
                             program_source_file,
                             program_compile_options,
                             program_link_options);
    if (nullptr == program) {
        LOG(INFO) << "program create failed";
        return 1;
    }
    for (std::string &_kernel_name : kernel_names) {
        if (this->kernels_.count(_kernel_name) == 1) {
            LOG(INFO) << "kernel name duplicated: " << _kernel_name;
            return 2;
        }
        std::string _real_kernel_name =
            _kernel_name.substr(0, _kernel_name.find("/"));
        cl_kernel kernel = CreateKernel_(program, _real_kernel_name);
        if (nullptr == kernel) {
            LOG(INFO) << "kernel create failed";
            return 3;
        }
        this->kernels_[_kernel_name] = kernel;
        LOG(INFO) << "Loaded kernel: " << _kernel_name;
    }
    return 0;
}
cl_kernel &KernelWrapper::GetKernelByName(const std::string &name) {
    CHECK_EQ(kernels_.count(name), 1) << "Kernel " << name << " not found";
    return this->kernels_.at(name);
}

cl_mem KernelWrapper::CreateBuffer_(const cl_context &ctx, size_t size) {
    cl_int err;
    cl_mem buf = clCreateBuffer(ctx, CL_MEM_READ_WRITE, size, nullptr, &err);
    CHECK_CL_SUCCESS(err, "clCreateBuffer failed");
    return buf;
}

/**
 * clCreateBuffer wrapper, allocate and fill ZERO
 * @param buf_name string for unordered map
 * @param size size of buffer in bytes
 * @return error code
 */
cl_uint KernelWrapper::CreateBuffer(const std::string &buf_name, size_t size) {
    if (m_buf_pool.count(buf_name)) {
        LOG(INFO) << "buf_name key duplicated";
        return 2;
    }
    cl_mem buf = CreateBuffer_(this->m_ocl_manager_->getContext(), size);
    if (nullptr == buf) {
        return 1;
    }
    size_t pattern = 0;
    clEnqueueFillBuffer(this->GetCommandQueue(),
                        buf,
                        &pattern,
                        1,
                        0,
                        size,
                        0,
                        nullptr,
                        nullptr);
    m_buf_pool[buf_name] = buf;
    return 0;
}
cl_mem KernelWrapper::GetBuffer(const std::string &buf_name) const {
    if (m_buf_pool.count(buf_name) == 0) {
        LOG(INFO) << "buf_name key error: not exist: " << buf_name;
        return nullptr;
    }
    return m_buf_pool.at(buf_name);
}
void KernelWrapper::ReleaseBuffer(const std::string &buf_name) {
    if (m_buf_pool.count(buf_name) == 0) {
        LOG(ERROR) << "buf_name key error: not exist: " << buf_name;
        return;
    }
    clReleaseMemObject(m_buf_pool.at(buf_name));
    m_buf_pool.erase(buf_name);
}
int KernelWrapper::LoadKernel(const std::string &program_source_file,
                              const std::string &program_compile_options,
                              const std::string &program_link_options,
                              const std::string &load_kernel_name) {
    LOG(INFO) << "test" << load_kernel_name;
    std::vector<std::string> kernel_names = {load_kernel_name};
    return LoadKernel(program_source_file,
                      program_compile_options,
                      program_link_options,
                      kernel_names);
}

} // namespace oclk