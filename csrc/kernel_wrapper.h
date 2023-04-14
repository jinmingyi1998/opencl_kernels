//
// Created by jimmy on 23-3-30.
//

#ifndef OPENCL_DEMO_KERNEL_WRAPPER_H
#define OPENCL_DEMO_KERNEL_WRAPPER_H

#include "ocl_manager.h"
#include <glog/logging.h>
#include <memory>
#include <string>
#include <vector>
#define INIT_LOCK                                                              \
    static std::mutex init_lock;                                               \
    const std::lock_guard<std::mutex> lock(init_lock);

namespace oclk {
// base class for kernel wrapper
class KernelWrapper {
public:
    std::string kernel_name      = "noop";
    std::string source_file_name = "";
    bool has_initialized         = false;
    KernelWrapper() {
        LOG(WARNING) << "kernel: " << kernel_name
                     << " has not been initialized yet!";
    }
    explicit KernelWrapper(std::shared_ptr<OclManager> manager_ptr);
    ~KernelWrapper();
    std::vector<std::string> kernel_dtypes      = {"float", "half"};
    std::vector<std::string> kernel_vcetor_size = {"4", "8", "16"};

    /**
     * Get the string representation of dtype
     * @tparam T input value Type
     * @param v input value
     * @return string representation of dtype. NOTE: if not match, return float
     */
    template <typename T> std::string &dtype2str(T const &v) {
        if (std::is_same<T, float>::value)
            return kernel_dtypes[0];
        else if (std::is_same<T, cl_half>::value)
            return kernel_dtypes[1];
        return kernel_dtypes[0];
    }

protected:
    cl_command_queue &GetCommandQueue();

    cl_kernel &GetKernelByName(const std::string &name);
    int LoadKernel(const std::string &program_source_file,
                   const std::string &program_compile_options,
                   const std::string &program_link_options,
                   std::vector<std::string> &kernel_names);
    int LoadKernel(const std::string &program_source_file,
                   const std::string &program_compile_options,
                   const std::string &program_link_options,
                   const std::string &load_kernel_name);

    cl_uint CreateBuffer(const std::string &buf_name, size_t size);
    cl_mem GetBuffer(const std::string &buf_name) const;
    void ReleaseBuffer(const std::string &buf_name);

private:
    std::shared_ptr<OclManager> m_ocl_manager_;

    std::unordered_map<std::string, cl_program> programs; // {filename: program}

    std::unordered_map<std::string, cl_kernel>
        kernels_;                                       // {kernel_name: kernel}
    std::unordered_map<std::string, cl_mem> m_buf_pool; // {buffer_name: buffer}

private:
    static cl_program CreateProgram_(const cl_context &ctx,
                                     const cl_device_id &device,
                                     const std::string &filename,
                                     const std::string &compile_options,
                                     const std::string &link_options);
    static cl_kernel CreateKernel_(cl_program const &program,
                                   const std::string &kernel_name);
    static cl_mem CreateBuffer_(const cl_context &ctx, size_t size);
};
} // namespace oclk
#endif // OPENCL_DEMO_KERNEL_WRAPPER_H
