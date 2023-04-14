//
// Created by jimmy on 23-4-11.
//

#ifndef OPENCL_DEMO_BUFFER_WRAPPER_H
#define OPENCL_DEMO_BUFFER_WRAPPER_H
#include "../ocl_demo.h"
#include "CL/cl.h"
#include "calc.h"
#include "image_wrapper.h"
namespace oclk {
class BufferWrapper {
public:
    BufferWrapper(const cl_context &ctx,
                  size_t size,
                  size_t dtype_size,
                  const std::string &name)
        : ctx_(ctx)
        , m_size_(size) {
        m_buf_name_      = name;
        m_capacity_      = binary_round_up(size, 32);
        size_t byte_size = m_capacity_ * dtype_size;
        m_buffer_        = CreateBuffer_(ctx, byte_size);
    }
    template <typename T>
    BufferWrapper(const cl_context &ctx,
                  cl_command_queue &command_queue,
                  std::vector<T> arr,
                  const std::string &name)
        : ctx_(ctx) {
        m_buf_name_      = name;
        m_size_          = arr.size();
        m_capacity_      = binary_round_up(arr.size(), 32);
        size_t byte_size = m_capacity_ * sizeof(T);
        m_buffer_        = CreateBuffer_(ctx, byte_size);
        int err          = clEnqueueWriteBuffer(command_queue,
                                       m_buffer_,
                                       CL_TRUE,
                                       0,
                                       m_size_ * sizeof(T),
                                       arr.data(),
                                       0,
                                       nullptr,
                                       nullptr);
        CHECK_RTN_PRINT_ERR_NO_RETURN(err, "write buffer failed");
    }
    ~BufferWrapper();
    cl_mem &GetBuffer();
    size_t size() const;
    size_t capacity() const;
    std::string name() const;
    std::string rename(const std::string &name);

private:
    cl_mem m_buffer_;
    const cl_context &ctx_;
    std::string m_buf_name_;
    size_t m_size_;     // number of data T, NOT in bytes
    size_t m_capacity_; // rounded up size, NOT in bytes
    static cl_mem CreateBuffer_(const cl_context &ctx, size_t size);
    void ReleaseBuffer();
};
} // namespace oclk
#endif // OPENCL_DEMO_BUFFER_WRAPPER_H
