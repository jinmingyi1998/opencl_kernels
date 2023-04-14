//
// Created by jimmy on 23-4-11.
//

#include "buffer_wrapper.h"
namespace oclk {
/**
 * create buffer
 * @param ctx cl context
 * @param size size in bytes
 * @return buffer cl_mem object
 */
cl_mem BufferWrapper::CreateBuffer_(const cl_context &ctx, size_t size) {
    cl_int err;
    cl_mem buf = clCreateBuffer(ctx, CL_MEM_READ_WRITE, size, nullptr, &err);
    CHECK_CL_SUCCESS(err, "clCreateBuffer failed");
    return buf;
}
cl_mem &BufferWrapper::GetBuffer() { return m_buffer_; }
void BufferWrapper::ReleaseBuffer() { clReleaseMemObject(m_buffer_); }
BufferWrapper::~BufferWrapper() {
    ReleaseBuffer();
    m_size_     = 0;
    m_capacity_ = 0;
    m_buffer_   = nullptr;
}
size_t BufferWrapper::size() const { return m_size_; }
size_t BufferWrapper::capacity() const { return m_capacity_; }
std::string BufferWrapper::name() const { return m_buf_name_; }
std::string BufferWrapper::rename(const std::string &name) {
    m_buf_name_ = name;
    return m_buf_name_;
}
} // namespace oclk