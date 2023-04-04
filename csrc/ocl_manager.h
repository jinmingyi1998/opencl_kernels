#ifndef OPENCL_DEMO_OCL_MANAGER_H
#define OPENCL_DEMO_OCL_MANAGER_H
#include <CL/cl.h>
#include <unordered_map>
#include <vector>

class OclManager {
public:
    OclManager();
    ~OclManager();
    cl_uint CreatePlatform();
    cl_uint CreateDevice();
    cl_uint CreateContext();
    cl_uint CreateCommandQueue();

    cl_command_queue &getCommandQueue();

    const cl_device_id &getDeviceId() const;
    const cl_context &getContext() const;

private:
    cl_platform_id CreatePlatform_();
    cl_device_id CreateDevice_(const cl_platform_id &platform_id,
                               cl_uint *num_device);
    cl_context CreateContext_(const cl_device_id &deviceId);
    cl_command_queue CreateCommandQueue_(const cl_context &context,
                                         const cl_device_id &device);
    void Cleanup();
    cl_platform_id m_platform_id_;
    cl_device_id m_device_id_;
    cl_uint m_num_devices_;
    cl_context m_context_;
    cl_command_queue m_command_queue_;
};
#endif // OPENCL_DEMO_OCL_MANAGER_H
