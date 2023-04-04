#include <algorithm>
#include <iostream>

#include "ocl_demo.h"
#include "ocl_manager.h"
cl_platform_id OclManager::CreatePlatform_() {
    cl_platform_id platformId;
    cl_uint numPlatforms;
    cl_int err = clGetPlatformIDs(1, &platformId, &numPlatforms);
    CHECK_CL_SUCCESS(err, "clGetPlatformIDs failed")
    if (numPlatforms <= 0) {
        LOG(INFO) << "No platforms found";
        return nullptr;
    }
    return platformId;
}
cl_uint OclManager::CreatePlatform() {
    cl_platform_id platformId = CreatePlatform_();
    if (nullptr == platformId) {
        return 1;
    }
    m_platform_id_ = platformId;
    return 0;
}

cl_device_id OclManager::CreateDevice_(const cl_platform_id &platform_id,
                                       cl_uint *num_device) {
    cl_int err =
        clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 0, nullptr, num_device);
    CHECK_CL_SUCCESS(err, "Error getting device number")
    if (*num_device <= 0) {
        LOG(INFO) << "No devices found";
        return nullptr;
    }
    cl_device_id deviceId;
    err = clGetDeviceIDs(
        platform_id, CL_DEVICE_TYPE_GPU, 1, &deviceId, num_device);
    CHECK_CL_SUCCESS(err, "Error getting device")
    return deviceId;
}
cl_uint OclManager::CreateDevice() {
    cl_uint num_device     = 0;
    cl_device_id device_id = CreateDevice_(m_platform_id_, &num_device);
    if (nullptr == device_id) {
        return 1;
    }
    m_num_devices_ = num_device;
    m_device_id_   = device_id;
    return 0;
}

cl_context OclManager::CreateContext_(const cl_device_id &deviceId) {
    cl_int err;
    cl_context ctx =
        clCreateContext(nullptr, 1, &deviceId, nullptr, nullptr, &err);
    CHECK_CL_SUCCESS(err, "failed to create context")
    return ctx;
}
cl_uint OclManager::CreateContext() {
    cl_context ctx = CreateContext_(m_device_id_);
    if (nullptr == ctx) {
        return 1;
    }
    m_context_ = ctx;
    return 0;
}
cl_command_queue OclManager::CreateCommandQueue_(const cl_context &context,
                                                 const cl_device_id &device) {
    cl_int err;
    cl_command_queue commandQueue =
        clCreateCommandQueueWithProperties(context, device, nullptr, &err);
    CHECK_CL_SUCCESS(err, "clCreateCommandQueueWithProperties failed")
    if (!commandQueue) {
        LOG(INFO) << "clCreateCommandQueueWithProperties failed, queue is NULL";
        return nullptr;
    }
    return commandQueue;
}
cl_uint OclManager::CreateCommandQueue() {
    cl_command_queue commandQueue =
        CreateCommandQueue_(m_context_, m_device_id_);
    if (nullptr == commandQueue) {
        return 1;
    }
    m_command_queue_ = commandQueue;
    return 0;
}
void OclManager::Cleanup() {
    if (m_command_queue_) clReleaseCommandQueue(m_command_queue_);
    m_command_queue_ = nullptr;
    if (m_context_) clReleaseContext(m_context_);
    m_context_ = nullptr;
}

OclManager::OclManager() {
    cl_uint err;
    err = CreatePlatform();
    CHECK_RTN_PRINT_ERR_NO_RETURN(err, "failed to create platform");
    err = CreateDevice();
    CHECK_RTN_PRINT_ERR_NO_RETURN(err, "failed to create device");
    err = CreateContext();
    CHECK_RTN_PRINT_ERR_NO_RETURN(err, "failed to create context");
    err = CreateCommandQueue();
    CHECK_RTN_PRINT_ERR_NO_RETURN(err, "failed to create command queue");
    LOG(INFO) << "CL context created";
}
OclManager::~OclManager() {
    Cleanup();
    LOG(INFO) << "CL context destroyed";
}
cl_command_queue &OclManager::getCommandQueue() { return m_command_queue_; }
const cl_device_id &OclManager::getDeviceId() const { return m_device_id_; }
const cl_context &OclManager::getContext() const { return m_context_; }
