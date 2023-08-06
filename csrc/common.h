//
// Created by jimmy on 23-4-24.
//
#ifndef CLKERNELBENCH_COMMON_H
#define CLKERNELBENCH_COMMON_H
#include <CL/cl.h>
#include <iostream>
#include <map>
#include <sstream>
#include <string.h>
#include <string>
#include <vector>

#include "spdlog/spdlog.h"

#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE

#define ASSERT_PRINT(cond, msg)                                                \
    do {                                                                       \
        if (!(cond)) {                                                         \
            spdlog::critical("Assert failed: {}", msg);                        \
            abort();                                                           \
        }                                                                      \
    } while (0)

#define CHECK_RTN(e, msg)                                                      \
    do {                                                                       \
        if (e != 0)                                                            \
            spdlog::critical("rtn code: {} message: {}", (signed int)e, msg);  \
    } while (0)

namespace oclk {

static std::vector<cl_mem> allocated_gpumem;

int release_allocated_gpumem();

inline int binary_round_up(int value, int round_up_value) {
    return (value + round_up_value - 1) & (-round_up_value);
}

template <typename T> std::string stringify(T val) {
    std::stringstream ss;
    ss << val;
    std::string s = ss.str();
    return s;
}
template <typename T>
std::string parse_fields_to_name(std::vector<std::string> &key,
                                 std::vector<T> &values) {
    std::string name = "";
    for (int i = 0; i < key.size(); i++) {
        name.append(key.at(i));
        name.append("#");
        name.append(stringify(values.at(i)));
        if (i != key.size() - 1) name.append("$$");
    }
    return name;
}
/**
 * accept a kv, parse to a string representation
 * @tparam T
 * @param kv
 * @return
 */
template <typename T>
std::string parse_fields_to_name(std::map<std::string, T> &kv) {
    std::string name = "";
    if (kv.size() < 1) {
        return "";
    }
    for (auto it = kv.begin(); it != kv.end(); it++) {
        name.append(it->first);
        name.append("#");
        name.append(stringify(it->second));
        name.append("$$");
    }
    name = name.substr(0, name.length() - 1); // remove the last slash
    return name;
}
static inline void init_spdlog() {
    spdlog::set_pattern("[%H:%M:%S %z][%^%l%$][%P-%t] : %v");
}
struct OCLENV {
    cl_platform_id platform_id     = nullptr;
    cl_device_id device_id         = nullptr;
    cl_uint num_device             = 0;
    cl_context context             = nullptr;
    cl_command_queue command_queue = nullptr;
    cl_uint numPlatforms           = 0;
    int init() {
        init_spdlog();
        cl_int err = clGetPlatformIDs(1, &platform_id, &numPlatforms);
        CHECK_RTN(err, "clGetPlatformIDs failed");
        err = clGetDeviceIDs(
            platform_id, CL_DEVICE_TYPE_GPU, 0, nullptr, &num_device);
        CHECK_RTN(err, "Error getting device number");
        ASSERT_PRINT(num_device > 0, "No devices found");
        err = clGetDeviceIDs(
            platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &num_device);
        CHECK_RTN(err, "Error getting device");
        char deviceName[128];
        char deviceVersion[128];
        err = clGetDeviceInfo(
            device_id, CL_DEVICE_NAME, sizeof(deviceName), deviceName, NULL);
        err |= clGetDeviceInfo(device_id,
                               CL_DEVICE_VERSION,
                               sizeof(deviceVersion),
                               deviceVersion,
                               NULL);
        CHECK_RTN(err, "Error getting device info");
        spdlog::info("Loaded device {} version {}", deviceName, deviceVersion);
        context =
            clCreateContext(nullptr, 1, &device_id, nullptr, nullptr, &err);
        CHECK_RTN(err, "failed to create context");
        command_queue = clCreateCommandQueueWithProperties(
            context, device_id, nullptr, &err);
        CHECK_RTN(err, "clCreateCommandQueueWithProperties failed");
        return 0;
    }
};
static OCLENV ocl_instance;

cl_mem CreateBuffer(cl_context ctx, size_t size, size_t dtype_size);
cl_mem CreateImage2D(cl_context ctx,
                     size_t w,
                     size_t h,
                     cl_uint dtype = CL_UNORM_INT8);
struct ArgWrapper {
    std::string name = "";
    std::vector<char> bytes{};
    template <typename T> ArgWrapper(std::string name, T value) {
        this->name = name;
        bytes.resize(sizeof(T));
        memcpy((void *)bytes.data(), (void *)(&value), sizeof(T));
    }
};

/**
 * calculate and round up global and local work size
 * @param global_work_size
 * @param local_work_size
 * @return pair(global_work_size, local_work_size)
 */
std::pair<std::vector<long>, std::vector<long>>
GetWorkSize(std::vector<long> global_work_size,
            std::vector<long> local_work_size);
inline void read_data_from_buffer(cl_command_queue commandQueue,
                                  cl_mem buf,
                                  void *ptr,
                                  size_t size) {
    // read from buffer, write to ptr, size in bytes
    cl_int err = clEnqueueReadBuffer(
        commandQueue, buf, true, 0, size, ptr, 0, nullptr, nullptr);
    CHECK_RTN(err, "clEnqueueReadBuffer failed");

    clFlush(commandQueue);
    err = clFinish(commandQueue);
    CHECK_RTN(err, "clFinish failed");
}
template <typename T>
inline void read_buffer_to_vector(cl_command_queue commandQueue,
                                  cl_mem buf,
                                  std::vector<T> &vec) {
    read_data_from_buffer(
        commandQueue, buf, vec.data(), vec.size() * sizeof(T));
}
inline void write_data_to_buffer(cl_command_queue commandQueue,
                                 cl_mem buf,
                                 void *ptr,
                                 size_t size) {
    cl_int err = clEnqueueWriteBuffer(
        commandQueue, buf, true, 0, size, ptr, 0, nullptr, nullptr);
    CHECK_RTN(err, "write buffer failed");
}
template <typename T>
inline void write_vector_to_buffer(cl_command_queue commandQueue,
                                   cl_mem buf,
                                   const std::vector<T> &vec) {
    write_data_to_buffer(
        commandQueue, buf, (void *)vec.data(), vec.size() * sizeof(T));
}

template <typename T>
inline void write_vector_to_image(cl_command_queue commandQueue,
                                  cl_mem image,
                                  size_t w,
                                  size_t h,
                                  const std::vector<T> &vec) {
    size_t origin[3]{0, 0, 0};
    size_t region[3]{w, h, 1};
    cl_int err = clEnqueueWriteImage(commandQueue,
                                     image,
                                     true,
                                     origin,
                                     region,
                                     0,
                                     0,
                                     vec.data(),
                                     0,
                                     nullptr,
                                     nullptr);
    CHECK_RTN(err, "write image failed");
}
inline void
clear_buffer(cl_command_queue commandQueue, cl_mem buf, size_t buffer_size) {
    cl_uint fill_zero = 0;
    cl_int err        = clEnqueueFillBuffer(
        commandQueue, buf, &fill_zero, 1, 0, buffer_size, 0, nullptr, nullptr);
    CHECK_RTN(err, "fill buffer failed");
}

} // namespace oclk
#endif // CLKERNELBENCH_COMMON_H
