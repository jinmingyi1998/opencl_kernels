//
// Created by jimmy on 23-5-12.
//
#include "common.h"

namespace oclk {

inline size_t roundup_value(const size_t x, const size_t v) {

    return (x + v - 1) / v * v;
}
cl_mem CreateBuffer(cl_context ctx, size_t size, size_t dtype_size) {
    int capacity  = binary_round_up(size, 64);
    int byte_size = capacity * dtype_size;
    cl_int err;
    cl_mem buf =
        clCreateBuffer(ctx, CL_MEM_READ_WRITE, byte_size, nullptr, &err);
    ASSERT_PRINT(err == 0, "clCreateBuffer failed");
    return buf;
}
cl_mem CreateImage2D(cl_context ctx, size_t w, size_t h, cl_uint dtype) {
    cl_mem img;
    cl_int err;
    cl_image_format format{CL_RGBA, dtype};
    cl_image_desc desc{
        CL_MEM_OBJECT_IMAGE2D,
        w,
        h,
        1,
        1,
    };
    img = clCreateImage(ctx, CL_MEM_READ_WRITE, &format, &desc, nullptr, &err);

    ASSERT_PRINT(err==0, "clCreateBuffer failed");
    return img;
}
std::pair<std::vector<size_t>, std::vector<size_t>>
GetWorkSize(std::vector<size_t> global_work_size,
            std::vector<size_t> local_work_size) {
    if (global_work_size.size() != local_work_size.size() ||
        global_work_size.size() > 3 || global_work_size.size() < 1) {
        std::stringstream ss;
        ss << "work size error\nglobal work size: " << global_work_size.size()
           << " local work size: " << local_work_size.size();
        throw std::runtime_error(ss.str());
    }
    for (int i = 0; i < global_work_size.size(); i++) {
        if (global_work_size.at(i) == 0) break;
        if (local_work_size.at(i) == 0) local_work_size.at(i) = 1;
        global_work_size.at(i) =
            roundup_value(global_work_size.at(i), local_work_size.at(i));
        global_work_size.at(i) = roundup_value(global_work_size.at(i), 16);
        global_work_size.at(i) =
            std::max(global_work_size.at(i), local_work_size.at(i));
    }
    return std::make_pair(global_work_size, local_work_size);
}
} // namespace oclk