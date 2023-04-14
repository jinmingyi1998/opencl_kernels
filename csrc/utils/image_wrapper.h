//
// Created by jimmy on 23-4-10.
//

#ifndef OPENCL_DEMO_IMAGE_WRAPPER_H
#define OPENCL_DEMO_IMAGE_WRAPPER_H
#include "../ocl_demo.h"
#include "CL/cl.h"
#include "buffer_wrapper.h"
#include <glog/logging.h>
#include <type_traits>
#include <vector>
namespace oclk {
/**
 * create a cl image. 3 dims are Height, Width, Depth
 */
class ImageWrapper {
public:
    ImageWrapper(const cl_context &ctx,
                 size_t height,
                 size_t width,
                 size_t depth);
    ~ImageWrapper();
    cl_mem &GetImage();
    uint Height() const;
    uint Width() const;
    uint Depth() const;

public:
    static const size_t DIM_1   = 1;
    static const size_t DIM_2   = 2;
    static const size_t DIM_3   = 3;
    static const size_t NO_AXIS = -1;

private:
    const cl_context &ctx_;
    uint h = 0;
    uint w = 0;
    uint d = 0;
    cl_mem m_image_;
};

} // namespace oclk
#endif // OPENCL_DEMO_IMAGE_WRAPPER_H
