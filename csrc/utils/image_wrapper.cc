//
// Created by jimmy on 23-4-10.
//

#include "image_wrapper.h"
#include <CL/cl.h>
#include <vector>
namespace oclk {
cl_mem &ImageWrapper::GetImage() { return this->m_image_; }
uint ImageWrapper::Width() const { return this->w; }
uint ImageWrapper::Height() const { return h; }
uint ImageWrapper::Depth() const { return d; }
ImageWrapper::~ImageWrapper() {
    clReleaseMemObject(m_image_);
    h = 0;
    w = 0;
    d = 0;
}
ImageWrapper::ImageWrapper(const cl_context &ctx,
                           size_t height,
                           size_t width,
                           size_t depth)
    : ctx_(ctx)
    , h(height)
    , w(width)
    , d(depth) {
    int err     = 0;
    auto format = cl_image_format{CL_R, CL_UNSIGNED_INT8};
    auto desc   = cl_image_desc{CL_MEM_OBJECT_IMAGE2D, w, h, d, 1};

    this->m_image_ =
        clCreateImage(ctx_, CL_MEM_WRITE_ONLY, &format, &desc, nullptr, &err);
    CHECK_RTN_PRINT_ERR_NO_RETURN(err, "create image failed");
}
} // namespace oclk