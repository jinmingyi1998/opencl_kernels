//
// Created by jimmy on 23-4-11.
//

#ifndef OPENCL_DEMO_BUFFER_IMAGE_CONVERTER_H
#define OPENCL_DEMO_BUFFER_IMAGE_CONVERTER_H
#include "../kernel_wrapper.h"
#include "../utils.h"
#include <string>
namespace oclk {
class BufferImageConverter : public KernelWrapper {
public:
    BufferImageConverter(){};
    BufferImageConverter(std::shared_ptr<OclManager> manager_ptr);
    ~BufferImageConverter();
    enum Direction { b2i, i2b };
    static inline std::string direction_to_str(Direction direction) {
        if (direction == Direction::b2i) return "b2i";
        return "i2b";
    }
    void buffer_2_image(BufferWrapper &buffer, ImageWrapper &image) {
        std::string load_kernel_name = this->kernel_name;
        load_kernel_name.append("_")
            .append(direction_to_str(Direction::b2i))
            .append("_")
            .append("float32_TO_int8");
        auto kernel        = GetKernelByName(load_kernel_name);
        long buffer_offset = 0;
        int err            = 0;
        err |= clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer.GetBuffer());
        err |= clSetKernelArg(kernel, 1, sizeof(long), &buffer_offset);
        err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &image.GetImage());
        CHECK_RTN_PRINT_ERR_NO_RETURN(err, "set kernel arguments failed");
        size_t global_work_size[2] = {
            //            (size_t)binary_round_up(image.Width(), 1),
            //            (size_t)binary_round_up(image.Height(), 1)
            image.Width(),
            image.Height()};
        size_t local_work_size[2] = {1, 1};
        err                       = clEnqueueNDRangeKernel(GetCommandQueue(),
                                     kernel,
                                     2,
                                     nullptr,
                                     global_work_size,
                                     local_work_size,
                                     0,
                                     nullptr,
                                     nullptr);
        CHECK_RTN_PRINT_ERR_NO_RETURN(err, "clEnqueueNDRangeKernel failed");
        LOG(INFO) << "Success w = " << image.Width()
                  << " h = " << image.Height();
    }
};
} // namespace oclk
#endif // OPENCL_DEMO_BUFFER_IMAGE_CONVERTER_H
