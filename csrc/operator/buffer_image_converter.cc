//
// Created by jimmy on 23-4-11.
//

#include "buffer_image_converter.h"
#include <glog/logging.h>
namespace oclk {

BufferImageConverter::BufferImageConverter(
    std::shared_ptr<OclManager> manager_ptr)
    : KernelWrapper(manager_ptr) {
    INIT_LOCK
    kernel_name                  = "buffer_image_converter";
    source_file_name             = "../../kernel/buffer_image.cl";
    std::string load_kernel_name = "buffer_image_converter_b2i_float32_TO_int8";
    LoadKernel(source_file_name, "", "", load_kernel_name);
    LOG(INFO) << "BufferImageConverter b2i float32_to_int8 loaded";
}
BufferImageConverter::~BufferImageConverter() { }
} // namespace oclk
