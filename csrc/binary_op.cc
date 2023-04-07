//
// Created by jimmy on 23-3-29.
//

#include "binary_op.h"
#include "ocl_demo.h"
#include <glog/logging.h>
#include <iostream>

namespace oclk {

BinaryOp::BinaryOp(OclManager *managerPtr)
    : KernelWrapper(managerPtr) {
    if (!managerPtr) {
        return;
    }
    std::vector<std::string> kernel_name_methods = {
        "naive", "stride", "vec", "vec_stride"};
    /**
     * generate compile options, compile once, get multiple kernels
     *  by a vector of kernel_names.
     *  so 2 outer for-loop generate compile options,
     *  the inner for-loop generate kernel names in one compiled program
     */
    for (auto &suf : op_suffix) {
        for (auto &dtype_str : kernel_dtypes) {
            std::vector<std::string> kernel_names;
            for (auto &met : kernel_name_methods) {
                std::string t_kernel_name = kernel_name;
                t_kernel_name.append("_")
                    .append(met)
                    .append("_")
                    .append(suf)
                    .append("/")
                    .append(dtype_str);
                kernel_names.push_back(t_kernel_name);
            }
            LoadKernel(source_file_name,
                       " -DDTYPE=" + dtype_str + " -DBINARY_OP_OPT=" + suf,
                       "",
                       kernel_names);
        }
    }
    LOG(INFO) << "BinaryOp Loaded";
    LOG(INFO) << "kernel filename: " << source_file_name << std::endl;
}
} // namespace oclk