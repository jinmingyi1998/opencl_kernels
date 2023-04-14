//
// Created by jimmy on 23-3-29.
//

#include "binary_op.h"
#include <glog/logging.h>
#include <iostream>

namespace oclk {
BinaryOp::BinaryOp(std::shared_ptr<OclManager> managerPtr)
    : KernelWrapper(managerPtr) {
    INIT_LOCK
    source_file_name = "../../kernel/binary_op.cl";
    kernel_name      = "binary_op";
    if (!managerPtr) {
        return;
    }
    /**
     * generate compile options, compile once, get multiple kernels
     *  by a vector of kernel_names.
     *  so 2 outer for-loop generate compile options,
     *  the inner for-loop generate kernel names in one compiled program
     */
    for (int op_int = ADD; op_int != NOPE; op_int++) {
        auto suf = opt2string(static_cast<OPT>(op_int));
        for (auto &dtype_str : kernel_dtypes) {
            std::vector<std::string> kernel_names;
            for (int method_int = NAIVE; method_int != NOMETHOD; method_int++) {
                auto met = method2string(static_cast<METHOD>(method_int));
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
                       " -DCL_DTYPE=" + dtype_str + " -DBINARY_OP_OPT=" + suf,
                       "",
                       kernel_names);
        }
    }
    LOG(INFO) << "BinaryOp Loaded";
    LOG(INFO) << "kernel filename: " << source_file_name << std::endl;
}
} // namespace oclk