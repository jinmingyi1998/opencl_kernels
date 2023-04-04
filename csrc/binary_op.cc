//
// Created by jimmy on 23-3-29.
//

#include "binary_op.h"
#include "ocl_demo.h"
#include <glog/logging.h>
#include <iostream>
BinaryOp::BinaryOp(OclManager *managerPtr)
    : KernelWrapper(managerPtr) {
    std::string source_file_name = "../../kernel/binary_op.cl";

    std::vector<std::string> kernel_name_methods = {"naive", "stride","vec","vec_stride"};
    for (auto &suf : op_suffix) {
        std::vector<std::string> kernel_names;
        for (auto &met : kernel_name_methods) {
            std::string kernel_name = "binary_op_";
            kernel_name.append(met).append("_").append(suf);
            kernel_names.push_back(kernel_name);
        }
        LoadKernel(
            source_file_name,
            "-DBINARY_OP_OPT=" + suf, "", kernel_names);
    }
    LOG(INFO) << "BinaryOp Loaded";
    LOG(INFO) << "kernel filename: " << source_file_name << std::endl;
}

void BinaryOp::binary_op(
    const float *a, const float *b, float *result, size_t length, OPT opt) {
    int err;
    std::string kernel_name_suf = opt_string(opt);

    cl_kernel &kernel    = GetKernelByName("binary_op_naive_"+kernel_name_suf);
    LOG(INFO) << "running kernel name: "<< "binary_op_naive_"+kernel_name_suf;
    uint bytesize_length = length * sizeof(float);
    err                  = CreateBuffer("a", bytesize_length);
    CHECK_RTN_PRINT_ERR_NO_RETURN(err, "create_buffer a failed");
    err = CreateBuffer("b", bytesize_length);
    CHECK_RTN_PRINT_ERR_NO_RETURN(err, "create_buffer b failed");
    err = CreateBuffer("c", bytesize_length);
    CHECK_RTN_PRINT_ERR_NO_RETURN(err, "create_buffer c failed");
    err = clEnqueueWriteBuffer(GetCommandQueue(),
                               GetBuffer("a"),
                               CL_TRUE,
                               0,
                               bytesize_length,
                               a,
                               0,
                               nullptr,
                               nullptr);
    CHECK_RTN_PRINT_ERR_NO_RETURN(err, "write buffer a failed")
    err = clEnqueueWriteBuffer(GetCommandQueue(),
                               GetBuffer("b"),
                               CL_TRUE,
                               0,
                               bytesize_length,
                               b,
                               0,
                               nullptr,
                               nullptr);
    CHECK_RTN_PRINT_ERR_NO_RETURN(err, "write buffer b failed")
    auto buf_a = GetBuffer("a");
    auto buf_b = GetBuffer("b");
    auto buf_c = GetBuffer("c");
    err        = clSetKernelArg(kernel, 0, sizeof(cl_mem), &buf_a);
    CHECK_RTN_PRINT_ERR_NO_RETURN(err, "set kernel arg a failed")
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &buf_b);
    CHECK_RTN_PRINT_ERR_NO_RETURN(err, "set kernel arg b failed")
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &buf_c);
    CHECK_RTN_PRINT_ERR_NO_RETURN(err, "set kernel arg c failed")
    int l = length;
    err   = clSetKernelArg(kernel, 3, sizeof(int), &l);
    CHECK_RTN_PRINT_ERR_NO_RETURN(err, "set kernel arg length failed")
    size_t local_work_size = 4;

    err = clEnqueueNDRangeKernel(GetCommandQueue(),
                                 kernel,
                                 1,
                                 NULL,
                                 &length,
                                 &local_work_size,
                                 0,
                                 nullptr,
                                 nullptr);
    CHECK_RTN_PRINT_ERR_NO_RETURN(err, "clEnqueueNDRangeKernel failed")

    err = clEnqueueReadBuffer(GetCommandQueue(),
                              GetBuffer("c"),
                              CL_TRUE,
                              0,
                              bytesize_length,
                              result,
                              0,
                              nullptr,
                              nullptr);
    CHECK_RTN_PRINT_ERR_NO_RETURN(err, "read buffer c failed")
    ReleaseBuffer("a");
    ReleaseBuffer("b");
    ReleaseBuffer("c");
}
