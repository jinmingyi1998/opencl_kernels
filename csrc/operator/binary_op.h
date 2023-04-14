//
// Created by jimmy on 23-3-29.
//

#ifndef OPENCL_DEMO_BINARY_OP_H
#define OPENCL_DEMO_BINARY_OP_H
#include "../kernel_wrapper.h"
#include "../ocl_demo.h"
#include "../utils/calc.h"
#include <bitset>
#include <glog/logging.h>
#include <string>
#include <unordered_map>
namespace oclk {

class BinaryOp : public KernelWrapper {
public:
    enum OPT {
        ADD  = 1,
        SUB  = 2,
        MUL  = 3,
        DIV  = 4,
        POW  = 5,
        MIN  = 6,
        MAX  = 7,
        NOPE = 8,
    };
    enum METHOD {
        NAIVE      = 1,
        STRIDE     = 2,
        VEC        = 3,
        VEC_STRIDE = 4,
        NOMETHOD   = 5,
    };
    BinaryOp() { }
    BinaryOp(std::shared_ptr<OclManager> managerPtr);

    template <typename T>
    void binary_op(const std::vector<T> &a,
                   const std::vector<T> &b,
                   std::vector<T> &result,
                   OPT opt,
                   METHOD kernel_method,
                   size_t local_work_size) {
        if (local_work_size <= 0) local_work_size = 4;
        LOG_ASSERT(a.size() == result.size()) << "vector size should be same";
        std::bitset<32> local_work_size_bitset((unsigned long)local_work_size);
        LOG_ASSERT(local_work_size_bitset.count() == 1)
            << "local work size must be multiple of 2";

        std::string kernel_name_suf = opt2string(opt);
        std::string run_kernel_name = kernel_name + "_" +
                                      method2string(kernel_method) + "_" +
                                      kernel_name_suf;
        std::string dtype_str = dtype2str(a[0]);
        run_kernel_name.append("/").append(dtype_str);
        cl_kernel &kernel = GetKernelByName(run_kernel_name);

        // round up align
        size_t length                = a.size();
        uint real_bytesize_length    = length * sizeof(T);
        size_t rounded_length        = binary_round_up(length, 16);
        uint rounded_bytesize_length = rounded_length * sizeof(T);
        size_t global_work_size      = length;
        if (kernel_method == METHOD::VEC || kernel_method == METHOD::STRIDE) {
            global_work_size = binary_round_up(int(global_work_size / 4), 4);
        }
        else if (kernel_method == METHOD::VEC_STRIDE) {
            global_work_size = binary_round_up((int)global_work_size / 16, 16);
        }
        global_work_size = global_work_size < 1 ? 1 : global_work_size;
        global_work_size = binary_round_up(global_work_size, local_work_size);

        // create array buffers
        cl_uint err;
        err = CreateBuffer("a", rounded_bytesize_length);
        CHECK_RTN_PRINT_ERR_NO_RETURN(err, "create_buffer a failed");
        err = CreateBuffer("b", rounded_bytesize_length);
        CHECK_RTN_PRINT_ERR_NO_RETURN(err, "create_buffer b failed");
        err = CreateBuffer("c", rounded_bytesize_length);
        CHECK_RTN_PRINT_ERR_NO_RETURN(err, "create_buffer c failed");
        err = clEnqueueWriteBuffer(GetCommandQueue(),
                                   GetBuffer("a"),
                                   CL_TRUE,
                                   0,
                                   real_bytesize_length,
                                   a.data(),
                                   0,
                                   nullptr,
                                   nullptr);
        CHECK_RTN_PRINT_ERR_NO_RETURN(err, "write buffer a failed");
        err = clEnqueueWriteBuffer(GetCommandQueue(),
                                   GetBuffer("b"),
                                   CL_TRUE,
                                   0,
                                   real_bytesize_length,
                                   b.data(),
                                   0,
                                   nullptr,
                                   nullptr);
        CHECK_RTN_PRINT_ERR_NO_RETURN(err, "write buffer b failed");
        auto buf_a = GetBuffer("a");
        auto buf_b = GetBuffer("b");
        auto buf_c = GetBuffer("c");
        err        = clSetKernelArg(kernel, 0, sizeof(cl_mem), &buf_a);
        CHECK_RTN_PRINT_ERR_NO_RETURN(err, "set kernel arg a failed");
        err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &buf_b);
        CHECK_RTN_PRINT_ERR_NO_RETURN(err, "set kernel arg b failed");
        err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &buf_c);
        CHECK_RTN_PRINT_ERR_NO_RETURN(err, "set kernel arg c failed");
        err = clSetKernelArg(kernel, 3, sizeof(int), &length);
        CHECK_RTN_PRINT_ERR_NO_RETURN(err, "set kernel arg length failed");

        err = clEnqueueNDRangeKernel(GetCommandQueue(),
                                     kernel,
                                     1,
                                     NULL,
                                     &global_work_size,
                                     &local_work_size,
                                     0,
                                     nullptr,
                                     nullptr);
        CHECK_RTN_PRINT_ERR_NO_RETURN(err, "clEnqueueNDRangeKernel failed");

        err = clEnqueueReadBuffer(GetCommandQueue(),
                                  GetBuffer("c"),
                                  CL_TRUE,
                                  0,
                                  real_bytesize_length,
                                  result.data(),
                                  0,
                                  nullptr,
                                  nullptr);
        CHECK_RTN_PRINT_ERR_NO_RETURN(err, "read buffer c failed");
        ReleaseBuffer("a");
        ReleaseBuffer("b");
        ReleaseBuffer("c");
    }
    static std::string opt2string(OPT opt) {
        switch (opt) {
            case ADD:
                return "ADD";
            case SUB:
                return "SUB";
            case MUL:
                return "MUL";
            case DIV:
                return "DIV";
            case POW:
                return "POW";
            case MIN:
                return "MIN";
            case MAX:
                return "MAX";
            default:
                return "";
        }
    }
    static OPT string2opt(const std::string &str) {
        if (str == "ADD") {
            return ADD;
        }
        else if (str == "SUB") {
            return SUB;
        }
        else if (str == "MUL") {
            return MUL;
        }
        else if (str == "DIV") {
            return DIV;
        }
        else if (str == "POW") {
            return POW;
        }
        else if (str == "MIN") {
            return MIN;
        }
        else if (str == "MAX") {
            return MAX;
        }
        else {
            return NOPE;
        }
    }
    static std::string method2string(METHOD method) {
        switch (method) {
            case NAIVE:
                return "naive";
            case STRIDE:
                return "stride";
            case VEC:
                return "vec";
            case VEC_STRIDE:
                return "vec_stride";
            default:
                return "unknown";
        }
    }
};
} // namespace oclk
#endif // OPENCL_DEMO_BINARY_OP_H
