//
// Created by jimmy on 23-3-29.
//

#ifndef OPENCL_DEMO_BINARY_OP_H
#define OPENCL_DEMO_BINARY_OP_H
#include "kernel_wrapper.h"
#include <glog/logging.h>
#include <string>

class BinaryOp : public KernelWrapper {
public:
    enum OPT { ADD = 1, SUB = 2, MUL = 3, DIV = 4, POW = 5, MIN = 6, MAX = 7 };
    std::vector<std::string> op_suffix{
        "ADD",
        "SUB",
        "MUL",
        "DIV",
        "POW",
        "MIN",
        "MAX",
    };

    explicit BinaryOp(OclManager *managerPtr);

    void binary_op(
        const float *a, const float *b, float *result, size_t length, OPT opt);
    static std::string opt_string(OPT opt) {
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
};
#endif // OPENCL_DEMO_BINARY_OP_H
