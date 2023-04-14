//
// Created by jimmy on 23-4-3.
//

#ifndef OPENCL_DEMO_OCL_DEMO_H
#define OPENCL_DEMO_OCL_DEMO_H
#include <glog/logging.h>

#define STRING_CAT2(S1, S2) S1##_##S2
#define STRING_CAT3(S1, S2, S3) S1##_##S2##_##S3

#define CHECK_CL_SUCCESS(e, msg)                                               \
    CHECK_EQ(CL_SUCCESS, e)                                                    \
        << "rtn code: " << (signed int)e << " message:" << msg

#define CHECK_RTN_PRINT_ERR_NO_RETURN(e, msg)                                  \
    CHECK_EQ(e, 0) << "rtn code: " << (signed int)e << " message:" << msg
namespace oclk { } // namespace oclk
#endif             // OPENCL_DEMO_OCL_DEMO_H
