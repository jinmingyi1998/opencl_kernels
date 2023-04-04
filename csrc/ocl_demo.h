//
// Created by jimmy on 23-4-3.
//

#ifndef OPENCL_DEMO_OCL_DEMO_H
#define OPENCL_DEMO_OCL_DEMO_H
#include <glog/logging.h>

#define CHECK_CL_SUCCESS(e, msg)                                               \
    CHECK_EQ(CL_SUCCESS, e) << "rtn code: " << e << " message:" << msg;

#define CHECK_RTN_PRINT_ERR_NO_RETURN(e, msg)                                  \
    CHECK_EQ(e, 0) << "rtn code: " << e << " message:" << msg;

#endif // OPENCL_DEMO_OCL_DEMO_H
