#include "binary_op.h"
#include <cmath>
#include <gtest/gtest.h>
#include <iostream>
using namespace std;
TEST(KERNEL, binary_op) {
    // construct
    OclManager mgr      = OclManager();
    BinaryOp add_kernel = BinaryOp(&mgr);

    // init var
    size_t length = 256;
    auto *a       = new float[length];
    auto *b       = new float[length];
    auto *c       = new float[length];

    srand(time(0));
    for (int i = 0; i < length; i++) {
        a[i] = (float)rand() / (float)RAND_MAX * 2.f - 1.f;
        b[i] = (float)rand() / (float)RAND_MAX * 2.f - 1.f;
        c[i] = 0.0f;
    }

    // calc and correct
    add_kernel.binary_op(a, b, c, length, BinaryOp::OPT::ADD);
    for (int i = 0; i < length; i++) {
        EXPECT_FLOAT_EQ(c[i], a[i] + b[i]);
    }

    add_kernel.binary_op(a, b, c, length, BinaryOp::OPT::SUB);
    for (int i = 0; i < length; i++) {
        ASSERT_FLOAT_EQ(c[i], a[i] - b[i]);
    }
    add_kernel.binary_op(a, b, c, length, BinaryOp::OPT::MUL);
    for (int i = 0; i < length; i++) {
        EXPECT_FLOAT_EQ(c[i], a[i] * b[i]);
    }
    add_kernel.binary_op(a, b, c, length, BinaryOp::OPT::DIV);
    for (int i = 0; i < length; i++) {
        EXPECT_FLOAT_EQ(c[i], a[i] / b[i]);
    }
    add_kernel.binary_op(a, b, c, length, BinaryOp::OPT::POW);
    for (int i = 0; i < length; i++) {
        float r  = pow(a[i], b[i]);
        if (isnan(r))continue;

        EXPECT_FLOAT_EQ(c[i], pow(a[i], b[i]));
    }
    add_kernel.binary_op(a, b, c, length, BinaryOp::OPT::MIN);
    for (int i = 0; i < length; i++) {
        EXPECT_FLOAT_EQ(c[i], a[i] < b[i] ? a[i] : b[i]);
    }
    add_kernel.binary_op(a, b, c, length, BinaryOp::OPT::MAX);
    for (int i = 0; i < length; i++) {
        EXPECT_FLOAT_EQ(c[i], a[i] > b[i] ? a[i] : b[i]);
    }

    // destroy
    delete[] a;
    delete[] b;
    delete[] c;
    GTEST_SUCCESS_("binary op test pass");
}