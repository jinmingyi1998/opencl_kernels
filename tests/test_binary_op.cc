#include "binary_op.h"
#include <clblast_half.h>
#include <cmath>
#include <gtest/gtest.h>
#include <iostream>
#define TEST_TYPE float

using namespace std;
using namespace oclk;
static auto add_kernel = BinaryOp(OclManager::GetInstance().get());

class TestBinaryOp : public testing::TestWithParam<
                         std::tuple<int, BinaryOp::OPT, std::string, int>> {
protected:
    void SetUp() {
        std::tie(length, opt, kernel_method, local_work_size) = GetParam();
    }
    int length;
    BinaryOp::OPT opt;
    std::string kernel_method;
    int local_work_size;
    template <typename T> void RunCompareCpuResultADD();
    template <typename T> void RunCompareCpuResultSUB();
    template <typename T> void RunCompareCpuResultMUL();
    template <typename T> void RunCompareCpuResultDIV();
    template <typename T> void RunCompareCpuResultPOW();
    template <typename T> void RunCompareCpuResultMIN();
    template <typename T> void RunCompareCpuResultMAX();
};

/**
 * generate an array with randome float number, in domain [-0.1,0.1]
 * // TODO: move this function out, as a common utility function
 * @tparam T
 * @param length vector length, as arr.size()
 * @return a vector instance
 */
template <typename T> inline vector<T> generate_random_array(size_t length) {
    vector<T> arr;
    arr.resize(length);
    srand(time(0));
    for (int i = 0; i < length; i++) {
        if (is_same<T, cl_half>::value) {
            FloatToHalf((float)rand() / RAND_MAX * 2.f - 1.f);
        }
        else {
            arr[i] = static_cast<T>((float)rand() / RAND_MAX * 2.f - 1.f);
        }
    }
    return arr;
}

INSTANTIATE_TEST_SUITE_P(
    TestBinaryOpSuite,
    TestBinaryOp,
    testing::Combine(testing::Values(1, 100, 512, 1000, 1024, 2000, 4000),
                     testing::Values(BinaryOp::ADD,
                                     BinaryOp::SUB,
                                     BinaryOp::MUL,
                                     BinaryOp::DIV,
                                     BinaryOp::MAX,
                                     BinaryOp::MIN,
                                     BinaryOp::POW),
                     testing::Values("naive", "stride", "vec", "vec_stride"),
                     testing::Values(1, 4, 8)));
#define CHECKFUNC(CALC_OPT, LOOPBODY)                                          \
    template <typename T> void TestBinaryOp::RunCompareCpuResult##CALC_OPT() { \
        auto a = generate_random_array<T>(length);                             \
        auto b = generate_random_array<T>(length);                             \
        auto c = vector<T>(length);                                            \
        add_kernel.binary_op(a,                                                \
                             b,                                                \
                             c,                                                \
                             length,                                           \
                             BinaryOp::OPT::CALC_OPT,                          \
                             kernel_method,                                    \
                             local_work_size);                                 \
        vector<float> calc_a   = ConvertToFloat(a);                            \
        vector<float> calc_b   = ConvertToFloat(b);                            \
        vector<float> result_c = ConvertToFloat(c);                            \
        for (int i = 0; i < length; i++) {                                     \
            LOOPBODY;                                                          \
        }                                                                      \
    }

template <typename T> vector<float> ConvertToFloat(vector<T> &arr) {
    vector<float> new_arr;
    new_arr.resize(arr.size());
    for (int i = 0; i < arr.size(); i++) {
        if (is_same<T, cl_half>::value) {
            new_arr[i] = HalfToFloat(arr[i]);
        }
        else if (is_same<T, float>::value) {
            new_arr[i] = arr[i];
        }
    }
    return new_arr;
}

CHECKFUNC(ADD, EXPECT_NEAR(result_c[i], calc_a[i] + calc_b[i], 0.1))
CHECKFUNC(SUB, EXPECT_NEAR(result_c[i], calc_a[i] - calc_b[i], 0.1);)
CHECKFUNC(MUL, EXPECT_NEAR(result_c[i], calc_a[i] * calc_b[i], 0.1);)
CHECKFUNC(DIV, {
    float r = calc_a[i] / calc_b[i];
    if (isnan(r)) continue;
    EXPECT_NEAR(result_c[i], calc_a[i] / calc_b[i], 0.1);
})
CHECKFUNC(POW, {
    float r = pow((float)calc_a[i], (float)calc_b[i]);
    if (isnan(r)) continue;
    EXPECT_NEAR(result_c[i], pow(calc_a[i], calc_b[i]), 0.1);
})
CHECKFUNC(MIN, {
    EXPECT_FLOAT_EQ(result_c[i], calc_a[i] < calc_b[i] ? calc_a[i] : calc_b[i]);
})
CHECKFUNC(MAX, {
    EXPECT_FLOAT_EQ(result_c[i], calc_a[i] > calc_b[i] ? calc_a[i] : calc_b[i]);
})

TEST_P(TestBinaryOp, RunCompareCpuResult) {
    LOG(INFO) << "length = " << length << " opt = " << BinaryOp::opt2string(opt)
              << " method = " << kernel_method;
    switch (opt) {
            // TODO: add type cl_half
        case BinaryOp::OPT::ADD:
            RunCompareCpuResultADD<TEST_TYPE>();
            break;
        case BinaryOp::OPT::SUB:
            RunCompareCpuResultSUB<TEST_TYPE>();
            break;
        case BinaryOp::OPT::MUL:
            RunCompareCpuResultMUL<TEST_TYPE>();
            break;
        case BinaryOp::OPT::DIV:
            RunCompareCpuResultDIV<TEST_TYPE>();
            break;
        case BinaryOp::OPT::POW:
            RunCompareCpuResultPOW<TEST_TYPE>();
            break;
        case BinaryOp::OPT::MIN:
            RunCompareCpuResultMIN<TEST_TYPE>();
            break;
        case BinaryOp::OPT::MAX:
            RunCompareCpuResultMAX<TEST_TYPE>();
            break;
        default:
            LOG(FATAL) << "not support opt=" << BinaryOp::opt2string(opt);
            FAIL();
    }
}