#include "operator/binary_op.h"
#include <clblast_half.h>
#include <cmath>
#include <glog/logging.h>
#include <gtest/gtest.h>
#define TEST_TYPE cl_half

using namespace std;
using namespace oclk;
static BinaryOp add_kernel;

class TestBinaryOp
    : public testing::TestWithParam<
          std::tuple<int, BinaryOp::OPT, BinaryOp::METHOD, int>> {
protected:
    void SetUp() {
        if (!add_kernel.has_initialized) {
            add_kernel = BinaryOp(OclManager::GetInstance());
        }
        std::tie(length, opt, kernel_method, local_work_size) = GetParam();
    }
    int length;
    BinaryOp::OPT opt;
    BinaryOp::METHOD kernel_method;
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
            arr[i] = FloatToHalf((float)rand() / RAND_MAX * 2.f - 1.f);
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
    testing::Combine(
        testing::Values(
            1, 5, 111, 333, 555, 1111, 2222, 4444, 8888, 16666, 32222, 128888),
        testing::Values(BinaryOp::ADD,
                        BinaryOp::SUB,
                        BinaryOp::MUL,
                        BinaryOp::DIV,
                        BinaryOp::MAX,
                        BinaryOp::MIN,
                        BinaryOp::POW),
        testing::Values(BinaryOp::NAIVE,
                        BinaryOp::STRIDE,
                        BinaryOp::VEC,
                        BinaryOp::VEC_STRIDE),
        testing::Values(1, 2, 4, 8, 16, 32, 64)));

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

#define CHECKFUNC(CALC_OPT, LOOPBODY)                                          \
    template <typename T> void TestBinaryOp::RunCompareCpuResult##CALC_OPT() { \
        auto a = generate_random_array<T>(length);                             \
        auto b = generate_random_array<T>(length);                             \
        auto c = vector<T>(length);                                            \
        add_kernel.binary_op(                                                  \
            a, b, c, BinaryOp::OPT::CALC_OPT, kernel_method, local_work_size); \
        vector<float> calc_a   = ConvertToFloat(a);                            \
        vector<float> calc_b   = ConvertToFloat(b);                            \
        vector<float> result_c = ConvertToFloat(c);                            \
        for (int i = 0; i < a.size(); i++) {                                   \
            LOOPBODY;                                                          \
        }                                                                      \
    }

CHECKFUNC(ADD, ASSERT_NEAR(result_c[i], calc_a[i] + calc_b[i], 0.1))
CHECKFUNC(SUB, ASSERT_NEAR(result_c[i], calc_a[i] - calc_b[i], 0.1);)
CHECKFUNC(MUL, ASSERT_NEAR(result_c[i], calc_a[i] * calc_b[i], 0.1);)
CHECKFUNC(DIV, {
    float r = calc_a[i] / calc_b[i];
    if (isnan(r)) continue;
    ASSERT_NEAR(result_c[i], calc_a[i] / calc_b[i], 0.2);
})
CHECKFUNC(POW, {
    float r = pow((float)calc_a[i], (float)calc_b[i]);
    if (isnan(r)) continue;
    ASSERT_NEAR(result_c[i], pow(calc_a[i], calc_b[i]), 0.2);
})
CHECKFUNC(MIN, {
    EXPECT_FLOAT_EQ(result_c[i], calc_a[i] < calc_b[i] ? calc_a[i] : calc_b[i]);
})
CHECKFUNC(MAX, {
    EXPECT_FLOAT_EQ(result_c[i], calc_a[i] > calc_b[i] ? calc_a[i] : calc_b[i]);
})
#define TESTNAME(type) STRING_CAT2(RunCompareGPUResult, type)
TEST_P(TestBinaryOp, TESTNAME(TEST_TYPE)) {
    LOG(INFO) << "length = " << length << " opt = " << BinaryOp::opt2string(opt)
              << " method = " << kernel_method;
    switch (opt) {
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