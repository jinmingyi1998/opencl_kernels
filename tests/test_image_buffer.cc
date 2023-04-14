//
// Created by jimmy on 23-4-11.
//
#include "ocl_manager.h"
#include "operator/buffer_image_converter.h"
#include "utils.h"
#include <CL/cl.h>
#include <gtest/gtest.h>
#include <vector>
using namespace std;
using namespace oclk;
static BufferImageConverter buffer_image_kernel;
class TestImageWrapper : public testing::TestWithParam<size_t> {
protected:
    void SetUp() {
        if (!buffer_image_kernel.has_initialized) {
            buffer_image_kernel =
                BufferImageConverter(OclManager::GetInstance());
        }
        length = GetParam();
    }
    size_t length;
};
TEST_P(TestImageWrapper, TestImageAndBufferWrapper) {
    vector<float> arr(length);
    srand(time(0));
    for (int i = 0; i < length; i++) {
        arr[i] = (float)rand() / RAND_MAX * 20.f;
    }
    auto mgr = OclManager::GetInstance();
    auto buf_a =
        BufferWrapper(mgr->getContext(), mgr->getCommandQueue(), arr, "arr");
    auto img = ImageWrapper(mgr->getContext(), 100, length / 100, 1);
    buffer_image_kernel.buffer_2_image(buf_a, img);
    SUCCEED();
}
INSTANTIATE_TEST_SUITE_P(TEST_OCLK_API,
                         TestImageWrapper,
                         testing::Values(1e2, 1e3, 1e4, 1e5, 1e6));
