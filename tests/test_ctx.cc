//
// Created by jimmy on 23-3-30.
//
#include <gtest/gtest.h>
#include <iostream>
#include <ocl_manager.h>
using namespace std;
using namespace oclk;
TEST(API, context_test) {
    auto mgr = OclManager::GetInstance();
    SUCCEED();
}
