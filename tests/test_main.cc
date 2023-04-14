//
// Created by jimmy on 23-4-4.
//

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
int main(int argc, char **argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, false);
    google::InitGoogleLogging(argv[0]);
    FLAGS_logtostdout = 1;
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}