//
// Created by jimmy on 23-4-4.
//

#include <glog/logging.h>
#include <gtest/gtest.h>
int main(int argc, char **argv) {
    google::InitGoogleLogging(argv[0]);
    FLAGS_logtostdout=1;
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}