//
// Created by jimmy on 23-4-12.
//
#include "helper/helper.h"
#include "helper/timer.h"
#include "ocl_demo.h"
#include "operator/binary_op.h"
#include "utils.h"
#include <chrono>
#include <gflags/gflags.h>
#include <vector>
DEFINE_int32(warmup, 3, "warmup times");
DEFINE_int32(repeat, 50, "repeat times");
DEFINE_int32(start_length, 100, "start test length");
DEFINE_int32(max_length,
             10000,
             "max test length, test length will grow from start length to "
             "max_test_length");
DEFINE_bool(dump_csv, true, "dump to csv, default true");
DEFINE_string(output_csv, "outptu.csv", "output csv filename");

using namespace std;
using namespace oclk;

int main(int argc, char *argv[]) {
    google::InitGoogleLogging(argv[0]);
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    FLAGS_logtostdout = 1;
    BinaryOp kernel   = BinaryOp(OclManager::GetInstance());
    size_t length     = FLAGS_start_length;
    size_t base       = length;
    while (length <= FLAGS_max_length) {
        vector<float> a(length);
        vector<float> b(length);
        vector<float> c(length);
        for (int i = 0; i < length; i++) {
            a.at(i) = static_cast<float>((float)rand() / RAND_MAX * 2.f - 1.f);
            b.at(i) = static_cast<float>((float)rand() / RAND_MAX * 2.f - 1.f);
            c.at(i) = static_cast<float>((float)rand() / RAND_MAX * 2.f - 1.f);
        }
        for (int worksize = 1; worksize <= 16; worksize <<= 1) {
            for (int opt_int = BinaryOp::OPT::ADD;
                 opt_int != BinaryOp::OPT::NOPE;
                 opt_int++) {
                auto opt = static_cast<BinaryOp::OPT>(opt_int);
                for (int method_int = BinaryOp::METHOD::NAIVE;
                     method_int != BinaryOp::NOMETHOD;
                     method_int++) {
                    auto method     = static_cast<BinaryOp::METHOD>(method_int);
                    int warmup_time = FLAGS_warmup;
                    int repeat      = FLAGS_repeat;
                    while (warmup_time--) {
                        kernel.binary_op(a, b, c, opt, method, worksize);
                    }
                    clFinish(OclManager::GetInstance()->getCommandQueue());

                    vector<string> keys{"name",
                                        "opt",
                                        "method",
                                        "worksize",
                                        "length",
                                        "repeat"};
                    vector<string> values{"binary_op",
                                          BinaryOp::opt2string(opt),
                                          BinaryOp::method2string(method),
                                          stringify(worksize),
                                          stringify(length),
                                          stringify(repeat)};
                    auto test_name = parse_fields_to_name(keys, values);
                    TIMER_BLOCK(test_name, {
                        while (repeat--) {
                            kernel.binary_op(a, b, c, opt, method, worksize);
                        }
                        clFinish(OclManager::GetInstance()->getCommandQueue());
                    })
                }
            }
        }
        if (length / base == 10) base *= 10;
        length += base;
    }
    if (FLAGS_dump_csv) {
        TimeMonitor::DumpCSV(FLAGS_output_csv);
    }
    else {
        TimeMonitor::ShowAll();
    }
}
