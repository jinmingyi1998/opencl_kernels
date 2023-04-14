//
// Created by jimmy on 23-4-13.
//

#ifndef OPENCL_DEMO_TIMER_H
#define OPENCL_DEMO_TIMER_H

#include <algorithm>
#include <glog/logging.h>
#include <ios>
#include <map>
#include <mutex>
#include <thread>
#include <unordered_map>
namespace oclk {

class TimeMonitor {
public:
    static int Init(const std::string &name);
    static int AddData(const std::string &name, const double &data);
    static int ShowAll();
    static int DumpCSV(const std::string &csvOutputFilename);

    /**
     * 初始化的时候记录初始时间，变量离开作用域以后会做统计
     * 多线程中可以使用。
     */
    class ScopedCumulator final {
    public:
        ScopedCumulator(const std::string &name);
        ~ScopedCumulator();

    private:
        std::string mName;
        double mStartTime;
    };
};
inline const long int time_s() {
    return std::chrono::duration_cast<std::chrono::seconds>(
               std::chrono::high_resolution_clock::now().time_since_epoch())
        .count();
}
inline const long int time_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::high_resolution_clock::now().time_since_epoch())
        .count();
}
inline const long int time_us() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::high_resolution_clock::now().time_since_epoch())
        .count();
}
inline const long int time_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::high_resolution_clock::now().time_since_epoch())
        .count();
}
inline const double time_ms_d() {
    double t = (double)time_us();
    return t / 1000.0;
}
#define TIMER_BLOCK(name, _block)                                              \
    {                                                                          \
        auto *_ = new TimeMonitor::ScopedCumulator(name);                      \
        { _block }                                                             \
        delete _;                                                              \
    };
} // namespace oclk
#endif // OPENCL_DEMO_TIMER_H
