//
// Created by jimmy on 23-4-13.
//

#ifndef OPENCL_DEMO_TIMER_H
#define OPENCL_DEMO_TIMER_H

#include <algorithm>
#include <ios>
#include <map>
#include <mutex>
#include <thread>
#include <unordered_map>

#include "common.h"
namespace oclk {
class TimerResult;
class TimeMonitor {
public:
    static int Init(const std::string &name);
    static int AddData(const std::string &name, const double &data);
    static int ShowAll();
    static int ShowTimer(const std::string &name);
    static TimerResult GetTimerResultObj(const std::string &name);
    static int Clear();

public:
    /**
     * 初始化的时候记录初始时间，变量离开作用域以后会做统计
     * 多线程中可以使用。
     */
    class ScopedCumulator final {
    public:
        ScopedCumulator(const std::string &name, double ratio = 1.0);
        ~ScopedCumulator();

    private:
        std::string mName;
        double mStartTime;
        double mRatio = 1.0;
    };

public:
    static std::mutex mMutex;
    static std::unordered_map<std::string, double> mMonitorTotalData;
    static std::unordered_map<std::string, std::vector<double>>
        mMonitorTotalDataDetail;
    static std::unordered_map<std::string, int> mMonitorTotalDataCnt;
};
inline long int time_s() {
    return std::chrono::duration_cast<std::chrono::seconds>(
               std::chrono::high_resolution_clock::now().time_since_epoch())
        .count();
}
inline long int time_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::high_resolution_clock::now().time_since_epoch())
        .count();
}
inline long int time_us() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::high_resolution_clock::now().time_since_epoch())
        .count();
}
inline long int time_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::high_resolution_clock::now().time_since_epoch())
        .count();
}
inline double time_ms_d() {
    double t = (double)time_us();
    return t / 1000.0;
}
#define TIMER_BLOCK(name, _block)                                              \
    do {                                                                       \
        TimeMonitor::ScopedCumulator _(name, 1.0 / _timer_repeat);             \
        { _block }                                                             \
    } while (0)

#define TIMER_BLOCK_REPEAT(name, _timer_repeat, _block)                        \
    [&]() {                                                                    \
        TimeMonitor::ScopedCumulator _(name, 1.0 / _timer_repeat);             \
        for (int __timer_repeat_19980313 = 0;                                  \
             __timer_repeat_19980313 < _timer_repeat;                          \
             __timer_repeat_19980313++) {                                      \
            _block                                                             \
        }                                                                      \
    }()
#define TIMER_KERNEL_BLOCK_REPEAT(name, _timer_repeat, commandQueue, _block)   \
    [&]() {                                                                    \
        TimeMonitor::ScopedCumulator _(name, 1.0 / _timer_repeat);             \
        for (int __timer_repeat_19980313 = 0;                                  \
             __timer_repeat_19980313 < _timer_repeat;                          \
             __timer_repeat_19980313++) {                                      \
            _block                                                             \
        }                                                                      \
        clFlush(commandQueue);                                                 \
        clFinish(commandQueue);                                                \
    }()

class TimerArgs {
private:
    bool enable            = false;
    unsigned long warmup   = 0;
    unsigned long repeat   = 1;
    std::string timer_name = "timer_dummy";

public:
    TimerArgs() = default;
    explicit TimerArgs(bool Enable,
                       unsigned long Warmup,
                       unsigned long Repeat,
                       const std::string &TimerName);
    unsigned long getWarmup() const;
    void setWarmup(unsigned long Warmup);
    unsigned long getRepeat() const;
    void setRepeat(unsigned long Repeat);
    bool isEnable() const;
    void setEnable(bool Enable);
    const std::string &getTimerName() const;
    void setTimerName(const std::string &TimerName);
};
const TimerArgs disabled_timer_arg(false, 0, 1, "timer_dummy");

class TimerResult {
public:
    TimerResult(const std::string &name)
        : name(name) { }
    TimerResult(const std::string &name,
                long cnt,
                double avg,
                double stdev,
                double total)
        : name(name)
        , cnt(cnt)
        , avg(avg)
        , stdev(stdev)
        , total(total) { }

    std::string ToString() const {
        int size_s =
            std::snprintf(nullptr,
                          0,
                          "[Timer %s] [CNT: %ld] [AVG: %.3lfms] [STDEV "
                          "%.3lfms] [TOTAL %.3lfms]",
                          name.c_str(),
                          cnt,
                          avg,
                          stdev,
                          total) +
            1; // Extra space for '\0'
        if (size_s <= 0) {
            throw std::runtime_error("Error during formatting.");
        }
        auto size = static_cast<size_t>(size_s);
        std::unique_ptr<char[]> buf(new char[size]);
        std::snprintf(buf.get(),
                      size,
                      "[Timer %s] [CNT: %ld] [AVG: %.3lfms] [STDEV "
                      "%.3lfms] [TOTAL %.3lfms]",
                      name.c_str(),
                      cnt,
                      avg,
                      stdev,
                      total);
        return std::string(buf.get(), buf.get() + size - 1);
    }

public:
    std::string name;
    long cnt     = 0;
    double avg   = 0.0;
    double stdev = 0.0;
    double total = 0.0;
};
const TimerResult no_result("no result", 0, 0, 0, 0);
};     // namespace oclk
#endif // OPENCL_DEMO_TIMER_H
