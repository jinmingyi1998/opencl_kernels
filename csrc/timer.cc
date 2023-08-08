//
// Created by jimmy on 23-4-13.
//

#include "timer.h"
#include <cmath>
#include <numeric>
namespace oclk {
std::mutex TimeMonitor::mMutex;
std::unordered_map<std::string, double> TimeMonitor::mMonitorTotalData;
std::unordered_map<std::string, std::vector<double>>
    TimeMonitor::mMonitorTotalDataDetail;
std::unordered_map<std::string, int> TimeMonitor::mMonitorTotalDataCnt;
int TimeMonitor::Init(const std::string &name) {
    mMonitorTotalDataCnt[name]    = 0;
    mMonitorTotalData[name]       = 0;
    mMonitorTotalDataDetail[name] = std::vector<double>();
    mMonitorTotalDataDetail[name].reserve(100);
    return 0;
}
static double calc_stdev(std::vector<double> v) {
    double sum  = std::accumulate(v.begin(), v.end(), 0.0);
    double mean = sum / v.size();

    double sq_sum = std::inner_product(v.begin(), v.end(), v.begin(), 0.0);
    double stdev  = std::sqrt(sq_sum / v.size() - mean * mean);
    return stdev;
}
static std::string stringify(double val) {
    char buf[100];
    sprintf(buf, "%lf", val);
    return buf;
}

int TimeMonitor::AddData(const std::string &name, const double &data) {
    const std::lock_guard<std::mutex> sLock(mMutex);
    if (mMonitorTotalDataCnt.count(name) == 0) Init(name);

    mMonitorTotalData[name] += data;
    mMonitorTotalDataDetail[name].push_back(data);
    ++mMonitorTotalDataCnt[name];
    spdlog::debug("[{}][CUR: {:4.3f}ms][MOVING AVG: {:4.3f}ms][CNT: {:3d}]",
                  name,
                  data,
                  mMonitorTotalData[name] / mMonitorTotalDataCnt[name],
                  mMonitorTotalDataCnt[name]);
    return 0;
}
int TimeMonitor::ShowTimer(const std::string &name) {
    const std::lock_guard<std::mutex> sLock(mMutex);
    auto iter = mMonitorTotalData.find(name);
    if (iter == mMonitorTotalData.end()) {
        spdlog::error("Timer [{}] not found!", name);
        return 1;
    }

    spdlog::info("[Timer {}] [CNT: {}] [AVG: {:4.3f}ms] [STDEV {:4.3f}ms] "
                 "[TOTAL {:4.3f}ms]",
                 iter->first,
                 mMonitorTotalDataCnt[iter->first],
                 mMonitorTotalData[iter->first] /
                     mMonitorTotalDataCnt[iter->first],
                 calc_stdev(mMonitorTotalDataDetail[iter->first]),
                 mMonitorTotalData[iter->first]);
    return 0;
}

/**
 * return a object:
 * {
 *      "name": "",
 *      "count":"", // the number of calls
 *      "average": 1.23f,
 *      "std": 1.23f,
 *      "total": 1.23f
 * }
 * @param name
 * @return
 */
int TimeMonitor::DumpObj(const std::string &name) { return 0; }
int TimeMonitor::ShowAll() {
    const std::lock_guard<std::mutex> sLock(mMutex);
    for (auto iter = mMonitorTotalData.begin(); iter != mMonitorTotalData.end();
         ++iter) {
        spdlog::info("[Timer {}] [CNT: {}] [AVG: {:4.3f}ms] [STDEV {:4.3f}ms] "
                     "[TOTAL {:4.3f}ms]",
                     iter->first,
                     mMonitorTotalDataCnt[iter->first],
                     mMonitorTotalData[iter->first] /
                         mMonitorTotalDataCnt[iter->first],
                     calc_stdev(mMonitorTotalDataDetail[iter->first]),
                     mMonitorTotalData[iter->first]);
    }
    return 0;
}
int TimeMonitor::Clear() {
    mMonitorTotalDataCnt.clear();
    mMonitorTotalData.clear();
    mMonitorTotalDataDetail.clear();
    return 0;
}

TimeMonitor::ScopedCumulator::ScopedCumulator(const std::string &name,
                                              double ratio) {
    mName      = name;
    mRatio     = ratio;
    mStartTime = time_ns() * 1e-6;
}

TimeMonitor::ScopedCumulator::~ScopedCumulator() {
    double time = time_ns() * 1e-6 - mStartTime;
    time *= mRatio;
    AddData(mName, time);
}
TimerArgs::TimerArgs(bool Enable,
                     unsigned long Warmup,
                     unsigned long Repeat,
                     const std::string &TimerName)
    : warmup(Warmup)
    , repeat(Repeat)
    , enable(Enable)
    , timer_name(TimerName) { }
unsigned long TimerArgs::getWarmup() const { return warmup; }
unsigned long TimerArgs::getRepeat() const { return repeat; }
bool TimerArgs::isEnable() const { return enable; }
const std::string &TimerArgs::getTimerName() const { return timer_name; }
// void TimerArgs::setWarmup(unsigned long Warmup) { warmup = Warmup; }
// void TimerArgs::setRepeat(unsigned long Repeat) { repeat = Repeat; }
// void TimerArgs::setEnable(bool Enable) { enable = Enable; }
// void TimerArgs::setTimerName(const std::string &TimerName) {
//     timer_name = TimerName;
// }
} // namespace oclk