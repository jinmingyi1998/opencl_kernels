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

    VLOG(2) << "[" << name << "] [CUR: " << stringify(data) << " ms] "
            << "[MOVING AVG: "
            << stringify(mMonitorTotalData[name] / mMonitorTotalDataCnt[name])
            << " ms] "
            << "[CNT: " << mMonitorTotalDataCnt[name] << "] " << std::endl;
    return 0;
}

int TimeMonitor::ShowAll() {
    const std::lock_guard<std::mutex> sLock(mMutex);
    for (auto iter = mMonitorTotalData.begin(); iter != mMonitorTotalData.end();
         ++iter) {
        LOG(ERROR) << std::fixed << "[TOTAL][" << iter->first << "] "
                   << "[CNT: " << mMonitorTotalDataCnt[iter->first] << "] "
                   << "[AVG: "
                   << stringify(mMonitorTotalData[iter->first] /
                                mMonitorTotalDataCnt[iter->first])
                   << " ms] "
                   << "[STDEV: "
                   << calc_stdev(mMonitorTotalDataDetail[iter->first])
                   << " ms] "
                   << "[TOTAL: " << stringify(mMonitorTotalData[iter->first])
                   << " ms]";
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
void TimerArgs::setWarmup(unsigned long Warmup) { warmup = Warmup; }
unsigned long TimerArgs::getRepeat() const { return repeat; }
void TimerArgs::setRepeat(unsigned long Repeat) { repeat = Repeat; }
bool TimerArgs::isEnable() const { return enable; }
void TimerArgs::setEnable(bool Enable) { enable = Enable; }
const std::string &TimerArgs::getTimerName() const { return timer_name; }
void TimerArgs::setTimerName(const std::string &TimerName) {
    timer_name = TimerName;
}
} // namespace oclk