//
// Created by jimmy on 23-4-13.
//

#include "timer.h"
#include "csv2.hpp"
#include <cmath>
#include <numeric>
namespace oclk {
static std::mutex mMutex;
static std::unordered_map<std::string, double> mMonitorTotalData;
static std::unordered_map<std::string, std::vector<double>>
    mMonitorTotalDataDetail;
static std::unordered_map<std::string, int> mMonitorTotalDataCnt;
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

    LOG(INFO) << "[" << name << "] [CUR: " << stringify(data) << " ms] "
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

int TimeMonitor::DumpCSV(const std::string &csvOutputFilename) {
    const std::lock_guard<std::mutex> sLock(mMutex);
    std::ofstream fout(csvOutputFilename);
    std::vector<std::vector<std::string>> rows;
    for (auto &iter : mMonitorTotalDataDetail) {
        std::vector<std::string> row;
        row.push_back(iter.first);
        for (auto v : iter.second) {
            row.push_back(stringify(v));
        }
        rows.push_back(row);
    }
    csv2::Writer writer(fout);
    writer.write_rows(rows);
    return 0;
}

TimeMonitor::ScopedCumulator::ScopedCumulator(const std::string &name) {
#ifdef ENABLE_TIMER
    mName      = name;
    mStartTime = time_us() * 1e-3;
#endif // ENABLE_TIMER
}

TimeMonitor::ScopedCumulator::~ScopedCumulator() {
#ifdef ENABLE_TIMER
    double time = time_us() * 1e-3 - mStartTime;
    AddData(mName, time);
#endif // ENABLE_TIMER
}
} // namespace oclk