//
// Created by jimmy on 23-4-14.
//

#ifndef OPENCL_DEMO_HELPER_H
#define OPENCL_DEMO_HELPER_H
#include <map>
#include <sstream>
#include <string>
#include <vector>
template <typename T> std::string stringify(T val) {
    std::stringstream ss;
    ss << val;
    std::string s = ss.str();
    return s;
}
template <typename T>
std::string parse_fields_to_name(std::vector<std::string> &key,
                                 std::vector<T> &values) {
    std::string name = "";
    for (int i = 0; i < key.size(); i++) {
        name.append(key.at(i));
        name.append("@");
        name.append(stringify(values.at(i)));
        if (i != key.size() - 1) name.append("/");
    }
    return name;
}
/**
 * accept a kv, parse to a string representation
 * @tparam T
 * @param kv
 * @return
 */
template <typename T>
std::string parse_fields_to_name(std::map<std::string, T> &kv) {
    std::string name = "";
    if (kv.size() < 1) {
        return "";
    }
    for (auto it = kv.begin(); it != kv.end(); it++) {
        name.append(it->first);
        name.append("@");
        name.append(stringify(it->second));
        name.append("/");
    }
    name = name.substr(0, name.length() - 1); // remove the last slash
    return name;
}
#endif // OPENCL_DEMO_HELPER_H
