//
// Created by jimmy on 23-4-3.
//

#ifndef OPENCL_DEMO_CALC_H
#define OPENCL_DEMO_CALC_H
namespace oclk {
/**
 * round up a value with an exp2{2,4,8,16,32,...} value
 * e.g. 58 round up with 32: get 64
 */
inline int binary_round_up(int value, int round_up_value) {
    return (value + round_up_value - 1) & (-round_up_value);
}
} // namespace oclk
#endif // OPENCL_DEMO_CALC_H
