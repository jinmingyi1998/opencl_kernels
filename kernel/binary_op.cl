#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifndef CL_DTYPE
#define CL_DTYPE float
#endif

#ifndef VECTOR_LENGTH
#define VECTOR_LENGTH 4
#endif

#ifndef BINARY_OP_STRIDE
#define BINARY_OP_STRIDE 4
#endif

#ifndef BINARY_OP_OPT
#define BINARY_OP_OPT NOPE
#endif

#define GET_VEC_TYPE(type__, size__) type__##size__
#define VECTORIZED_TYPE(type__, size__) GET_VEC_TYPE(type__, size__)
#define CL_DTYPE4 VECTORIZED_TYPE(CL_DTYPE, 4)
#define CL_DTYPE8 VECTORIZED_TYPE(CL_DTYPE, 8)
#define CL_DTYPE16 VECTORIZED_TYPE(CL_DTYPE, 16)

#define VLOAD_LEN(len__) vload##len__
#define VSTORE_LEN(len__) vstore##len__
#define VLOAD_LENGTH(len__) VLOAD_LEN(len__)
#define VSTORE_LENGTH(len__) VSTORE_LEN(len__)

#define STR_SUFFIX(a, b) a##_##b
#define FUNCNAME_SUFFIX(func, suffix) STR_SUFFIX(func, suffix)

#define GEN_OPERATOR(OPT, result)                                              \
    inline CL_DTYPE solve_##OPT(CL_DTYPE a, CL_DTYPE b) { return result; }

#define GEN_OPERATOR_V4(OPT, result)                                           \
    inline CL_DTYPE4 solve_v4_##OPT(CL_DTYPE4 a, CL_DTYPE4 b) { return result; }

#define CALL_GEN_OPERATOR__(OPT, a, b) solve_##OPT(a, b)
#define CALL_GEN_OPERATOR(OPT, a, b) CALL_GEN_OPERATOR__(OPT, a, b)

#define CALL_GEN_OPERATOR_V4__(OPT, a, b) solve_v4_##OPT(a, b)
#define CALL_GEN_OPERATOR_V4(OPT, a, b) CALL_GEN_OPERATOR_V4__(OPT, a, b)

GEN_OPERATOR(NOPE, 0)
GEN_OPERATOR(ADD, a + b)
GEN_OPERATOR(SUB, a - b)
GEN_OPERATOR(MUL, a *b)
GEN_OPERATOR(DIV, a / b)
GEN_OPERATOR(POW, pow(a, b))
GEN_OPERATOR(MIN, (a < b ? a : b))
GEN_OPERATOR(MAX, (a < b ? b : a))

GEN_OPERATOR_V4(NOPE, 0)
GEN_OPERATOR_V4(ADD, a + b)
GEN_OPERATOR_V4(SUB, a - b)
GEN_OPERATOR_V4(MUL, a *b)
GEN_OPERATOR_V4(DIV, a / b)
GEN_OPERATOR_V4(POW, pow(a, b))
GEN_OPERATOR_V4(MIN, min(a,b))
GEN_OPERATOR_V4(MAX, max(a,b))

kernel void FUNCNAME_SUFFIX(binary_op_naive, BINARY_OP_OPT)(
    global const float *a, global const float *b, global float *c, int length) {
    int idx = get_global_id(0);
    if (idx >= length) return;
    // c[idx] = process_operator(a[idx], b[idx]);
    c[idx] = CALL_GEN_OPERATOR(BINARY_OP_OPT, a[idx], b[idx]);
}

kernel void FUNCNAME_SUFFIX(binary_op_stride, BINARY_OP_OPT)(
    global const float *a, global const float *b, global float *c, int length) {
    int idx       = get_global_id(0);
    int start_idx = idx * BINARY_OP_STRIDE;
    float res[BINARY_OP_STRIDE];
    if (start_idx >= length) return;
    for (int i = 0; i < BINARY_OP_STRIDE; i++) {
        // res[i] = process_operator(a[start_idx + i], b[start_idx + i]);
        res[i] = CALL_GEN_OPERATOR(
            BINARY_OP_OPT, a[start_idx + i], b[start_idx + i]);
    }
    for (int i = 0; i < BINARY_OP_STRIDE; i++) {
        c[i + start_idx] = res[i];
    }
}
kernel void FUNCNAME_SUFFIX(binary_op_vec_stride, BINARY_OP_OPT)(
    global const float *a, global const float *b, global float *c, int length) {
    // process 4[stride] * 4[float4] = 16 elements
    int idx       = get_global_id(0);
    int start_idx = idx * BINARY_OP_STRIDE * VECTOR_LENGTH;
    if (start_idx >= length) return;
    CL_DTYPE4 res[BINARY_OP_STRIDE];
    for (int i = 0; i < BINARY_OP_STRIDE; i++) {
        CL_DTYPE4 va = VLOAD_LENGTH(VECTOR_LENGTH)(i, a + start_idx);
        CL_DTYPE4 vb = VLOAD_LENGTH(VECTOR_LENGTH)(i, b + start_idx);
        res[i]       = CALL_GEN_OPERATOR_V4(BINARY_OP_OPT, va, vb);
    }
    for (int i = 0; i < BINARY_OP_STRIDE; i++) {
        VSTORE_LENGTH(VECTOR_LENGTH)(res[i], i, c + start_idx);
    }
}
kernel void FUNCNAME_SUFFIX(binary_op_vec, BINARY_OP_OPT)(global const float *a,
                                                          global const float *b,
                                                          global float *c,
                                                          int length) {
    int idx       = get_global_id(0);
    int start_idx = idx * VECTOR_LENGTH;
    if (start_idx >= length) return;
    CL_DTYPE4 va = VLOAD_LENGTH(VECTOR_LENGTH)(0, a + start_idx);
    CL_DTYPE4 vb = VLOAD_LENGTH(VECTOR_LENGTH)(0, b + start_idx);
    CL_DTYPE4 vc = CALL_GEN_OPERATOR_V4(BINARY_OP_OPT, va, vb);
    VSTORE_LENGTH(VECTOR_LENGTH)(vc, 0, c + start_idx);
}