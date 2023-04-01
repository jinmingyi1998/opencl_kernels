#ifndef BINARY_OP_STRIDE
#define BINARY_OP_STRIDE 4
#endif

kernel void binary_op_naive(global const float *a,
                            global const float *b,
                            global float *c,
                            int length) {
  int idx = get_global_id(0);
  if (idx >= length) return;
  c[idx] = a[idx] + b[idx];
}

kernel void binary_op_vec_stride(global const float *a,
                              global const float *b,
                              global float *c,
                              int length) {
  // process 4[stride] * 4[float4] = 16 elements
  int idx = get_global_id(0);
  int start_idx = idx * BINARY_OP_STRIDE * 4;
  if (start_idx >= length) return;
  float4 res[BINARY_OP_STRIDE];
  for (int i = 0; i < BINARY_OP_STRIDE; i++) {
    float4 av = vload4(i, a + start_idx);
    float4 bv = vload4(i, b + start_idx);
    res[i] = av + bv;
  }
  for (int i = 0; i < BINARY_OP_STRIDE; i++) {
    vstore4(res[i], i, c + start_idx);
  }
}
kernel void binary_op_vec(global const float *a,
                          global const float *b,
                          global float *c,
                          int length) {
  int idx = get_global_id(0);
  int start_idx = idx * 4;
  if (start_idx >= length) return;
  float4 va = vload4(0, a + start_idx);
  float4 vb = vload4(0, b + start_idx);
  float4 vc = va + vb;
  vstore4(vc, 0, c + start_idx);
}
kernel void binary_op_stride(global const float *a,
                             global const float *b,
                             global float *c,
                             int length) {
  int idx = get_global_id(0);
  int start_idx = idx * BINARY_OP_STRIDE;
  float res[BINARY_OP_STRIDE];
  if (start_idx >= length) return;
  for (int i = 0; i < BINARY_OP_STRIDE; i++) {
    res[i] = a[start_idx + i] + b[start_idx + i];
  }
  for (int i = 0; i < BINARY_OP_STRIDE; i++) {
    c[i + start_idx] = res[i];
  }
}