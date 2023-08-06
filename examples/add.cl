kernel void
add(global float *a, global float *b, int length, global float *out) {
    int x = get_global_id(0);
    if (x < length) {
        out[x] = a[x] + b[x];
    }
}

kernel void
add_constant(global float *a, float x, long length, global float *out) {
    long i = get_global_id(0);
    if (i < length) {
        out[i] = a[i] + x;
    }
}

#ifndef BATCH_SIZE
#define BATCH_SIZE 4
#endif
kernel void
add_batch(global float *a, global float *b, long length, global float *out) {
    long x = get_global_id(0) * BATCH_SIZE;
    for (int i = 0; i < BATCH_SIZE && x < length; i++, x++) {
        out[x] = a[x] + b[x];
    }
}