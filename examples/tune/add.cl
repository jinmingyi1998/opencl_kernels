
#ifndef VECTOR_SIZE
#define VECTOR_SIZE 4
#endif

#ifndef TILE_SIZE
#define TILE_SIZE 4
#endif

#define DTYPE_V_CONN_(len__) float##len__
#define DTYPE_V(x__) DTYPE_V_CONN_(x__)
#define VLOAD_LEN_(len__) vload##len__
#define VSTORE_LEN_(len__) vstore##len__
#define VLOAD_LEN(len__) VLOAD_LEN_(len__)
#define VSTORE_LEN(len__) VSTORE_LEN_(len__)

#define VEC DTYPE_V(VECTOR_SIZE)

#define VLOAD VLOAD_LEN(VECTOR_SIZE)
#define VSTORE VSTORE_LEN(VECTOR_SIZE)

kernel void add(global float *a, global float *b, global float *c, int length) {
    int x = get_global_id(0);
    if (x < length) {
        c[x] = a[x] + b[x];
    }
}

kernel void
add_vector(global float *a, global float *b, global float *c, int length) {
    int x = get_global_id(0) * VECTOR_SIZE;
    if (x < length) {
        if (x + VECTOR_SIZE > length) {
            x = length - VECTOR_SIZE;
        }
        VSTORE(VLOAD(0, a + x) + VLOAD(0, b + x), 0, c + x);
    }
}

kernel void
add_tile(global float *a, global float *b, global float *c, int length) {
    int x = get_global_id(0) * TILE_SIZE;
    for (int i = 0; i < TILE_SIZE && x + i < length; i++) {
        c[x + i] = a[x + i] + b[x + i];
    }
}