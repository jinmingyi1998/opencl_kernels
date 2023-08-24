kernel void
add_float(global float *a, global float *b, global float *c, long w, long h) {
    long i = get_global_id(0);
    long j = get_global_id(1);
    if (i < w && j < h) {
        long idx = j * w + i;
        c[idx] = b[idx] + a[idx];
    }
}

kernel void add_int(global int *a, global int *b, global int *c, int limit) {
    int i = get_global_id(0);
    if (i < limit) {
        c[i] = b[i] + a[i];
    }
}

#ifndef VALUE
#define VALUE 123
#endif
kernel void set_value_macro(global int *a, int limit) {
    int i = get_global_id(0);
    if (i < limit) {
        a[i] = VALUE;
    }
}
#undef VALUE

kernel void set_value(global float *a, float v, int limit) {
    int i = get_global_id(0);
    if (i < limit) {
        a[i] = v;
    }
}

typedef struct SHAPE{
    int s[4];
}SHAPE;
kernel void set_struct(global int*a, SHAPE shape){
    int x = get_global_id(0);
    int y = get_global_id(1);
    int z = get_global_id(2);
    if(x < shape.s[0] && y < shape.s[1] && z < shape.s[2]){
        int idx = x+y*shape.s[0]+z*shape.s[1]*shape.s[0];
        a[idx] = idx;
    }
}
