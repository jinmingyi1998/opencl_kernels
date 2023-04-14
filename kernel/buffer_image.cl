__constant sampler_t SAMPLER =
    CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
__kernel void buffer_image_converter_b2i_float32_TO_int8(
    __global float *arr, long offset, __write_only image2d_t out) {
    const int x      = get_global_id(0);
    const int y      = get_global_id(1);
    const int width  = get_image_width(out);
    const int height = get_image_height(out);
    if (x >= width || y >= height) return;

    arr += offset;
    int2 pos;
    pos.x   = x;
    pos.y   = y;
    uint4 v = (uint4)((uint)arr[mad24(width, y, x)], 0, 0, 0);
    write_imageui(out, pos, v);
}