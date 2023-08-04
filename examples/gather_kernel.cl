typedef unsigned long ul;
kernel void gather_kernel(const global CL_DTYPE *in,
                          const global ul *index,
                          global CL_DTYPE *out,
                          const ul index_size,
                          const ul slice_rows,
                          const ul out_rows,
                          const ul out_cols) {
    ul x = get_global_id(0);
    ul y = get_global_id(1);
    if (y >= out_rows || x >= out_cols) return;
    ul in_row_idx = (y / index_size) * slice_rows + index[y % index_size];
    ul from_i = in_row_idx * out_cols + x;
    ul to_i = y * out_cols + x;
    out[to_i] = in[from_i];
}