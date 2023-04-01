from collections import defaultdict
from pprint import pp
from time import time_ns

import matplotlib.pyplot as plot
import numpy as np
import pyopencl as cl
import pyopencl.cltypes as cltype
from tqdm import tqdm


def load_kernel_file(filename):
    with open(filename) as f:
        return f.read()


def binary_round_up(v: int, round_v: int) -> int:
    return (int(v) + round_v - 1) & (-round_v)


def main():
    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)
    mf = cl.mem_flags

    local_size = (32,)
    prg = cl.Program(ctx, load_kernel_file("./kernels/add.cl")).build(
        options="-DBINARY_OP_STRIDE=8"
    )

    def run_kernel_and_time(kernel_name, global_size=None):
        t0 = time_ns()
        prg.__getattr__(kernel_name)(
            queue,
            global_size,
            local_size,
            buf_a,
            buf_b,
            buf_c,
            cltype.int(a_np.shape[0]),
        )
        t1 = time_ns()
        res_np = np.empty_like(a_np)
        cl.enqueue_copy(queue, res_np, buf_c)
        # Check on CPU with Numpy:
        assert np.allclose(res_np, cpu_result), f"result wrong in kernel {kernel_name}"
        duration = t1 - t0
        return duration

    duration_log = defaultdict(list)  # kernel_name : time list[]
    for i in tqdm(range(1000)):
        data_size = binary_round_up((i + 1) * 1000, 32)
        data_length = data_size
        for t in range(10):
            a_np = np.random.rand(data_length).astype(np.float32)
            b_np = np.random.rand(data_length).astype(np.float32)

            cpu_result = a_np + b_np

            buf_a = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a_np)
            buf_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b_np)
            buf_c = cl.Buffer(ctx, mf.WRITE_ONLY, a_np.nbytes)

            global_work_size = a_np.shape[0]
            duration_log["binary_op_vec"].append(
                (
                    run_kernel_and_time(
                        "binary_op_vec", (binary_round_up(global_work_size // 4, 32),)
                    ),
                    data_length,
                )
            )
            duration_log["binary_op_vec_stride"].append(
                (
                    run_kernel_and_time(
                        "binary_op_vec_stride", (int(global_work_size),)
                    ),
                    data_length,
                )
            )
            duration_log["binary_op_naive"].append(
                (
                    run_kernel_and_time("binary_op_naive", (int(global_work_size),)),
                    data_length,
                )
            )
            duration_log["binary_op_stride"].append(
                (
                    run_kernel_and_time("binary_op_stride", (int(global_work_size),)),
                    data_length,
                )
            )
            del (buf_a, buf_b, buf_c, a_np, b_np)

    scatter_color = {
        "binary_op_vec": "green",
        "binary_op_vec_stride": "lightcoral",
        "binary_op_naive": "orange",
        "binary_op_stride": "navy",
    }
    for k, v in duration_log.items():
        print(f"{k:20s} {np.asarray(v)[:,0].mean():.4f}")
    for k, v in duration_log.items():
        # print(f"{k:20s} {np.asarray(v).mean():.4f}")
        v = np.asarray(v)
        xdata = v[:, 1]
        ydata = v[:, 0]
        plot.scatter(xdata, ydata, s=4, c=scatter_color[k], label=k)

        # plot.cla()
    plot.legend(
        scatterpoints=1,
    )
    plot.ylabel("time(ns)")
    plot.xlabel("array length")
    plot.ylim([1.4e5, 1e6])
    plot.savefig(f"add_kernel.png")


if __name__ == "__main__":
    main()
