import functools

import numpy as np

import oclk

oclk.init()
oclk.load_kernel("gather_kernel.cl", "gather_kernel", " -DCL_DTYPE=float ")

a = np.arange(9 * 4 * 5 * 6 * 7 * 8).reshape((9, 4, 5, 6, 7, 8))
a = np.float32(a)
index = np.array([0, 1], dtype=np.int64)
axis = 1
gt = a[:, index, ...]
out = np.zeros_like(gt)
gt = np.float32(gt)
out = np.float32(out)

slice_rows = a.shape[axis]
out_rows = functools.reduce(lambda x, y: x * y, gt.shape[: axis + 1])
out_cols = gt.size // out_rows
rtn = oclk.run(
    kernel_name="gather_kernel",
    input={
        "in": a,
        "index": index,
        "out": out,
        "index_size": index.size,
        "slice_rows": slice_rows,
        "out_rows": out_rows,
        "out_cols": out_cols,
    },
    output=["out"],
    local_work_size=[1, 1],
    global_work_size=gt.shape,
    timer=oclk.TimerArgs(True, 10, 1000, "gather"),
)

print("PYTHON OUTPUT", out.shape)
print("PYTHON OUTPUT", gt.shape)
print("PYTHON OUTPUT", out.dtype)
print("PYTHON OUTPUT", (gt - out).sum())
print(out.reshape([-1])[:16])
