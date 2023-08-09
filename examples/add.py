import os
from typing import Dict, List, Union

import numpy as np

from oclk import Runner, TimerArgs


def wrap_args(**kwargs) -> List[Dict[str, Union[str, np.ndarray, int, float]]]:
    """
    easily make the arg dict
    """
    return [{"name": k, "value": v} for k, v in kwargs.items()]


def add():
    r = Runner()
    a = np.random.random([64, 64])
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.random.random(a.shape)
    b = np.ascontiguousarray(b, dtype=np.float32)

    out = np.zeros_like(a)
    out = np.ascontiguousarray(out, dtype=np.float32)

    arr_length = a.size

    result = r.run(
        kernel_name="add",
        input=[
            {"name": "a", "value": a},
            {"name": "b", "value": b},
            {"name": "length", "value": arr_length, "type": "int"},
            {"name": "out", "value": out},
        ],
        output=["out"],
        local_work_size=[1],
        global_work_size=[arr_length],
    )
    print(result.timer_result)  # not set timer, so there is an empty timer_result

    c = np.float32(a + b)
    print(np.abs(c - out).max())


def add_constant():
    r = Runner()

    a = np.random.random([1, 2, 3, 4, 5])
    x = 12.3

    a = np.ascontiguousarray(a, dtype=np.float32)
    arr_length = a.size
    out = np.zeros_like(a)
    out = np.ascontiguousarray(out, dtype=np.float32)

    timer = TimerArgs(True, 1, 10, "add_constant")
    rtn = r.run(
        kernel_name="add_constant",
        input=wrap_args(a=a, x=x, length=arr_length, out=out),
        output=["out"],
        local_work_size=[1],
        global_work_size=[arr_length],
        wait=True,
        timer=timer,
    )
    print(rtn.timer_result)

    b = np.float32(a + x)
    print(np.abs(out - b).max())


def add_batch():
    r = Runner()

    a = np.random.random([1, 2, 3, 4, 5])
    b = np.random.random([1, 2, 3, 4, 5])
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    out = np.zeros_like(a)
    out = np.ascontiguousarray(out, dtype=np.float32)
    arr_length = a.size
    timer = TimerArgs(True, 10, 100, "add_batch")
    result = r.run(
        kernel_name="add_batch",
        input=wrap_args(a=a, b=b, length=arr_length, out=out),
        output=["out"],
        local_work_size=[1],
        global_work_size=[arr_length // 4],
        wait=True,
        timer=timer,
    )
    print(result.timer_result)

    a = np.float32(a)
    b = np.float32(b)
    c = a + b
    c = np.float32(c)
    print(np.abs(out - c).max())


def main():
    r = Runner()  # first init
    kernel_file = os.path.join(os.path.dirname(__file__), "add.cl")
    r.load_kernel(kernel_file, ["add", "add_constant"])
    r.load_kernel(kernel_file, "add_batch", "-DBATCH_SIZE=4")

    add()
    add_constant()
    add_batch()
    r.release_kernel("add")
    r.release_kernel("add_constant")
    r.release_kernel("add_batch")


if __name__ == "__main__":
    main()
