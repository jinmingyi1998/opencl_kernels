from oclk import Runner, input_maker, RunnerCtx
from oclk.benchmark import benchmark
import numpy as np
import os

kernel_file = os.path.join(os.path.dirname(__file__), "add.cl")


def test_add():
    a = np.ascontiguousarray(np.random.random([16, 16]), np.float32)
    b = np.ascontiguousarray(np.random.random([16, 16]), np.float32)
    c = np.ascontiguousarray(a + b, np.float32)
    c_cpu = np.ascontiguousarray(a + b, np.float32)
    r = Runner()
    kernel_name = "add_float"
    r.load_kernel(kernel_file, kernel_name)
    r.run(
        kernel_name=kernel_name,
        input=input_maker(a=a, b=b, c=c, w=a.shape[0], h=a.shape[1]),
        output=["c"],
        global_work_size=list(a.shape),
        local_work_size=[1, 1],
    )
    r.release_kernel(kernel_name)
    assert np.linalg.norm(c - c_cpu).sum() < 1e-6


def test_add_int():
    dtype = np.int32
    a = np.ascontiguousarray(np.random.random([16, 16]), dtype)
    b = np.ascontiguousarray(np.random.random([16, 16]), dtype)
    c = np.ascontiguousarray(a + b, dtype)
    c_cpu = np.ascontiguousarray(a + b, dtype)
    r = Runner()
    kernel_name = "add_int"
    r.load_kernel(kernel_file, kernel_name)
    r.run(
        kernel_name=kernel_name,
        input=input_maker(a=a, b=b, c=c, limit=(a.size, "int")),
        output=["c"],
        global_work_size=[a.size],
        local_work_size=[4],
    )
    r.release_kernel(kernel_name)
    assert np.linalg.norm(c - c_cpu).sum() < 1e-6


def test_set_macro():
    dtype = np.int32
    a = np.ascontiguousarray(np.zeros([32, 32]), dtype)
    b = np.ascontiguousarray(np.ones_like(a) * 56723, dtype)
    r = Runner()
    kernel_name = "set_value_macro"
    r.load_kernel(kernel_file, kernel_name, "-DVALUE=56723")
    r.run(
        kernel_name=kernel_name,
        input=input_maker(a=a, limit=(a.size, "int")),
        output=["a"],
        global_work_size=[a.size],
        local_work_size=[4],
    )
    r.release_kernel(kernel_name)
    assert np.linalg.norm(b - a).sum() < 1e-6


def test_set_value():
    dtype = np.float32
    a = np.ascontiguousarray(np.zeros([32, 32]), dtype)
    b = np.ascontiguousarray(np.ones_like(a) * 133, dtype)
    r = Runner()
    kernel_name = "set_value"
    r.load_kernel(kernel_file, kernel_name)
    r.run(
        kernel_name=kernel_name,
        input=input_maker(a=a, v=(133, "float"), limit=(a.size, "int")),
        output=["a"],
        global_work_size=[a.size],
        local_work_size=[4],
    )
    r.release_kernel(kernel_name)
    assert np.linalg.norm(b - a).sum() < 1e-6


def test_set_struct():
    dtype = np.int32
    a = np.ascontiguousarray(np.zeros([4, 4, 4]), dtype)
    b = np.ascontiguousarray(np.arange(4 * 4 * 4).reshape([4, 4, 4]), dtype)
    kernel_name = "set_struct"
    with RunnerCtx(kernel_file, kernel_name) as r:
        r.run(
            kernel_name=kernel_name,
            input=input_maker(
                a=a,
                shape=[
                    {"value": 4, "type": "int"},
                    {"value": 4, "type": "int"},
                    {"value": 4, "type": "int"},
                    {"value": 0, "type": "int"},
                ],
            ),
            output=["a"],
            global_work_size=[4, 4, 4],
            local_work_size=[4, 1, 1],
        )
    assert np.linalg.norm(b - a).sum() < 1e-6


def test_bench():
    benchmark("test/bench_add.yaml")
