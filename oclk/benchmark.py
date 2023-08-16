from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import typer
import yaml

from oclk.benchmark_config import (
    ArgValueGenerator,
    Kernel,
    KernelArg,
    Suite,
    arg_support_type,
    dict_to_Suite,
)
from oclk.oclk_runner import Runner, TimerArgs

app = typer.Typer()


@contextmanager
def benchmark_run(filename, kernel_name, compile_option=""):
    r = Runner()
    r.load_kernel(filename, kernel_name, compile_option)
    try:
        yield r
    finally:
        r.release_kernel(kernel_name)


def run_suite(suite: Suite):
    for k in suite.kernels:
        with benchmark_run(suite.kernel_file, k.name, k.definition) as r:
            assert r is not None
            input = []
            for i, arg in enumerate(k.args):
                d = {"name": arg.name if arg.name else f"arg#{i}"}
                if arg.type not in arg_support_type:
                    raise ValueError(
                        f"arg.type must be int, float, array, got {arg.type}"
                    )
                if arg.type == "array":
                    if arg.value.method == "random":
                        v = np.random.random(arg.shape)
                        if (
                            isinstance(arg.value.value, list)
                            and len(arg.value.value) == 2
                        ):
                            start, end = arg.value.value[0], arg.value.value[1]
                            v = v * (end - start) - start
                    else:
                        v = np.ones(arg.shape, dtype=np.dtype(arg.dtype))
                        v *= arg.value.value
                    v = np.ascontiguousarray(v, dtype=np.dtype(arg.dtype))
                    d["value"] = v
                else:
                    if arg.value.method == "random":
                        if arg.type in ["float", "double"]:
                            v = np.random.rand()
                            if (
                                isinstance(arg.value.value, list)
                                and len(arg.value.value) == 2
                            ):
                                start, end = arg.value.value[0], arg.value.value[1]
                                v = v * (end - start) - start
                        else:
                            if (
                                isinstance(arg.value.value, list)
                                and len(arg.value.value) == 2
                            ):
                                start, end = arg.value.value[0], arg.value.value[1]
                                v = np.random.randint(start, end + 1)
                            else:
                                v = np.random.randint(0, 65535)
                    else:
                        v = arg.value.value
                    d["value"] = v
                    d["type"] = arg.type
                input.append(d)

            timer = TimerArgs(
                True,
                name=".".join([suite.timer.prefix, k.name, k.suffix]),
                warmup=suite.timer.warmup,
                repeat=suite.timer.repeat,
            )
            rtn = r.run(
                kernel_name=k.name,
                input=input,
                output=[],
                local_work_size=k.local_work_size,
                global_work_size=k.global_work_size,
                wait=True,
                timer=timer,
            )
            print(rtn.timer_result)


@app.command()
def benchmark(config_file: str):
    with open(config_file, "r") as f_cfg:
        cfg: List = yaml.safe_load(f_cfg.read())
    for suite_cfg in cfg:
        suite: Suite = dict_to_Suite(suite_cfg)
        run_suite(suite)


if __name__ == "__main__":
    app()
