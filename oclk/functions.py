import sys
from typing import Dict, List, Union

import numpy as np

try:
    import oclk.oclk_C as _C

    init = _C.init
    load_kernel = _C.load_kernel
except ImportError:
    import sys

    print("WARN:  there is not C module!", file=sys.stderr)

    # make a dummy module to avoid Exception, useful for sphinx-apidoc
    class DummyC:
        def load_kernel(self, *args, **kwargs):
            ...

        def init(self, *args, **kwargs):
            ...

    _C = DummyC
    init = _C.init
    load_kernel = _C.load_kernel


def init() -> int:
    err = _C.init()
    assert err == 0
    return err


def loak_kernel(
    cl_file: str, kernel_name: str, compile_option: Union[str, List[str]]
) -> int:
    if not isinstance(compile_option, str):
        assert isinstance(compile_option, list)
        compile_option = " ".join(compile_option)
    err = _C.load_kernel(cl_file, kernel_name, compile_option)
    assert err == 0
    return err


def release_kernel(kernel_name: str) -> int:
    return _C.release_kernel(kernel_name)


class TimerArgs:
    def __init__(self, enable: bool, warmup: int, repeat: int, name: str):
        self.enable = enable
        self.warmup = warmup
        self.repeat = repeat
        self.name = name

    def __dict__(self):
        return {
            "enable": self.enable,
            "warmup": self.warmup,
            "repeat": self.repeat,
            "name": self.name,
        }


TimerArgsDisabled = TimerArgs(False, 0, 0, "no-name")


def run(
    *,
    kernel_name: str,
    input: List[Dict[str, Union[int, float, np.array]]],
    output: List[str],
    local_work_size: List[int],
    global_work_size: List[int],
    wait: bool = True,
    timer: Union[Dict, TimerArgs] = TimerArgsDisabled
) -> List[np.ndarray]:
    timer_dict = timer
    if isinstance(timer, TimerArgs):
        timer_dict = timer.__dict__()
    return _C.run(
        kernel_name=kernel_name,
        input=input,
        output=output,
        local_work_size=local_work_size,
        global_work_size=global_work_size,
        wait=wait,
        timer=timer_dict,
    )
