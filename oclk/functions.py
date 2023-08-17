from typing import Dict, List, Optional, Union

import numpy as np

try:
    import oclk.oclk_C as _C

    init = _C.init
    load_kernel = _C.load_kernel
    clear_timer = _C.clear_timer
except ImportError:
    import sys

    # make a dummy module to avoid Exception, useful for sphinx-apidoc
    class DummyC:
        @staticmethod
        def load_kernel(*args, **kwargs):
            ...

        @staticmethod
        def init(*args, **kwargs):
            ...

        @staticmethod
        def clear_timer():
            ...

        @staticmethod
        def device_info():
            ...

        class RunnerReturn:
            ...

    _C = DummyC
    init = _C.init
    load_kernel = _C.load_kernel
    clear_timer = _C.clear_timer
    device_info = _C.device_info


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


def run(
    *,
    kernel_name: str,
    input: List[Dict[str, Union[int, float, np.array]]],
    local_work_size: List[int],
    global_work_size: List[int],
    output: Optional[List[str]] = None,
    wait: bool = True,
    timer: Dict = None,
) -> _C.RunnerReturn:
    if not (isinstance(local_work_size, list) and isinstance(local_work_size[0], int)):
        raise TypeError("local_work_size type must be List[int]")
    if not (
        isinstance(global_work_size, list) and isinstance(global_work_size[0], int)
    ):
        raise TypeError("global_work_size type must be List[int]")
    if timer is None:
        timer_dict = {}
    else:
        timer_dict = timer
    if output is None:
        output = []
    return _C.run(
        kernel_name=kernel_name,
        input=input,
        output=output,
        local_work_size=local_work_size,
        global_work_size=global_work_size,
        wait=wait,
        timer=timer_dict,
    )
