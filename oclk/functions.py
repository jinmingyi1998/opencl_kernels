from typing import Dict, List, Optional, Union, Tuple

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
    input: List[Dict[str, Union[str, int, float, List[Dict], np.array]]],
    local_work_size: Union[List[int],Tuple[int]],
    global_work_size: Union[List[int],Tuple[int]],
    output: Optional[List[str]] = None,
    wait: bool = True,
    timer: Dict = None,
) -> _C.RunnerReturn:
    if not (isinstance(local_work_size, (list,tuple)) and isinstance(local_work_size[0], int)):
        raise TypeError("local_work_size type must be List[int]")
    if not (
        isinstance(global_work_size, (list,tuple)) and isinstance(global_work_size[0], int)
    ):
        raise TypeError("global_work_size type must be List[int]")
    if len(local_work_size) != len(global_work_size):
        raise ValueError(
            f"local_work_size, global_work_size must has the same length, got {len(local_work_size)} , {len(global_work_size)}"
        )
    if not isinstance(output, list):
        raise TypeError(f"output must be a list, got {type(output)}")
    for s in output:
        if not isinstance(s, str):
            raise TypeError(f"output must be list of str, got {s}:{type(s)}")
    if not isinstance(input, list):
        raise TypeError(f"input must be list, got {type(input)}")
    for d in input:
        if not isinstance(d["name"], str):
            raise TypeError(f"name must be str, got {type(d['name'])}")
        if not isinstance(d["value"], (int, float, list, np.ndarray)):
            raise TypeError(f"value must be one of int, float, list, np.ndarray")
        if "type" in d:
            if d["type"] not in [
                "float",
                "int",
                "unsigned int",
                "double",
                "long",
                "unsigned long",
            ]:
                raise ValueError(
                    f"'type' must be 'float','int','unsigned int','double','long','unsigned long', but got {d['type']}"
                )

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
