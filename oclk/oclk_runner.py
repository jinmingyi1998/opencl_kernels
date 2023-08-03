from functools import wraps
from typing import Dict, List, Union

import numpy as np

import oclk.functions as F


def check_init(fn):
    @wraps(fn)
    def wrapper(self: "Runner", *args, **kwargs):
        if not self.has_initialized:
            import sys

            print(f"ERROR: never init, check the error log before", file=sys.stderr)
            return
        return fn(self, *args, **kwargs)

    return wrapper


class Runner:
    has_initialized = False
    kernel_list = {}

    def __new__(cls):
        if not cls.has_initialized:
            F.init()
            cls.has_initialized = True
        return super().__new__(cls)

    @check_init
    def load_kernel(self, cl_file, kernel_name, compile_option):
        if kernel_name in self.kernel_list:
            import sys

            print(f"ERROR: kernel name {kernel_name} already exists!", file=sys.stderr)
            return
        kernel_name[kernel_name] = {
            "name": kernel_name,
            "file": cl_file,
            "compile_option": compile_option,
        }
        F.loak_kernel(cl_file, kernel_name, compile_option)

    @check_init
    def run(
        self,
        *,
        kernel_name: str,
        input: Dict[str, Union[int, float, np.array]],
        output: List[str],
        local_work_size: List[int],
        global_work_size: List[int],
        wait: bool = True,
        timer: Union[Dict, F.TimerArgs] = F.TimerArgsDisabled,
    ) -> List[np.ndarray]:
        return F.run(
            kernel_name, input, output, local_work_size, global_work_size, wait, timer
        )
