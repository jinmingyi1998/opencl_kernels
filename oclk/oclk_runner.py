from functools import wraps
from typing import Dict, List, Optional, Union

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
    def load_kernel(
        self,
        cl_file: str,
        kernel_name: Union[str, List[str]],
        compile_option: Optional[Union[str, List[str]]] = None,
    ):
        if compile_option is None:
            compile_option = ""
        if isinstance(compile_option, list):
            compile_option = " ".join(compile_option)

        if isinstance(kernel_name, str):
            kernel_name = [kernel_name]
        for kn in kernel_name:
            if kn in self.kernel_list:
                import sys

                print(f"ERROR: kernel name {kn} already exists!", file=sys.stderr)
                return
            err = F.loak_kernel(cl_file, kn, compile_option)
            if err == 0:
                self.kernel_list[kn] = {
                    "name": kn,
                    "file": cl_file,
                    "compile_option": compile_option,
                }

    @check_init
    def run(
        self,
        *,
        kernel_name: str,
        input: List[Dict[str, Union[int, float, np.array]]],
        output: List[str],
        local_work_size: List[int],
        global_work_size: List[int],
        wait: bool = True,
        timer: Union[Dict, F.TimerArgs] = F.TimerArgsDisabled,
    ) -> List[np.ndarray]:
        return F.run(
            kernel_name=kernel_name,
            input=input,
            output=output,
            local_work_size=local_work_size,
            global_work_size=global_work_size,
            wait=wait,
            timer=timer,
        )
