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
    has_initialized: bool = False
    kernel_list: Dict[str, Dict[str, str]] = {}

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
        """
        Load kernel with filename and function name

        :param cl_file:         filename can be absolute or relative path
        :param kernel_name:     kernel_name is the kernel functions' name
        :param compile_option:  compile option can be strings like `-DMY_DEF=1`, **`-D` is necessary**
        """
        if compile_option is None:
            compile_option = ""
        if isinstance(compile_option, list):
            compile_option = " ".join(compile_option)

        if isinstance(kernel_name, str):
            kernel_name: List[str] = [kernel_name]
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
    def release_kernel(self, kernel_name: str) -> int:
        """
        unload kernel from context, kernel name cannot be duplicated.

        If you want to reload a kernel, you have to release it firstly.

        :param kernel_name:
        :return:
        """
        assert (
            kernel_name in self.kernel_list
        ), f"{kernel_name=}, not exists in loaded kernels"
        err = F.release_kernel(kernel_name)
        if err == 0:
            self.kernel_list.pop(kernel_name)
        assert err == 0
        return err

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
        """
        run the kernel

        :param kernel_name:     the name of the kernel
        :param input:
            Dictionary to set input args, in the same order as kernel function
                * **args from np.array should be contiguous array**
                * constant args:
                    * python type: float -> c type: float
                    * python type: int -> c type: long
                    * or specify c type with field "type", support types:
                        * [unsigned] int
                        * [unsigned] long
                        * float
                        * double
        :param output:              List of names to specify which array will be get back from GPU buffer
        :param local_work_size:  list of integer, specified work sizes. **local_work_size can be set to `[-1]`,
                                    then will pass `nullptr` to `clEnqueueNDRangeKernel`**
        :param global_work_size:
        :param wait:                Optional, default true, wait for GPU
        :param timer:
            Optional, arguments to set up a timer for benchmark kernels
              * warmup: recycle times before timing
              * repeat: repeat multiple times and get ***AVERAGE TIME*** of multiple times, the result is `elapsed time / repeat`
              * name: name of a global timer
        :return:
        """
        assert kernel_name in self.kernel_list
        return F.run(
            kernel_name=kernel_name,
            input=input,
            output=output,
            local_work_size=local_work_size,
            global_work_size=global_work_size,
            wait=wait,
            timer=timer,
        )
