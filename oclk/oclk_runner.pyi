from typing import Dict, List, Union, Optional

import numpy as np

class TimerResult:
    name: str
    cnt: str
    avg: str
    stdev: str
    total: str

class RunnerReturn:
    timer_result: TimerResult
    results: List[np.ndarray]

class TimerArgs:
    enable: bool
    warmup: int
    repeat: int
    name: str

    def __init__(self, enable: bool, warmup: int, repeat: int, name: str): ...
    def __dict__(self): ...

TimerArgsDisabled = TimerArgs(False, 0, 0, "no_name")

class Runner:
    has_initialized: bool
    kernel_list: Dict[str, Dict[str, str]]

    def load_kernel(
        self,
        cl_file: str,
        kernel_name: Union[str, List[str]],
        compile_option: Optional[Union[str, List[str]]] = None,
    ): ...
    def release_kernel(self, kernel_name: Union[str, List[str]]) -> int: ...
    def run(
            self,
            *,
            kernel_name: str,
            input: List[
                Dict[
                    str,
                    Union[int, float, np.array, List[Dict[str, Union[int, float, str]]]],
                ]
            ],
            local_work_size: Union[List[int],Tuple[int]],
            global_work_size: Union[List[int],Tuple[int]],
            output: Optional[List[str]] = None,
            wait: Optional[bool] = True,
            timer: Optional[Union[Dict, TimerArgs]] = TimerArgsDisabled,
    ) -> RunnerReturn: ...

def RunnerCtx(filename, kernel_name, compile_option=""): ...
