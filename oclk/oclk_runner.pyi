from typing import Dict, List, Union, Optional

import numpy as np

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
        input: List[Dict[str, Union[int, float, np.array]]],
        output: List[str],
        local_work_size: List[int],
        global_work_size: List[int],
        wait: bool = True,
        timer: Union[Dict, TimerArgs] = TimerArgsDisabled,
    ) -> List[np.ndarray]: ...
