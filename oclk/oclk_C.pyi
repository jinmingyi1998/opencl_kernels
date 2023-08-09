from typing import Union, List, Dict
import numpy as np

def init() -> int: ...
def loak_kernel(
    cl_file: str, kernel_name: str, compile_option: Union[str, List[str]]
) -> int: ...
def release_kernel(kernel_name: str) -> int: ...
def run(
    *,
    kernel_name: str,
    input: List[Dict[str, Union[int, float, np.array]]],
    output: List[str],
    local_work_size: List[int],
    global_work_size: List[int],
    wait: bool,
    timer: Dict
) -> RunnerReturn: ...

class TimerResult:
    name: str
    cnt: str
    avg: str
    stdev: str
    total: str

class RunnerReturn:
    timer_result: TimerResult
    results: List[np.ndarray]
