from typing import Dict, List, Union, Optional

import numpy as np

import oclk.functions as F

class Runner:
    def load_kernel(
        self,
        cl_file: str,
        kernel_name: Union[str, List[str]],
        compile_option: Optional[Union[str, List[str]]] = None,
    ): ...
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
    ) -> List[np.ndarray]: ...
