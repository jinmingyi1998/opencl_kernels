from typing import Dict, List, Optional, Union

from pydantic import BaseModel, field_validator


class ArgValueGenerator(BaseModel):
    method: str = "constant"
    value: Union[int, float, List[Union[int, float]]] = 0

    @classmethod
    def generate_check(cls, v: str):
        if v not in ["constant", "random"]:
            raise ValueError(f"generate field must be constant or random, got {v}")
        return v


arg_support_type = [
    "array",
    "int",
    "unsigned int",
    "long",
    "unsigned long",
    "float",
    "double",
]


class KernelArg(BaseModel):
    name: str = ""
    type: str  # see arg_support_type
    dtype: str = "float32"  # used for numpy array when type is array
    shape: Optional[List] = [1]
    value: ArgValueGenerator = ArgValueGenerator()

    @field_validator("type")
    @classmethod
    def type_check(cls, v: str):
        if v not in arg_support_type:
            raise ValueError(f"type must be array or int or float, got {v}")
        return v


def dict_to_KernelArg(d: Dict) -> KernelArg:
    if "value" in d:
        d["value"] = ArgValueGenerator(**d["value"])
    return KernelArg(**d)


class Kernel(BaseModel):
    name: str
    suffix: str = ""
    definition: str = ""
    local_work_size: List[int]
    global_work_size: List[int]
    args: List[KernelArg]


def dict_to_Kernel(d: Dict) -> KernelArg:
    if "args" in d:
        d["args"] = [dict_to_KernelArg(v) for v in d["args"]]
    return Kernel(**d)


class Timer(BaseModel):
    prefix: str = ""
    repeat: int = 1
    warmup: int = 0


class Suite(BaseModel):
    suite_name: str
    kernel_file: str
    kernels: List[Kernel]
    timer: Timer = Timer()


def dict_to_Suite(d: Dict) -> Suite:
    d["kernels"] = [dict_to_Kernel(k) for k in d["kernels"]]
    if "timer" in d:
        d["timer"] = Timer(**d["timer"])
    return Suite(**d)
