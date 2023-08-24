import abc
import functools
import itertools
from collections import defaultdict
from collections.abc import Iterable
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from tqdm import tqdm

import oclk.functions as F
from oclk.oclk_runner import RunnerCtx, Runner, RunnerReturn, TimerArgs


class TuneArgGenerator:
    def __init__(self, method, values):
        assert method in ["range", "values"]
        self.method = method  # range / values
        self.values = values
        self.candidates = []
        if self.method == "range":
            self.candidates = list(range(*self.values))
        elif self.method == "values":
            self.candidates = list(values)

    def __iter__(self):
        for v in self.candidates:
            yield v

    def __str__(self):
        return "TuneArgGenerator " + str(self.candidates)


class Tuner:
    tuner_registry = defaultdict(list)

    def __init__(self, name="", **kwargs):
        r = Runner()
        if not name:
            name = type(self).__name__
        self.name = name
        self.metrics = []
        self.timer = TimerArgs(True, 5, 100, name)
        for key, value in kwargs.items():
            self.__setattr__(key, value)

    @abc.abstractmethod
    def setup(self):
        """
        abstract method, will be called before tunable methods,
         used to initialize variables
        """
        ...

    def run(
        self,
        kernel_file: str,
        kernel_name: str,
        compile_option: str,
        *,
        input: List[Dict[str, Union[int, float, np.array]]],
        local_work_size: List[int],
        global_work_size: List[int],
        output: Optional[List[str]] = None,
        timer: Optional[Union[Dict, TimerArgs]] = None,
    ) -> RunnerReturn:
        """
        Wrapper for `Runner.run <#oclk.oclk_runner.Runner.run>`_
        In this method, Runner will load a kernel and run kernel, finally release kernel

        :param kernel_file: filename can be absolute or relative path
        :param kernel_name: kernel_name is the kernel functions' name
        :param compile_option: compile option can be strings like "-DMY_DEF=1", **"-D" is necessary**
        :param input: see `Runner.run`
        :param local_work_size: see `Runner.run`
        :param global_work_size: see `Runner.run`
        :param output: see `Runner.run`
        :param timer: see `Runner.run`
        :return: see `Runner.run`
        """
        if timer is None:
            timer = self.timer
        with RunnerCtx(kernel_file, kernel_name, compile_option) as r:
            F.clear_timer()
            return r.run(
                kernel_name=kernel_name,
                input=input,
                local_work_size=local_work_size,
                global_work_size=global_work_size,
                output=output,
                wait=True,
                timer=timer,
            )

    @staticmethod
    def range_arg(name, start, end, step=1):
        """
        decorator to generate ranged arguments,
        `start`,`end``,`step` are the same as `range()`

        :param name: argument name
        :param start: range start
        :param end: range end
        :param step: range step
        """
        g = TuneArgGenerator("range", (start, end, step))

        def decorator(fn):
            @functools.wraps(fn)
            def innfer_wrapper(self, *args, **kwargs):
                kwargs[name] = g
                return fn(self, *args, **kwargs)

            return innfer_wrapper

        return decorator

    @staticmethod
    def exp2_range(start, end):
        """
        generate exp2 values, from start(inclusive) to end(inclusive).

        :param start:
        :param end:
        :return:
        """
        a = start
        while a <= end:
            yield a
            a *= 2

    @staticmethod
    def worksize_arg(
        name,
        dim_size: int,
        dim0: List[int],
        dim1: Optional[List[int]] = None,
        dim2: Optional[List[int]] = None,
    ):
        """
        decorator to generate `worksize` arguments

        :param name: argument name
        :param dim_size: work dim size
        :param dim0: possible values for dim0
        :param dim1: possible values for dim1
        :param dim2: possible values for dim2
        """
        assert 0 < dim_size <= 3
        dims = []
        for i in range(dim_size):
            assert (
                locals()[f"dim{i}"] is not None
            ), f"dim_size is {dim_size}, but dim{i} is None"
            dims.append(locals()[f"dim{i}"])
        worksizes: List[List[int]] = [list(t) for t in itertools.product(*dims)]
        g = TuneArgGenerator("values", worksizes)

        def decorator(fn):
            @functools.wraps(fn)
            def innfer_wrapper(self, *args, **kwargs):
                kwargs[name] = g
                return fn(self, *args, **kwargs)

            return innfer_wrapper

        return decorator

    @staticmethod
    def values_arg(name, *args):
        """
        decorator, add a `values` argument generator

        :param name: the name of the argument
        :param args: all possible values
        """
        g = TuneArgGenerator("values", args)

        def decorator(fn):
            @functools.wraps(fn)
            def innfer_wrapper(self, *args, **kwargs):
                kwargs[name] = g
                return fn(self, *args, **kwargs)

            return innfer_wrapper

        return decorator

    @staticmethod
    def tune():
        """
        decorator, mark a tunable method.

        **NOTE:** method should be passed kwargs only, must return a value as the metric.
            all the returned values will be sorted by ASC order to pick the best.
            for instance, this value can be `rtn.timer_result.avg`

        :raise TuningSkip: raise `TuningSkip` to skip an argument combination
        """

        def decorator(fn):
            _module_name = fn.__module__
            _class_name = fn.__qualname__.split(".")[0]
            _method_name = fn.__name__
            Tuner.tuner_registry[(_module_name, _class_name)].append(_method_name)

            @functools.wraps(fn)
            def wrapper(self, *args, **kwargs):
                if args:
                    raise ValueError(
                        "Tune method takes no arguments, only accept kwargs"
                    )
                self.setup()
                args_set = []
                key_set = []

                for k, v in kwargs.items():
                    key_set.append(k)
                    if not issubclass(type(v), Iterable):
                        args_set.append([v])
                    else:
                        args_set.append(v)

                for fnargs in tqdm(list(itertools.product(*args_set))):
                    fnargs = dict(zip(key_set, fnargs))
                    try:
                        result = fn(self, **fnargs)
                        self.metrics.append((fnargs, result))
                    except TuningSkip as e:
                        ...

            return wrapper

        return decorator

    def top_result(self, k=5) -> List[Tuple[Dict[str, Any], float]]:
        """
        Get the top k results by ASC order

        :param k: top k
        :type k: int
        :return: top k results, for instance:

                 .. code-block::

                        [
                            (
                                {
                                    key:value,
                                    key2:value2,
                                    key3:value3,
                                },
                                1.23
                            ),
                            (
                                {
                                    key:value,
                                    key2:value2,
                                    key3:value3,
                                },
                                4.56
                            ),
                            (
                                {
                                    key:value,
                                    key2:value2,
                                    key3:value3,
                                },
                                7.89
                            )
                        ]

        :rtype: List[Tuple[Dict[str, Any], float]]
        """
        self.metrics.sort(key=lambda x: x[1])

        return self.metrics[:k]


class TuningSkip(BaseException):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
