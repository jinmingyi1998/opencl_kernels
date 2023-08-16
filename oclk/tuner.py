import contextlib
import functools
import itertools
from oclk.oclk_runner import Runner, TimerArgs


@contextlib.contextmanager
def _run(filename, kernel_name, compile_option=""):
    r = Runner()
    r.load_kernel(filename, kernel_name, compile_option)
    try:
        yield r
    finally:
        r.release_kernel(kernel_name)


def tune_():
    filename = ''
    kernel_name = ''
    compile_option = ''
    with _run(filename, kernel_name, compile_option) as r:
        ...


class TuneArgGenerator:
    def __init__(self, method, values):
        assert method in ['range', 'values']
        self.method = method  # range / values
        self.values = values
        self.candidates = []
        if self.method == 'range':
            self.candidates = list(range(*self.values))
        elif self.method == 'values':
            self.candidates = values

    def __iter__(self):
        for v in self.candidates:
            yield v



class Tunner:
    def __init__(self,*,name,**kwargs):
        for key,value in kwargs.items():
            self.__setattr__(key,value)
            self.metrics = []



    def range_arg(self,name,start,end,step=1):
        g = TuneArgGenerator('range',(start,end,step))
        def decorator(fn):
            @functools.wraps(fn)
            def innfer_wrapper(*args, **kwargs):
                kwargs[name] = g
                return fn(*args, **kwargs)
            return innfer_wrapper
        return decorator

    def values_arg(self,name,*args):
        g = TuneArgGenerator('values',*args)
        def decorator(fn):
            @functools.wraps(fn)
            def innfer_wrapper(*args, **kwargs):
                kwargs[name] = g
                return fn(*args, **kwargs)
            return innfer_wrapper
        return decorator

    def tune(self,fn):
        @functools.wraps(fn)
        def wrapper(*args,**kwargs):
            result = fn(*args,**kwargs)
            self.metrics.append(result)
        return wrapper

