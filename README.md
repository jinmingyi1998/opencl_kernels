# OpenCL Kernel Python Wrapper

## Install
### Install from wheel

_prebuild wheel has no glog, output might be ugly_

download wheel from [release](https://github.com/jinmingyi1998/opencl_kernels/releases) and install

### Compile from source

If you have glog installed, it is recommended to compile this package from source

**Clone this repo**

clone by http

```shell
git clone --recursive https://github.com/jinmingyi1998/opencl_kernels.git
```

with ssh

```shell
git clone --recursive git@github.com:jinmingyi1998/opencl_kernels.git
```

**Install**

```shell
cd opencl_kernels
python setup.py install
```
***DO NOT move this directory after install***
### Requirements

* OpenCL GPU hardware
* numpy
* cmake > 3.16

## Usage

### Kernel File:

a file named `add.cl`

```c
kernel void add(global float*a, global float*out, int int_arg, float float_arg){
    int x = get_global_id(0);
    if(x==0){
        printf(" accept int arg: %d, accept float arg: %f\n",int_arg,float_arg);
    }
    out[x] = a[x] * float_arg + int_arg;    
}
```

### Python Code

#### OOP Style

```python
import numpy as np
import oclk

a = np.random.rand(100, 100).reshape([10, -1])
a = np.float32(a)
out = np.zeros(a.shape)
out = np.float32(out)

runner = oclk.Runner()
runner.load_kernel("add.cl", "add", "")

timer = oclk.TimerArgs(
    enable=True,
    warmup=10,
    repeat=50,
    name='add_kernel'
)
runner.run(
    kernel_name="add",
    input={"a": a, "out": out, "int_arg": 1, "float_arg": 12.34},
    output=['out'],
    local_work_size=[1, 1],
    global_work_size=a.shape,
    timer=timer
)
# check result
a = a.reshape([-1])
out = out.reshape([-1])
print(a[:8])
print(out[:8])
```

#### Call with Functions

```python
import numpy as np
import oclk

a = np.random.rand(100, 100).reshape([10, -1])
a = np.float32(a)
out = np.zeros(a.shape)
out = np.float32(out)
oclk.init()
oclk.load_kernel("add.cl", "add", "")
r = oclk.run(
    kernel_name="add",
    input={"a": a, "out": out, "int_arg": 1, "float_arg": 12.34},
    output=['out'],
    local_work_size=[1, 1],
    global_work_size=a.shape
)
# check result
a = a.reshape([-1])
out = out.reshape([-1])
print(a[:8])
print(out[:8])
```

### Python api Usage

#### API

```python
def run(*, kernel_name: str,
        input: Dict[str, Union[int, float, np.array]],
        output: List[str],
        local_work_size: List[int],
        global_work_size: List[int],
        wait: bool = True,
        timer: Union[Dict, TimerArgs] = TimerArgsDisabled) -> List[np.ndarray]: ...
```

* input: Dictionary to set input args, in the same order as kernel function
* output: List of names to specify which array will be get back from GPU buffer
* local_work_size/global_work_work: list of integer, specified work sizes
* wait: Optional, default true, wait for GPU
* timer: Optional, arguments to set up a timer for benchmark kernels

#### example

```python
a = np.zeros([16, 16, 16], dtype=np.float32)
b = np.zeros([16, 16, 16], dtype=np.float32)
c = np.zeros([16, 16, 16], dtype=np.float32)
timer = TimerArgs(enable=True,
                  warmup=10,
                  repeat=100,
                  name='timer_name'
                  )
run(kernel_name='add',
    input={
        'a': a,
        'b': b,
        'int_arg': 1,
        'float_arg': 23.45,
        'c': c,
    },
    output=['c'],
    local_work_size=[1, 1, 1],
    global_work_size=a.shape,
    timer=timer
    )
```

## Known Issues

This package need to build multiple libs so far. Then this package cannot be installed by wheel or something else (Because
pip will build in a tmp env and move libs to site-packages, which will cause share libs link to removed temp share libs
and then link file not found). So clone to a directory and install this package with `python setup.py install` and don't move this directory.
