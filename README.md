# OpenCL Kernel Python Wrapper

## Install

### Requirements

* OpenCL GPU hardware
* numpy
* cmake(if compile from source)

### Install from wheel

```shell
pip install pyoclk
```

or download wheel from [release](https://github.com/jinmingyi1998/opencl_kernels/releases) and install

### Compile from source

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
    input=[
        {"name": "a", "value": a, },
        {"name": "out", "value": out, },
        {"name": "int_arg", "value": 1, },
        {"name": "float_arg", "value": 12.34}
    ],
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
    input=[
        {"name": "a", "value": a, },
        {"name": "out", "value": out, },
        {"name": "int_arg", "value": 1, },
        {"name": "float_arg", "value": 12.34}
    ],
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
    * **args from np.array should be contiguous array**
    * constant args:
        * python type: float -> c type: float
        * python type: int -> c type: long
        * or specify c type with field "type", support types:
            * [unsigned] int
            * [unsigned] long
            * float
            * double
* output: List of names to specify which array will be get back from GPU buffer
* local_work_size/global_work_work: list of integer, specified work sizes. **local_work_size can be set to `[-1]`, then
  will pass `nullptr` to `clEnqueueNDRangeKernel`**
* wait: Optional, default true, wait for GPU
* timer: Optional, arguments to set up a timer for benchmark kernels
  * warmup: recycle times before timing
  * repeat: repeat multiple times and get ***AVERAGE TIME*** of multiple times, the result is `elapsed time / repeat`
  * name: name of a global timer

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
    input=[
        {"name": "a", "value": a, },
        {"name": "b", "value": b, },
        {"name": "int_arg", "value": 1, "type": "int"},
        {"name": "float_arg", "value": 12.34},
        {"name": "c", "value": c}
    ],
    output=['c'],
    local_work_size=[1, 1, 1],
    global_work_size=a.shape,
    timer=timer
    )
```
