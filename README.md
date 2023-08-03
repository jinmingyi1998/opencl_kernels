# OpenCL Kernel Python Wrapper
## Usage
```python
import numpy as np
import oclk

a = np.random.rand(100, 100).reshape([-1])
a.dtype = np.float32
oclk.init()
oclk.load_kernel("print_arr.cl", "print_kernel", "")
out = oclk.run_impl_float(kernel_name="print_kernel",
input={"arr": a, "integer": 123, "float": 123.312},
output={"arr_out": a},
local_work_size=[1, 1],
global_work_size=[6, 6]
)

out = out[0]
print(a[:16])
print(out[:16])
print(out.size, out.shape)
print(id(a), id(out))
```