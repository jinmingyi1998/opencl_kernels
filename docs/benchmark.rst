Kernel Benchmark
***************************************************

How To
======================================================

1. write a config like `bench_add.yaml <https://github.com/jinmingyi1998/opencl_kernels/blob/master/examples/bench_add.yaml>`_
2. run :code:`python -m oclk.benchmark bench_add.yaml`

Configuration Reference
======================================================

A configuration is a list of `Suite`_

Suite
--------------------------------------------------

:`suite_name`\: str: the name of the suite
:`kernel_file`\: str: file path to the kernel file
:`kernels`\: List[`Kernel`_]: kernels in the file
:`timer`\: `Timer`_: define the timer

Kernel
------------------------------------------------

:`name`\: str: the name of the kernel
:`definition`\: str: program compile definition, with "-D"
:`local_work_size`\: List[int]: local work size of kernel
:`global_work_size`\: List[int]: global work size of kernel
:`args`\: List[`KernelArg`_]: list of arguments, in the order

KernelArg
--------------------------------------------------

:`name`\: str: name of the argument
:`value`\: `ArgValueGenerator`_: define how to generate arg value, default is constant zeros
:`type`\: str: arg type, can be :code:`array`, or ctypes: :code:`float`, :code:`int` ... see `Runner <src/oclk.html#oclk.oclk_runner.Runner.run>`_
:`dtype`\: str: :code:`np.dtype` for array, required when type is array, default is :code:`float32`, can be :code:`int32` :code:`int64` :code:`float32` :code:`float64` ...
:`shape`\: List[int]: array shape, required when type is array, default is :code:`[1]`

ArgValueGenerator
-------------------------------------------------

:`method`\: str: generate method, can be :code:`random` or :code:`constant`, default :code:`constant`
:`value`\: Union[int, float, List[Union[int, float]]]: constant value, or random value range. default :code:`0`

Timer
------------------------------------------------

:`prefix`\: str: timer name prefix, defalut ""
:`repeat`\: int: repeat times in timer's ONE call, defalut 1
:`warmup`\: int: warm up times before timing, defalut 0

Example
=======================================================
.. code-block:: yaml

    - suite_name: add
      kernel_file: examples/add.cl
      kernels:
        - name: add
          definition: ""
          args:
            - name: a
              type: array
              dtype: float32
              shape:
                - 64
                - 64
                - 64
              value:
                method: random
            - name: b
              type: array
              dtype: float32
              shape: [64, 64, 64]
              value:
                type: constant
                value: 0
            - name: length
              type: int
              value:
                type: constant
                value: 262144
            - name: out
              type: array
              dtype: float32
              shape: [64, 64, 64]
              # value field default: constant zero
          local_work_size: [1]
          global_work_size: [262144]
        - name: add_constant
          definition: ""
          local_work_size: [ 1 ]
          global_work_size: [ 262144 ]
          args:
            - name: a
              type: array
              dtype: float32
              shape:
                - 64
                - 64
                - 64
              value:
                method: random
            - name: x
              type: float
              shape: [ 64, 64, 64 ]
              value:
                type: constant
                value: 0
            - name: length
              type: long
              value:
                type: constant
                value: 262144
            - name: out
              type: array
              dtype: float32
              shape: [ 64, 64, 64 ]
        - name: add_batch
          definition: "-DBATCH_SIZE=4"
          local_work_size: [ 1 ]
          global_work_size: [ 65536 ]
          args:
            - name: a
              type: array
              dtype: float32
              shape:
                - 64
                - 64
                - 64
              value:
                method: random # constant, random
            - name: b
              type: array
              dtype: float32
              shape: [64, 64, 64]
              value:
                type: constant
                value: 0
            - name: length
              type: long
              value:
                type: constant
                value: 262144
            - name: out
              type: array
              dtype: float32
              shape: [64, 64, 64]
      timer:
        prefix: "bench_add"
        repeat: 5
        warmup: 5
