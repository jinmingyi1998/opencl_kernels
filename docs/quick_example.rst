A quick example
**************************************************************

Example calculate a+b
==============================================================

Kernel File:

a file named :code:`add.cl`

.. code-block:: c
    :caption: add.cl

    kernel void add(global float*a, global float*out, int int_arg, float float_arg){
        int x = get_global_id(0);
        if(x==0){
            printf(" accept int arg: %d, accept float arg: %f\n",int_arg,float_arg);
        }
        out[x] = a[x] * float_arg + int_arg;
    }

Python Code in OOP Style

.. code-block:: python3
    :emphasize-lines: 18-30

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
            {"name": "int_arg", "value": 1, "type": "int"},
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

or just call with Functions

.. code-block:: python3
    :emphasize-lines: 11-22    

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

see more examples on `github <https://github.com/jinmingyi1998/opencl_kernels/tree/master/examples>`_