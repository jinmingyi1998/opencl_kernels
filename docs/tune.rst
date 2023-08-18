Kernel Tuning
**************************************************

How To
====================================================
1. Given a OpenCL kernel file
2. Write a python file
    A. write a class, extend `Tuner <src/oclk.html#oclk.tuner.Tuner>`_
        a. write a method in the class, this method should only accept kwargs
        b. add decorators to generate arguments for this method

            :`worksize_arg <src/oclk.html#oclk.tuner.Tuner.worksize_arg>`_: can easily generate multi-dim work_size args
            :`values_arg <src/oclk.html#oclk.tuner.Tuner.values_arg>`_:   can enumerate some specific values
            :`range_arg <src/oclk.html#oclk.tuner.Tuner.range_arg>`_:    can generate ranged arguments like `range`

        c. add a decorator `Tuner.tune() <src/oclk.html#oclk.tuner.Tuner.tune>`_ to mark it as a tunable method

                **NOTE:** decorator :code:`tune()` should be the last decorator, all possible arguments are unpacked in :code:`tune()` and passed to your method

        d. implement this method to run once with a specific arguments combination,
            call `self.run() <src/oclk.html#oclk.tuner.Tuner.run>`_ to run a kernel.

    B. write a :code:`setup()` method *if needed*

           :code:`setup()` is an abstract method, will be called before tuning, you can initialize some variabels with `self`
3. execute and tune, two ways for you:
    A. execute in cli
        a. go to the directory where contains the tuner class above
        b. execute :code:`python -m oclk tune`
        c. then will execute all tuner class under the directory and output results in a json file
        d. see more options with :code:`python -m oclk tune --help`

    B. execute in python.
        a. Instantiate this class and call the above method
        b. Got the top :code:`k` best result and arguments combination with `tuner.top_result() <src/oclk.html#oclk.tuner.Tuner.top_result>`_

Example
==========================================================================

see `tune examples <https://github.com/jinmingyi1998/opencl_kernels/tree/master/examples/tune>`_