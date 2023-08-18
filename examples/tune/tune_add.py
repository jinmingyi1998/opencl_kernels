import os
from pprint import pp

import numpy as np
from oclk import input_maker
from oclk.tuner import Tuner, TuningSkip


class AddTuner(Tuner):
    def setup(self):
        shape = [16, 1024, 1024]  # 64MB
        dtype = np.float32
        a = np.random.random(shape)
        b = np.random.random(shape)

        self.a = np.ascontiguousarray(a, dtype)
        self.b = np.ascontiguousarray(b, dtype)
        self.gt = np.ascontiguousarray(a + b, dtype)
        self.dtype = dtype
        self.shape = shape

    @Tuner.worksize_arg("local_work_size", 1, list(Tuner.exp2_range(1, 1024)))
    @Tuner.values_arg("vector_size", 4, 8, 16)
    @Tuner.range_arg("tile_size", 4, 17, 4)
    @Tuner.values_arg("method", "naive", "vectorized", "tile")
    @Tuner.tune()
    def add(self, *, local_work_size, vector_size, tile_size, method):
        # raise TuningSkip skip duplicate args
        if method == "naive" and (vector_size != 4 or tile_size != 4):
            raise TuningSkip()
        if method == "tile" and vector_size != 4:
            raise TuningSkip()
        if method == "vectorized" and tile_size != 4:
            raise TuningSkip()

        # determine args for run()
        kernel_name = "add"
        global_size = self.a.size
        if method == "vectorized":
            kernel_name = "add_vector"
            global_size //= vector_size
        elif method == "tile":
            kernel_name = "add_tile"
            global_size //= tile_size

        c = np.zeros_like(self.a)
        c = np.ascontiguousarray(c, self.dtype)

        rtn = self.run(
            kernel_file=os.path.join(os.path.dirname(__file__), "add.cl"),
            kernel_name=kernel_name,
            compile_option=f"-DVECTOR_SIZE={vector_size} -DTILE_SIZE={tile_size}",
            input=input_maker(a=self.a, b=self.b, c=c, length=(self.a.size, "int")),
            output=["c"],  # [Optional] used to check results
            local_work_size=local_work_size,
            global_work_size=[global_size],
            timer=self.timer,
        )

        # [Optional] check whether the result is correct
        diff = np.abs(c - self.gt).max()
        assert (
            diff < 1e-6
        ), f"diff not zero : {diff=} {method=} {local_work_size=} {vector_size=} {tile_size=}"

        # return the timer's time
        return rtn.timer_result.avg


# execute and tune, two ways for you:
#     1. execute in python
#     2. execute in cli
# code below is executing in python
if __name__ == "__main__":
    # instantiate
    tuner = AddTuner()

    # tune
    tuner.add()

    # print top K=5 arguments combinations
    pp(tuner.top_result())
