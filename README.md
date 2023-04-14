# Opencl Kernels

This repository contains opencl kernels, also necessary OpenCL context Management.

## Produce:
a library

## Tests:

use [googletest](https://github.com/google/googletest)

entry: under tests/

TODO: implement test framework

## Kernels:

- [Binary Op](kernel/binary_op.cl)
  - add
  - sub
  - multiply
  - divide
  - min
  - max
  - pow

## Issues:
### gflags glogs gtests share library not found
it may occur when using system installed libraries but not default library location.
set LD_LIBRARY_PATH to library location.