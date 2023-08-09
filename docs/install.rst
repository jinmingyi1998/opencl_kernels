Install
*****************************************************************

Requirements
=================================================================

* OpenCL GPU hardware
* numpy
* cmake(if compile from source)

Install from wheel
=================================================================

.. code-block:: shell

    pip install pyoclk

or download wheel from `release <https://github.com/jinmingyi1998/opencl_kernels/releases>`_ and install

Compile from source
=================================================================
C
1. Clone this repo

.. code-block:: shell

    # http
    git clone --recursive https://github.com/jinmingyi1998/opencl_kernels.git

    # or with ssh
    git clone --recursive git@github.com:jinmingyi1998/opencl_kernels.git

2. Install

.. code-block:: shell

    cd opencl_kernels
    python setup.py install


**DO NOT move this directory after install**