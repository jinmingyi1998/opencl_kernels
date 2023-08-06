# run in docker quay.io/pypa/manylinux_2_28_x86_64:latest
.PHONY: all clean

all:
	/opt/python/cp36-cp36m/bin/python3  setup.py build bdist_wheel --plat-name=manylinux1_x86_64
	/opt/python/cp37-cp37m/bin/python3  setup.py build bdist_wheel --plat-name=manylinux1_x86_64
	/opt/python/cp38-cp38/bin/python3   setup.py build bdist_wheel --plat-name=manylinux1_x86_64
	/opt/python/cp39-cp39/bin/python3   setup.py build bdist_wheel --plat-name=manylinux1_x86_64
	/opt/python/cp310-cp310/bin/python3 setup.py build bdist_wheel --plat-name=manylinux1_x86_64
	/opt/python/cp311-cp311/bin/python3 setup.py build bdist_wheel --plat-name=manylinux1_x86_64
	/opt/python/cp312-cp312/bin/python3 setup.py build bdist_wheel --plat-name=manylinux1_x86_64

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf pyoclk.egg-info/