# run in docker quay.io/pypa/manylinux_2_28_x86_64:latest
.PHONY: all  build_all clean cp36 cp37 cp38 cp39 cp310 cp311 cp312

all: build_all
	/opt/python/cp36-cp36m/bin/python3 setup.py build bdist_wheel
	/opt/python/cp37-cp37m/bin/python3 setup.py build bdist_wheel
	/opt/python/cp38-cp38/bin/python3 setup.py build bdist_wheel
	/opt/python/cp39-cp39/bin/python3 setup.py build bdist_wheel
	/opt/python/cp310-cp310/bin/python3 setup.py build bdist_wheel
	/opt/python/cp311-cp311/bin/python3 setup.py build bdist_wheel
	/opt/python/cp312-cp312/bin/python3 setup.py build bdist_wheel

build_all: cp36 cp37 cp38 cp39 cp310 cp311 cp312


cp36:
	/opt/python/cp36-cp36m/bin/python3 setup.py build

cp37:
	/opt/python/cp37-cp37m/bin/python3 setup.py build

cp38:
	/opt/python/cp38-cp38/bin/python3 setup.py build

cp39:
	/opt/python/cp39-cp39/bin/python3 setup.py build

cp310:
	/opt/python/cp310-cp310/bin/python3 setup.py build

cp311:
	/opt/python/cp311-cp311/bin/python3 setup.py build

cp312:
	/opt/python/cp312-cp312/bin/python3 setup.py build

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf pyoclk.egg-info/