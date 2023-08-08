# run in docker quay.io/pypa/manylinux_2_28_x86_64:2023-08-07-e3f636d
.PHONY: all clean build_doc build_wheel

all: build_wheel build_doc

build_wheel:
	/opt/python/cp36-cp36m/bin/python3  setup.py build bdist_wheel --plat-name=manylinux_2_28_x86_64
	/opt/python/cp37-cp37m/bin/python3  setup.py build bdist_wheel --plat-name=manylinux_2_28_x86_64
	/opt/python/cp38-cp38/bin/python3   setup.py build bdist_wheel --plat-name=manylinux_2_28_x86_64
	/opt/python/cp39-cp39/bin/python3   setup.py build bdist_wheel --plat-name=manylinux_2_28_x86_64
	/opt/python/cp310-cp310/bin/python3 setup.py build bdist_wheel --plat-name=manylinux_2_28_x86_64
	/opt/python/cp311-cp311/bin/python3 setup.py build bdist_wheel --plat-name=manylinux_2_28_x86_64
	/opt/python/cp312-cp312/bin/python3 setup.py build bdist_wheel --plat-name=manylinux_2_28_x86_64

build_doc:
	cd docs; \
	sphinx-apidoc -M --implicit-namespaces -o src ../oclk ../oclk/functions.py ../oclk/version.py  ../oclk/__init__.py ../oclk/third_party ; \
	sphinx-build -b html . build


clean:
	rm -rf build/
	rm -rf dist/
	rm -rf pyoclk.egg-info/
	rm -rf docs/build