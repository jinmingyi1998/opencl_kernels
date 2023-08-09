# run in docker quay.io/pypa/manylinux_2_28_x86_64:2023-08-07-e3f636d
.PHONY: all clean build_doc build_wheel

all: build_wheel build_doc

build_wheel:
	rm -rf build/
	for PY in `ls /opt/python | grep cp` ;                                                           \
	do                                                                                               \
		/opt/python/$${PY}/bin/python3 setup.py build bdist_wheel --plat-name=$${AUDITWHEEL_PLAT} ;  \
	done

build_doc:
	cd docs; \
	sphinx-apidoc -M --implicit-namespaces -o src ../oclk ../oclk/functions.py ../oclk/version.py  ../oclk/__init__.py ../oclk/third_party ; \
	sphinx-build -b html . build


clean:
	rm -rf build/
	rm -rf dist/
	rm -rf pyoclk.egg-info/
	rm -rf docs/build