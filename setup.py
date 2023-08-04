import os
import subprocess
import sys
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

__version__ = "0.0.0"
exec(open("oclk/version.py").read())


class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str = "") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = os.fspath(Path(sourcedir).resolve())


class CMakeBuild(build_ext):
    def build_extension(self, ext: CMakeExtension) -> None:
        # Must be in this form due to bug in .resolve() only fixed in Python 3.10+
        ext_fullpath = Path.cwd() / self.get_ext_fullpath(ext.name)
        extdir = ext_fullpath.parent.resolve()

        # Using this requires trailing slash for auto-detection & inclusion of
        # auxiliary "native" libs

        debug = int(os.environ.get("DEBUG", 0)) if self.debug is None else self.debug
        cfg = "Debug" if debug else "Release"

        # CMake lets you override the generator - we need to check this.
        # Can be set with Conda-Build, for example.
        cmake_generator = os.environ.get("CMAKE_GENERATOR", "")
        cmake_args = []
        if self.compiler.compiler_type != "msvc":
            # Using Ninja-build since it a) is available as a wheel and b)
            # multithreads automatically. MSVC would require all variables be
            # exported for Ninja to pick it up, which is a little tricky to do.
            # Users can override the generator with CMAKE_GENERATOR in CMake
            # 3.15+.
            if not cmake_generator or cmake_generator == "Ninja":
                try:
                    import ninja

                    ninja_executable_path = Path(ninja.BIN_DIR) / "ninja"
                    cmake_args += [
                        "-GNinja",
                        f"-DCMAKE_MAKE_PROGRAM:FILEPATH={ninja_executable_path}",
                    ]
                except ImportError:
                    print("ninja not found")

        else:
            print("only support Linux", file=sys.stderr)
            exit(1)

        if sys.platform.startswith("darwin"):
            print("only support Linux", file=sys.stderr)
            exit(1)

        cmake_args.append(f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}{os.sep}oclk{os.sep}")
        cmake_args.append(f"-DPython_ROOT_DIR={os.path.dirname(sys.executable)}")
        cmake_args.append(f"-DCMAKE_BUILD_TYPE={cfg}")

        build_args = []
        # Adding CMake arguments set as environment variable
        # (needed e.g. to build for ARM OSx on conda-forge)
        if "CMAKE_ARGS" in os.environ:
            cmake_args += [item for item in os.environ["CMAKE_ARGS"].split(" ") if item]

        # In this example, we pass in the version to C++. You might not need to.
        cmake_args += [f"-DOCLK_VERSION_INFO={__version__}"]

        # Set CMAKE_BUILD_PARALLEL_LEVEL to control the parallel build level
        # across all generators.
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            # self.parallel is a Python 3 only way to set parallel jobs by hand
            # using -j in the build_ext call, not supported by pip or PyPA-build.
            if hasattr(self, "parallel") and self.parallel:
                # CMake 3.12+ only.
                build_args += [f"-j{self.parallel}"]

        build_temp = Path(self.build_temp) / ext.name
        if not build_temp.exists():
            build_temp.mkdir(parents=True)

        print("cmake args:", " ".join(cmake_args))
        subprocess.run(
            ["cmake", ext.sourcedir, *cmake_args], cwd=build_temp, check=True
        )
        print("build_args:", " ".join(build_args))
        subprocess.run(
            ["cmake", "--build", ".", *build_args], cwd=build_temp, check=True
        )


# The information here can also be placed in setup.cfg - better separation of
# logic and declaration, and simpler if you include description/version in a file.
with open("README.md", "r") as f:
    long_desc = f.read()
setup(
    name="pyoclk",
    version=__version__,
    author="Mingyi Jin",
    author_email="jinmingyi1998@sina.cn",
    description="A easy way to run OpenCL kernel files",
    long_description=long_desc,
    packages=["oclk", "oclk.third_party"],
    include_package_data=True,
    ext_modules=[CMakeExtension("oclk_C")],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
    install_requires=["numpy"],
    python_requires=">=3.6",
)
