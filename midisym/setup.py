from setuptools import setup, find_packages, Extension
import sys
import numpy
import os

# Get the absolute path of the current directory (where setup.py is located)
module_path = os.path.abspath(os.path.dirname(__file__))
print(f"Module path: {module_path}")
# Define the extension modules with absolute paths
ext_modules = [
    Extension(
        name="midisym.mymodule",  # Module name
        sources=[os.path.join(module_path, "src/mymodule.c")],  # Absolute path to C source file
        include_dirs=[numpy.get_include()],  # Include NumPy headers if needed
        extra_compile_args=[],  # Additional compile options
    ),
    Extension(
        name="midisym.mymodule2",  # C++ module
        sources=[os.path.join(module_path, "src/mymodule2.cpp")],  # Absolute path to C++ source file
        include_dirs=[numpy.get_include()],  # Include NumPy headers
        language="c++",  # Specify the language
        extra_compile_args=["-std=c++11"],  # Additional compiler arguments, e.g., C++11 standard
    ),
    Extension(
        name="midisym.csamplers",  # Another C module
        sources=[os.path.join(module_path, "src/gmsamplersmodule.c")],  # Absolute path to C source file
        extra_link_args=[],  # Additional linker options
        include_dirs=[numpy.get_include(), os.path.join(module_path, "include")],  # Include directories
    ),
]

# Setup function
setup(
    name="midisym",
    version="0.1.0",
    description="A custom simple MIDI reader based on mido",
    author="Eunjin Choi",
    author_email="jech@kaist.ac.kr",
    url="https://github.com/jech2/midisym",
    packages=find_packages(),  # Automatically find all packages
    install_requires=[
        "mido",  # Dependencies
        "symusic",
    ],
    ext_modules=ext_modules,  # Extension modules
)
