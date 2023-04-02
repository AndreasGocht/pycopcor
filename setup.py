import setuptools
import numpy

common_compile_args = [
    "-std=c++17",
    "-std=c++17",
    "-O3",
    "-fopenmp",
    "-fopenmp-simd",
    "-funroll-loops",
    "-g",
    "-ffast-math"]

common_link_args = ["-fopenmp"]
common_include_dirs = ["src/", "copcor/src/", "copcor/src/copula/", numpy.get_include()]
common_libraries = []

copula_entroy = setuptools.Extension("pycopcor.copula._entropy",
                                     sources=[
                                         "src/copula/entropy.cpp",
                                         "copcor/src/copula/entropy.c",
                                     ],
                                     include_dirs=common_include_dirs,
                                     extra_compile_args=common_compile_args,
                                     extra_link_args=common_link_args,
                                     libraries=common_libraries,
                                     )

copula_wolff = setuptools.Extension("pycopcor.copula._wolff",
                                    sources=[
                                        "src/copula/wolff.cpp",
                                        "copcor/src/copula/wolff.c",
                                    ],
                                    include_dirs=common_include_dirs,
                                    extra_compile_args=common_compile_args,
                                    extra_link_args=common_link_args,
                                    libraries=common_libraries,
                                    )

utils = setuptools.Extension("pycopcor._utils",
                             sources=[
                                 "src/utils/utils.cpp",
                                 "copcor/src/utils.c",
                             ],
                             include_dirs=common_include_dirs,
                             extra_compile_args=common_compile_args,
                             extra_link_args=common_link_args,
                             libraries=common_libraries,
                             )

setuptools.setup(name='pycopcor',
                 version='1.0.0',
                 description='A framework for non-linear dependency or correlation analysis using copulas',
                 packages=[
                     "pycopcor",
                     "pycopcor.copula",
                     "pycopcor.utils",
                 ],
                 package_dir={
                     "pycopcor": "src",
                 },
                 ext_modules=[
                     copula_entroy,
                     copula_wolff,
                     utils
                 ],
                 install_requires=["numpy", "scipy"],
                 author="Andreas Gocht-Zech",
                 url="https://github.com/AndreasGocht/pycopcor",
                 readme = "README.md",
                 classifiers = [
                    "License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)",
                    "Operating System :: POSIX",
                    "Operating System :: Unix",                    
                    "Programming Language :: Python :: 3.6",
                    "Programming Language :: Python :: 3.7",
                    "Programming Language :: Python :: 3.8",
                    "Programming Language :: Python :: 3.9",
                    "Programming Language :: Python :: 3.10",
                    "Programming Language :: Python :: Implementation :: CPython",
                    "Programming Language :: Python :: Implementation :: PyPy",
                    ]
                 )
