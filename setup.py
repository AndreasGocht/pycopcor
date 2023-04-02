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
common_include_dirs = ["src/", "copcor/src/", numpy.get_include()]
common_libraries = []

copula_entroy = setuptools.Extension("pycopcor.copula._entropy",
                                     sources=[
                                         "src/copula/entropy.cpp",
                                         "copcor/src/copula/entropy.c"
                                     ],
                                     include_dirs=common_include_dirs,
                                     extra_compile_args=common_compile_args,
                                     extra_link_args=common_link_args,
                                     libraries=common_libraries
                                     )

copula_wolff = setuptools.Extension("pycopcor.copula._wolff",
                                    sources=[
                                        "src/copula/wolff.cpp",
                                        "copcor/src/copula/wolff.c"
                                    ],
                                    include_dirs=common_include_dirs,
                                    extra_compile_args=common_compile_args,
                                    extra_link_args=common_link_args,
                                    libraries=common_libraries)

utils = setuptools.Extension("pycopcor._utils",
                             sources=[
                                 "src/utils/utils.cpp",
                                 "copcor/src/utils.c",
                             ],
                             include_dirs=common_include_dirs,
                             extra_compile_args=common_compile_args,
                             extra_link_args=common_link_args,
                             libraries=common_libraries
                             )

setuptools.setup(name='pycopcor',
                 version='1.0.0',
                 description='Copula based Corrleation Measures',
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
                 )
