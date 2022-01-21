from Cython.Build import cythonize
from Cython.Distutils import build_ext
from Cython import Build
from setuptools import setup
from setuptools.extension import Extension
import numpy


#experiments, fit_plots, plots_INS
setup(ext_modules=(cythonize([Extension("experiments_S2",
                                        sources=["experiments_S2.pyx"],
                                        include_dirs=[numpy.get_include()],
                                        language="c++",
                                        extra_compile_args=['-std=c++11', '-fopenmp', '-O3'],
                                        extra_link_args=['-fopenmp']
                                    )
                              ]
			            )
                  )
      )

