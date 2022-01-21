from Cython.Build import cythonize
from Cython.Distutils import build_ext
from Cython import Build
from setuptools import setup
from setuptools.extension import Extension
import numpy

'''
# For validation uncommment this section
setup(ext_modules=(cythonize([Extension("validation",
                                        sources=["validation.pyx"],
                                        include_dirs=[numpy.get_include()],
                                        language="c++",
                                        extra_compile_args=['-std=c++11', '-fopenmp', '-O3'],
                                        extra_link_args=['-fopenmp']
                                    )
                              ]
			            )
                  )
      )
'''

# For scenario uncommment this section
setup(ext_modules=(cythonize([Extension("scenario",
                                        sources=["scenario.pyx"],
                                        include_dirs=[numpy.get_include()],
                                        language="c++",
                                        extra_compile_args=['-std=c++11', '-fopenmp', '-O3'],
                                        extra_link_args=['-fopenmp']
                                    )
                              ]
			            )
                  )
      )
