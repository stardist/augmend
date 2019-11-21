from __future__ import absolute_import, print_function
from setuptools import setup, find_packages
from os import path

_dir = path.abspath(path.dirname(__file__))

with open(path.join(_dir,'augmend','version.py')) as f:
    exec(f.read())

setup(
    name='augmend',
    version=__version__,
    description='Augmend',
    license='BSD 3-Clause License',
    packages=find_packages(),
    package_data={"augmend":
                        ['transforms/kernels/*.cl'
                        ],

                    },
    python_requires='>=3.5'
)
