from __future__ import absolute_import, print_function
from setuptools import setup, find_packages
from pathlib import Path

_dir = Path(__file__).parent.resolve()

with open(_dir/'augmend'/'version.py') as f:
    exec(f.read())

def read_readme():
    try:
        with open(_dir/'README.md') as f:
            return f.read()
    except FileNotFoundError:
        print("README.md not found, using default description")
        return "A default description of your package."



setup(
    name='augmend',
    version=__version__,
    author='Martin Weigert, Uwe Schmidt',
    description='Augmend',
    url='https://github.com/stardist/augmend/',
    license='BSD 3-Clause License',
    long_description=read_readme(),
    long_description_content_type='text/markdown',
    packages=find_packages(),
    package_data={"augmend":
                        ['transforms/kernels/*.cl', 'README.md'],

                    },
    python_requires='>=3.5'
)
