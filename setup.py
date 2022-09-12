#!/usr/bin/env python

from setuptools import setup, find_packages


setup(
    name='Conditional BG',
    version='1.0',
    author='Sebastian Falkner',
    license='MIT',
    packages = find_packages(),
    install_requires=[
          'numba',
          'numpy',
          'scipy',
          'pyyaml',
          'matplotlib'
      ],
)
