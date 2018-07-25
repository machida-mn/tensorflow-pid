#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup
from setuptools import find_packages

setup(name='tensorflow-pid',
      version='1.0.0',
      description='Tensorflow implementation of PID optimization',
      url='https://github.com/machida-mn/tensorflow-pid',
      author='Ryohei Machida',
      author_email='machida_mn@complex.ist.hokudai.ac.jp',
      license='MIT',
      packages=find_packages(),
      install_requires=[
          'tensorflow'
      ])
