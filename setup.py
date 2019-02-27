#!/usr/bin/env python

import os
from setuptools import setup, find_packages

REQUIREMENTS = [line.strip() for line in
                open(os.path.join("requirements.txt")).readlines()]

setup(name='headpose',
      version='0.1.0',
      url='https://github.com/brionmario/pnp-head-pose-estimation-poc',
      description='Head pose estimation using opencv and dlib',
      author='Brion Silva',
      author_email='brion@apareciumlabs.com',
      packages=find_packages(exclude=('tests', 'docs')),
      package_data={'headpose': ['Readme.md']},
      install_requires=REQUIREMENTS,
      include_package_data=True,
      license="The MIT License (MIT)"
      )
