#! /usr/bin/env python
""" Dynamic configuration of distribution package. """

import os

from setuptools import find_packages, setup

# get __version__ from _version.py
ver_file = os.path.join('cblearn', '_version.py')
with open(ver_file) as f:
    exec(f.read())

setup(version=__version__)