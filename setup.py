#! /usr/bin/env python
""" Dynamic configuration of distribution package. """
from setuptools import setup
import versioneer


setup(version=versioneer.get_version(), cmdclass=versioneer.get_cmdclass())