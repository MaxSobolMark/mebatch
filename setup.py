#!/usr/bin/env python
"""A setuptools based setup module.

See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / "README.md").read_text(encoding="utf-8")

# Arguments marked as "Required" below must be included for upload to PyPI.
# Fields marked as "Optional" may be commented out.

setup(
    # This is the name of your project. The first time you publish this
    # package, this name will be registered for you. It will determine how
    # users can install this project, e.g.:
    #
    # $ pip install sampleproject
    #
    # And where it will live on PyPI: https://pypi.org/project/sampleproject/
    #
    # There are some restrictions on what makes a valid project name
    # specification here:
    # https://packaging.python.org/specifications/core-metadata/#name
    name="mebatch",  # Required
    version="1.0.0",  # Required
    install_requires=[
        "Click",
        "readchar",
        "fabric",
        "paramiko",
        "slack_sdk",
        "filelock",
        "gcs_mutex_lock @ git+https://github.com/thinkingmachines/gcs-mutex-lock.git#egg=gcs_mutex_lock",
    ],
    dependency_links=[
        "git+https://github.com/thinkingmachines/gcs-mutex-lock.git",
    ],
    py_modules=["mebatch"],
    packages=find_packages(),  # Required
    # Specify which Python versions you support. In contrast to the
    # 'Programming Language' classifiers above, 'pip install' will check this
    # and refuse to install the project if the version does not match. See
    # https://packaging.python.org/guides/distributing-packages-using-setuptools/#python-requires
    python_requires=">=3.7, <4",
    entry_points={"console_scripts": ["mebatch = mebatch.mebatch:mebatch"]},
)
