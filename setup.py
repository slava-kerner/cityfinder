#!/usr/bin/env python3
import os.path
from setuptools import setup, find_packages


# https://packaging.python.org/guides/single-sourcing-package-version/
version = {}
with open(os.path.join("cityfinder", "version.py")) as fp:
    exec(fp.read(), version)


setup(
    name='city finder',
    version=version["__version__"],
    author='Slava Kerner',
    author_email='slava.kerner@gmail.com',
    description='',
    long_description=open('README.md').read(),
    packages=find_packages(exclude=["tests"]),
    license="GPLv3",
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Utilities',
    ],
    install_requires=[
        'numpy',
        'pillow'
    ],
    extras_require={
        "dev": [
            "hypothesis",
        ]
    }
)
