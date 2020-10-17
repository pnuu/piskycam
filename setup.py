#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2020
#
# Author(s):
#
#   Panu Lahtinen <pnuu+git@iki.fi>
#
# This is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# this program.  If not, see <http://www.gnu.org/licenses/>.

from setuptools import setup

from piskycam import __version__

requirements = ['numpy', 'zarr', 'ephem', 'picamera', 'pyyaml']

setup(name="piskycam",
      version=__version__,
      description='Sky camera software written in Python for Raspberry Pi',
      author='Panu Lahtinen',
      author_email='pnuu+git@iki.fi',
      url="http://github.com/pnuu/piskycam",
      packages=['piskycam'],
      scripts=['bin/picam.py',
               'bin/zarr2img.py'],
      license="GPLv3",
      install_requires=requirements,
      classifiers=[
          'Development Status :: 2 - Pre-Alpha',
          'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
          'Programming Language :: Python',
          'Operating System :: Raspbian',
          'Intended Audience :: Science/Research',
          'Topic :: Scientific/Engineering',
          'Topic :: Communications'
      ],
      python_requires='>=3.7',
      )
