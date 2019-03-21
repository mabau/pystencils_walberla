import os
import sys
from setuptools import setup, find_packages
sys.path.insert(0, os.path.abspath('..'))
from custom_pypi_index.pypi_index import get_current_dev_version_from_git


setup(name='pystencils_walberla',
      version=get_current_dev_version_from_git(),
      description='pystencils code generation for waLBerla apps',
      author='Martin Bauer',
      license='AGPLv3',
      author_email='martin.bauer@fau.de',
      url='https://i10git.cs.fau.de/software/pystencils/',
      packages=['pystencils_walberla'],
      install_requires=['pystencils[alltrafos]', 'jinja2'],
      package_data={'pystencils_walberla': ['templates/*']},
      )
