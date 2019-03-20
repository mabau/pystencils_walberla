from setuptools import setup

setup(name='pystencils_walberla',
      description='pystencils code generation for waLBerla apps',
      author='Martin Bauer',
      license='AGPLv3',
      author_email='martin.bauer@fau.de',
      url='https://i10git.cs.fau.de/software/pystencils/',
      packages=['pystencils_walberla'],
      install_requires=['pystencils[alltrafos]', 'jinja2'],
      package_data={'pystencils_walberla': ['templates/*']},
      version_format='{tag}.dev{commits}+{sha}',
      )
