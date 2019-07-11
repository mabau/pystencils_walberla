from setuptools import setup
import subprocess


def version_number_from_git(tag_prefix='release/', sha_length=10, version_format="{version}.dev{commits}+{sha}"):
    def get_released_versions():
        tags = sorted(subprocess.check_output(['git', 'tag'], encoding='utf-8').split('\n'))
        versions = [t[len(tag_prefix):] for t in tags if t.startswith(tag_prefix)]
        return versions

    def tag_from_version(v):
        return tag_prefix + v

    def increment_version(v):
        parsed_version = [int(i) for i in v.split('.')]
        parsed_version[-1] += 1
        return '.'.join(str(i) for i in parsed_version)

    try:
        latest_release = get_released_versions()[-1]
    except subprocess.CalledProcessError:
        return open('RELEASE-VERSION', 'r').read()

    commits_since_tag = subprocess.getoutput('git rev-list {}..HEAD --count'.format(tag_from_version(latest_release)))
    sha = subprocess.getoutput('git rev-parse HEAD')[:sha_length]
    is_dirty = len(subprocess.getoutput("git status --untracked-files=no -s")) > 0

    if int(commits_since_tag) == 0:
        version_string = latest_release
    else:
        next_version = increment_version(latest_release)
        version_string = version_format.format(version=next_version, commits=commits_since_tag, sha=sha)

    if is_dirty:
        version_string += ".dirty"

    with open("RELEASE-VERSION", "w") as f:
        f.write(version_string)

    return version_string


setup(name='pystencils_walberla',
      version=version_number_from_git(),
      description='pystencils code generation for waLBerla apps',
      author='Martin Bauer',
      license='AGPLv3',
      author_email='martin.bauer@fau.de',
      url='https://i10git.cs.fau.de/pycodegen/pystencils_walberla',
      packages=['pystencils_walberla'],
      install_requires=['pystencils[alltrafos]', 'jinja2'],
      package_data={'pystencils_walberla': ['templates/*']},
      test_suite='pystencils_walberla_tests',
      project_urls={
          "Bug Tracker": "https://i10git.cs.fau.de/pycodegen/pystencils_walberla/issues",
          "Source Code": "https://i10git.cs.fau.de/pycodegen/pystencils_walberla",
      },
      )
