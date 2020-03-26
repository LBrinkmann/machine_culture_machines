from setuptools import setup, find_packages


def load_requirements(filename='requirements.txt'):
    with open(filename) as f:
        lines = f.readlines()
    return lines


setup(name='mc-machine-backend',
      version="0.0.1",
      description='',
      url='',
      author='MPIB - Human and Machines',
      author_email='',
      license='',
      packages=[package for package in find_packages()
                if package.startswith('machine_backend')],
      zip_safe=False,
      install_requires=load_requirements(),
        scripts=[
            'scripts/start-machine-backend',
            'scripts/build-machine-backend',
            'scripts/deploy-machine-backend'
        ]
      )
