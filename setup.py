from os import path

from setuptools import find_packages, setup

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


def read_requirements():
    requirements = []
    with open('requirements.txt') as file:
        requirements = file.read().splitlines()
    with open('dev-requirements.txt') as file:
        requirements.extend(file.read().splitlines())
    print("Requirements: ", requirements)
    return requirements


setup(
    name='secretflow',
    version='0.6.13.beta1',
    license='Apache 2.0',
    description='Secret Flow',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='SCI Center',
    author_email='secretflow-contact@service.alipay.com',
    url='https://github.com/secretflow/secretflow',
    packages=find_packages(exclude=('examples', 'examples.*', 'tests', 'tests.*')),
    install_requires=read_requirements(),
    extras_require={
        'dev': ['pylint'],
    },
)
