from setuptools import setup, find_packages

with open('requirements.txt', encoding='utf-8') as file:
    requirements = file.read().splitlines()

with open('README.md', encoding='utf-8') as file:
    long_description = file.read()

setup(
    name='invertible_cl',
    version='0.0.1',
    description='Invertible CL.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/mishgon/invertible_cl',
    packages=find_packages(include=('invertible_cl',)),
    python_requires='>=3.6',
    install_requires=requirements,
)
