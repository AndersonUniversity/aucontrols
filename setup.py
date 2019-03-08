from setuptools import setup, find_packages

setup(
    name='aucontrols',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'control',
        'scipy',
        'matplotlib'
    ],
    license='Creative Commons Attribution-Noncommercial-Share Alike license',
    long_description=open('README.md').read(),
)
