from setuptools import setup, find_packages

setup(
    name='nvidb',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'paramiko',
        'pytest',
        'PyYAML',
        'setuptools',
        'termcolor',
        'nvidia_ml_py',
    ],
    entry_points={
        'console_scripts': [
            'nvidb=nvidb.test.run:main',
        ],
    },
)
