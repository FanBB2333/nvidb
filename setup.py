from setuptools import setup, find_packages
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()
setup(
    name='nvidb',
    version='1.0.0',
    packages=find_packages(),
    url="https://github.com/FanBB2333/nvidb/",
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=[
        'pandas',
        'paramiko',
        'pytest',
        'PyYAML',
        'setuptools',
        'termcolor',
        'nvidia_ml_py',
        'blessed'
    ],
    entry_points={
        'console_scripts': [
            'nvidb=nvidb.test.run:main',
        ],
    },
)
