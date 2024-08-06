from setuptools import setup, find_packages
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()
setup(
    name='nvidb',
    version='0.1',
    packages=find_packages(),
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
    ],
    entry_points={
        'console_scripts': [
            'nvidb=nvidb.test.run:main',
        ],
    },
)
