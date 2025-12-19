from setuptools import setup, find_packages
from pathlib import Path
import re

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read VERSION from config.py without importing (to avoid dependency issues during install)
config_file = this_directory / "nvidb" / "config.py"
version_match = re.search(r'^VERSION\s*=\s*["\']([^"\']+)["\']', config_file.read_text(), re.MULTILINE)
VERSION = version_match.group(1) if version_match else "0.0.0"

setup(
    name='nvidb',
    version=VERSION,
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
