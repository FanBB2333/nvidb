from setuptools import setup, find_packages

setup(
    name='nvidb',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        # 'numpy',
    ],
    entry_points={
        'console_scripts': [
            'nvidbtest=test.run:main',
        ],
    },
)
