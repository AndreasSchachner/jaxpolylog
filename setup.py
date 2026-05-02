from setuptools import setup
from pathlib import Path

setup(
    name='jaxpolylog',
    version='0.2.0',
    description='JAX-compatible polylogarithm functions with automatic differentiation.',
    long_description=Path('README.md').read_text(encoding='utf-8'),
    long_description_content_type='text/markdown',
    author='Andreas Schachner',
    author_email='as3475@cornell.edu',
    url='https://github.com/AndreasSchachner/jaxpolylog',
    packages=['jaxpolylog'],
    python_requires='>=3.12',
    install_requires=[
        'numpy',
        'jax',
        'jaxlib',
    ],
    license='GPL-3.0-or-later',
    classifiers=[
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.12',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering :: Mathematics',
    ],
)
