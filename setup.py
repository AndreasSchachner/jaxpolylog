from setuptools import setup

setup(
    name='jaxpolylog',
    version='1.0.0',
    description='',
    author='Andreas Schachner',
    author_email='andreas.schachner@gmx.de',
    packages=['jaxpolylog'],
    python_requires='>=3.8',
    install_requires=[
        'numpy',
        'jax',
        'jaxlib',
        'partial'
    ],
)
