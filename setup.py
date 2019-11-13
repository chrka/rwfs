from setuptools import setup

from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="rwfs",
    version="0.0.1",
    description="Feature selection using simulated annealing",
    url="https://github.com/chrka/rwfs",
    author="Christoffer Karlsson",
    author_email="chrka@mac.com",
    license="MIT",
    packages=['rwfs'],
    install_requires=['numpy', 'scikit-learn'],
    zip_safe=False,
    long_description=long_description,
    long_description_content_type='text/markdown'
)