from setuptools import setup, find_packages
from codecs import open
from os import path

HERE = path.abspath(path.dirname(__file__))
with open(path.join(HERE, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="EazyML",
    version="0.1.1",
    description="Simple ML library",
    long_description="A python library for building, training, and testing simple machine learning models.",
    long_description_content_type="text/markdown",
    url="https://eazyml.readthedocs.io/",
    author="Joshua Geddes",
    author_email="joshuageddes333@gmail.com",
    license="MIT",
    classifiers=[
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent"
    ],
    packages=["eazyml"],
    include_package_data=True,
    install_requires=["numpy"]
)