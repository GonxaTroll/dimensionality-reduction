"""Setup script for dimensionality-reduction package."""

from setuptools import setup, find_packages

setup(
    name="dimensionality-reduction",
    version="0.1.0",
    packages=find_packages(exclude=["tests*", "examples*"]),
    install_requires=[
        "numpy>=1.20.0",
        "matplotlib>=3.3.0",
    ],
)
