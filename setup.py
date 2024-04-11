from pathlib import Path

from setuptools import find_packages, setup

requirements = [
    "pandas<2.0.0,>=1.5.0",
    "scipy",
    "networkx",
    "geopandas",
    "Shapely>=2.0.0",
    "matplotlib",
    "gerrychain",
    "sortedcontainers",
    "jsonlines",
    "opencv-python-headless",
    "imageio",
    "us",
    "pydantic",
    "censusdata",
    "seaborn",
    "maup<=1.1.0",
]

# Set the version --- ensure that the latest tag matches this value.
VERSION = "1.2.0"

# Description.
here = Path(__file__).parent
DESCRIPTION = (here / "README.md").read_text()

setup(
    name="gerrytools",
    version=VERSION,
    author="MGGG Redistricting Lab",
    author_email="engineering@mggg.org",
    description="Tools for processing and visualizing districting plans.",
    long_description=DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/mggg/gerrytools",
    packages=find_packages(exclude=["tests", "tutorials"]),
    install_requires=requirements,
    include_package_data=True,
    extras_require={
        "dev": [
            "pdoc3",
            "flake8",
            "pytest",
            "autopep8",
            "pytest-cov",
            "black",
            "isort",
        ],
        "mgrp": ["docker>=7.0.0"],
    },
)
