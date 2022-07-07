
from setuptools import setup, find_packages
from pathlib import Path

requirements = [
    "pandas", "scipy", "networkx", "geopandas", "shapely", "matplotlib",
    "gerrychain", "sortedcontainers", "gurobipy", "jsonlines", "opencv-python-headless",
    "imageio", "us", "pydantic", "censusdata", "seaborn", "maup"
]

# Set the version --- ensure that the latest tag matches this value.
VERSION = "1.0.2"

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
            "pytest-cov"
        ]
    }
)
