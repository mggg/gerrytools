
from setuptools import setup, find_packages

requirements = [
    "pandas", "scipy", "networkx", "geopandas", "shapely", "matplotlib",
    "gerrychain", "sortedcontainers", "gurobipy", "jsonlines", "opencv-python",
    "us"
]

setup(
    name="evaltools",
    author="MGGG Redistricting Lab",
    author_email="engineering@mggg.org",
    description="Tools for processing and visualizing districting plans.",
    url="https://github.com/mggg/plan-evaluation-processing",
    packages=find_packages(exclude=["tests"]),
    install_requires=requirements
)
