
from setuptools import setup, find_packages

requirements = [
    "pandas", "scipy", "networkx", "geopandas", "shapely", "matplotlib",
    "gerrychain"
]

setup(
    name="evaltools",
    author="MGGG Redistricting Lab",
    author_email="engineering@mggg.org",
    description="Tools for evaluating and visualizing districting plans.",
    url="https://github.com/mggg/plan-evaluation-tools",
    packages=find_packages(exclude=["tests"]),
    install_requires=requirements
)
