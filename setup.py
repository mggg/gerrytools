
from setuptools import setup

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
    packages=["evaltools"],
    install_requires=requirements
)
