
from setuptools import setup

requirements = [
    "pandas", "scipy", "networkx", "geopandas", "shapely", "matplotlib",
    "gerrychain"
]

setup(
    name="plan_evaluation",
    author="MGGG Redistricting Lab",
    author_email="contact@mggg.org",
    description="Evaluate and visualize districting plans.",
    url="https://github.com/mggg/plan-evaluation",
    packages=["evaltools"],
    install_requires=requirements
)
