
from setuptools import setup, find_packages

requirements = [
    "pandas", "scipy", "networkx", "geopandas", "shapely", "matplotlib",
    "gerrychain", "sortedcontainers", "gurobipy", "jsonlines", "opencv-python-headless",
    "imageio", "us", "pydantic", "censusdata", "seaborn", "maup"
]

setup(
    name="evaltools",
    author="MGGG Redistricting Lab",
    author_email="engineering@mggg.org",
    description="Tools for processing and visualizing districting plans.",
    url="https://github.com/mggg/evaltools",
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
