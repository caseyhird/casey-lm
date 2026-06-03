from setuptools import find_packages, setup

setup(
    name="casey-lm",
    version="1.0",
    package_dir={"": "casey-lm"},
    packages=find_packages(where="casey-lm"),
    python_requires=">=3.10",
)