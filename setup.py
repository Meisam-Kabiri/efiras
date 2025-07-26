from setuptools import setup, find_packages

setup(
    name="efiras",
    packages=find_packages(where="src"),  # Look in src directory
    package_dir={"": "src"},              # Root package is in src
)