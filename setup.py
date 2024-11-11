from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="transformers_102",
    version="0.1",
    package_dir={"transformers_102": "src"},
    packages=find_packages(),
    install_requires=requirements,
)