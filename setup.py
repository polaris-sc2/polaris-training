"""Groups the project overall info and requirements"""
from os.path import realpath, dirname, join
from setuptools import setup, find_packages


DISTNAME = "leelastar"
DESCRIPTION = "LeelaStar: Crowd-Sourced Deep Reinforcement Learning Agent for Starcraft 2"
AUTHOR = "LeelaStar Developers"
AUTHOR_EMAIL = "leelastar.devs@gmail.com"
URL = "https://github.com/leelastar/leelastar-training"
LICENSE = "Apache License, Version 2.0"
VERSION = "0.0.1"

PROJECT_ROOT = dirname(realpath(__file__))
REQUIREMENTS_FILE = join(PROJECT_ROOT, "requirements.txt")

with open(REQUIREMENTS_FILE) as f:
    REQUIREMENTS = f.read().splitlines()

if __name__ == "__main__":
    setup(
        name=DISTNAME,
        version=VERSION,
        description=DESCRIPTION,
        author=AUTHOR,
        author_email=AUTHOR_EMAIL,
        packages=find_packages(),
        install_requires=REQUIREMENTS,
        include_package_data=True,
        url=URL,
        license=LICENSE,
    )
