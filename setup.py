"""Groups the project overall info and requirements"""

from os.path import dirname, join, realpath
from setuptools import find_packages, setup


DISTNAME = "Polaris"
DESCRIPTION = "Polaris: Crowd-Sourced Deep Reinforcement Learning Agent for Starcraft 2"
AUTHOR = "Polaris Developers"
AUTHOR_EMAIL = "polaris.devs@gmail.com"
URL = "https://github.com/polaris-sc2/polaris-training"
LICENSE = "Apache License, Version 2.0"
VERSION = "0.0.1"

PROJECT_ROOT = dirname(realpath(__file__))
REQUIREMENTS_FILE = join(PROJECT_ROOT, "requirements.txt")

with open(REQUIREMENTS_FILE) as f:
    LINES = f.read().splitlines()
    DEPENDENCY_LINKS = [l for l in LINES if l.startswith("git+")]
    REQUIREMENTS = [l for l in LINES if not l.startswith("git+")]

if __name__ == "__main__":
    setup(
        name=DISTNAME,
        version=VERSION,
        description=DESCRIPTION,
        author=AUTHOR,
        author_email=AUTHOR_EMAIL,
        packages=find_packages(),
        install_requires=REQUIREMENTS,
        dependency_links=DEPENDENCY_LINKS,
        include_package_data=True,
        url=URL,
        license=LICENSE,
    )
