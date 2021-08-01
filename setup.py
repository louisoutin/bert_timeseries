from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup_args = dict(
    name="tst",
    version=0.1,
    description="Time series transformers.",
    url="https://gitlab.com/jungleai",
    author="Jungle",
    author_email="junglers@jungle.ai",
    classifiers=[
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
    ],
    packages=find_packages(exclude=["contrib", "docs"]),
    install_requires=requirements,
)

if __name__ == "__main__":
    setup(**setup_args)
