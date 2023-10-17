import os

from setuptools import find_packages, setup


def read(fname):
    """Read file from the main directory of this package"""
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="llm_lib",
    version=read("../version.txt"),
    author="Sid Kiblawi, Sam Preston, Robert Tinn",
    author_email=(
        "sidkiblawi@microsoft.com,sam.preston@microsoft.com,robert.tinn@microsoft.com"
    ),
    description="LLM utilities for clinical NLP applications",
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*"]),
    install_requires=[
        "chromadb>0.3",
        "hydra-core>=1.3",
        "langchain==0.0.272",
        "openai>=0.27",
        "tiktoken>0.3",
        "pandas",
        "scikit-learn",
    ],
    long_description=read("README.md"),
    classifiers=["Development Status :: 3 - Alpha", "Topic :: Utilities"],
    python_requires=">=3.9",
)
