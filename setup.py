import os

from setuptools import find_packages, setup


def read(fname):
    """Read file from the main directory of this package"""
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="rwd_llm",
    version="0.0.1",
    author="Sid Kiblawi, Sam Preston, Robert Tinn",
    author_email="sidkiblawi@microsoft.com,sam.preston@microsoft.com",
    description="LLM utilities for clinical NLP applications",
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*"]),
    install_requires=[
        "hydra-core>=1.3",
        "chromadb==0.6.3",
        "langchain==0.3.18",
        "openai>=1.0",
        "tiktoken>0.3",
        "haikunator>=2.1",
        "mlflow>=2.6",
        "pandas",
        "scikit-learn",
    ],
    long_description=read("README.md"),
    classifiers=["Development Status :: 3 - Alpha", "Topic :: Utilities"],
    python_requires=">=3.9",
)
