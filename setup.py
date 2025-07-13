"""
TalkToEBM installation script.
"""

import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read the version from the version file
version = {}
with open("t2ebm/version.py") as fp:
    exec(fp.read(), version)

setuptools.setup(
    name="t2ebm",
    version=version["__version__"],
    author="Sebastian Bordt, Ben Lengerich, Harsha Nori, Rich Caruana",
    author_email="sbordt@posteo.de",
    description="A Natural Language Interface to Explainable Boosting Machines",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/interpretml/TalkToEBM",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "."},
    packages=["t2ebm"],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "matplotlib>=3.3.0",
        "tiktoken>=0.4.0",
        "openai>=1.8.0",
        "scipy>=1.7.0",
        "interpret>=0.2.0",
        "requests>=2.25.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=3.0",
            "black>=22.0",
            "isort>=5.13.2,<6.0",
            "flake8>=4.0",
            "mypy>=1.0",
            "pre-commit>=2.0",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
            "nbsphinx>=0.8",
            "pandoc>=2.0",
        ],
    },
)
