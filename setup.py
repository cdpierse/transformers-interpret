from distutils.core import setup

from setuptools import find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


setup(
    name="transformers-interpret",
    packages=find_packages(
        exclude=[
            "*.tests",
            "*.tests.*",
            "tests.*",
            "tests",
            "examples",
            "docs",
            "out",
            "dist",
            "media",
            "test",
        ]
    ),
    version="0.5.2",
    license="Apache-2.0",
    description="Transformers Interpret is a model explainability tool designed to work exclusively with 🤗 transformers.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Charles Pierse",
    author_email="charlespierse@gmail.com",
    url="https://github.com/cdpierse/transformers-interpret",
    keywords=[
        "machine learning",
        "natural language proessing",
        "explainability",
        "transformers",
        "model interpretability",
    ],
    install_requires=["transformers>=3.0.0", "captum>=0.3.1"],
    classifiers=[
        "Development Status :: 3 - Alpha",  # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Programming Language :: Python :: 3.8",
    ],
)
