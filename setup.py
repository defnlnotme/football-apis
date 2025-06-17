from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="football-apis",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A Python package for accessing various football data APIs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/football-apis",
    packages=find_packages(),
    package_dir={"": "."},
    install_requires=[
        "requests>=2.25.0",
        "python-dotenv>=0.15.0",
        "pandas>=1.2.0",
        "beautifulsoup4>=4.9.0",
        "lxml>=4.6.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
