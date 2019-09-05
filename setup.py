import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="bayestuner",
    version="1.0.4",
    author="Yazid Janati El Idrissi",
    author_email="janati.yazid@gmail.com",
    description="A Bayesian Optimization Package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yazidjanati/bayestuner",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
