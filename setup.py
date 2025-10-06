from setuptools import setup, find_packages


def readme():
    with open("README.md") as f:
        return f.read()


setup(
    name="oven",
    version="1.0.0",
    description="Oven modeling for OC",
    long_description=readme(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent",
    ],
    url="https://github.com/siddharth-prabhu/OC_OvenModeling",
    author="Siddharth Prabhu",
    author_email="siddharth.prabhu15@gmail.com",
    keywords="demo project",
    license="MIT",
    packages= find_packages(), # ["oven"],
    install_requires=[],
    include_package_data=True,
)