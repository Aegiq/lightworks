from setuptools import find_packages, setup

version = {}
with open("lightworks/__version.py") as f:
    exec(f.read(), version)

setup(
    name="lightworks",
    author="Aegiq Ltd.",
    version=version["__version__"],
    description="Open-source Python SDK for photonic quantum computation.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Aegiq/lightworks",
    project_urls={
        "Source": "https://github.com/Aegiq/lightworks",
        "Documentation": "https://aegiq.github.io/lightworks/",
    },
    license="Apache 2.0",
    packages=find_packages(where=".", exclude=["tests"]),
    package_data={"lightworks": ["py.typed"]},
    python_requires=">=3.10",
    install_requires=[
        "thewalrus==0.20.0",
        "matplotlib>=3.7.1",
        "pandas>=2.0.1",
        "numpy>=1.24.3",
        "multimethod>=1.11.2",
        "drawsvg>=2.3.0",
        "CairoSVG>=2.8.0",
        "sympy>=1.12.0",
        "pyarrow",
        "ipython",
    ],
    extras_require={"qiskit": ["qiskit[visualization]>=1.1.0"]},
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Operating System :: OS Independent",
    ],
)
