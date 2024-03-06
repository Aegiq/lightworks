from setuptools import setup, find_packages

version = {}
with open("lightworks/__version.py") as f:
      exec(f.read(), version)

setup(name = "lightworks",
      version = version["__version__"],
      description = "SDK for Aegiq's photonic quantum computing platform.",
      license = "Apache 2.0",
      packages = find_packages(where=".", exclude = ["lightworks_tests"]),
      python_requires = ">=3.10",
      install_requires = ["thewalrus==0.20.0", "matplotlib>=3.7.1",
                          "pandas>=2.0.1", "numpy>=1.24.3", "ipykernel",
                          "bayesian-optimization>=1.4.3", "imageio>=2.31.0",
                          "networkx>=3.1.0", "drawsvg>=2.3.0", "zoopt>=0.4.2",
                          "Pyarrow"])
