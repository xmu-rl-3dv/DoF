from setuptools import setup

setup(
    name="DoF",
    description="A Diffusion Factorization Framework for Offline Multi-Agent Decision Making.",
    packages=["diffuser"],
    package_dir={
        "diffuser": "./diffuser",
    },
)
