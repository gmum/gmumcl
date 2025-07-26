from setuptools import setup, find_packages

setup(
    name="gmumcl",
    version="0.1.0",
    description="GMUM Continual Learning Framework",
    author="Jan Miksa, Patryk Krukowski",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "hydra-core>=1.3.2",
        "omegaconf>=2.3.0",
        "torch>=2.5.1",
        "torchvision>=0.19.1",
        "lightning>=2.5.0",
        "lightning-fabric>=2.5.0",
        "continuum>=1.2.7",
        "python-dotenv>=1.0.0",
    ],
    python_requires=">=3.8",
    license="MIT",
)
