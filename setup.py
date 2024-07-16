from setuptools import find_packages, setup

description = """Common (Cooperative) Multi-agent Environments - A collection of multi-agent environments for reinforcement learning research. The environments support PettingZoo ParallelEnv APIs."""

setup(
    name="co_mas",
    version="1.0.0",
    description="Common (Cooperative) Multi-agent Environments.",
    long_description=description,
    license="MIT License",
    keywords="Multi-agent Environment, PettingZoo, Multi-Agent Reinforcement Learning",
    packages=find_packages(),
    install_requires=[
        "pettingzoo>=1.24.3",
        "gymnasium",
        "loguru",
        "pre-commit",
    ],
)
