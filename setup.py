"""
Setup script for pandere-forge package.
This file is kept for backwards compatibility with older pip versions.
Configuration is primarily in pyproject.toml.
"""

from setuptools import setup

if __name__ == "__main__":
    setup(
        install_requires=[
            "pandas",
            "httpx",
            "ollama",
            "pandera",
            "numpy",
            "pydantic",
            "anthropic",
            "openai",
        ]
    )
