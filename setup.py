#   Setup script
# 
# 
from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="agiformer",
    version="0.1.0",
    description="AGIFORMER: Towards Artificial General Intelligence - A Revolutionary Transformer Architecture",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="AGIFORMER Team",
    author_email="",
    url="https://github.com/yourusername/agiformer",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "torchaudio>=2.0.0",
        "numpy>=1.24.0",
        "einops>=0.7.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "transformers>=4.30.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
        "training": [
            "wandb>=0.15.0",
            "accelerate>=0.20.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: Other/Proprietary License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
