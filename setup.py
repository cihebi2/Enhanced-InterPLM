# 安装配置 
# setup.py

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README_en.md").read_text(encoding='utf-8')

# Read requirements
with open('requirements.txt') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="enhanced-interplm",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@institution.edu",
    description="Enhanced InterPLM: Advanced Interpretability for Protein Language Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/enhanced-interplm",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
            "pre-commit>=3.3.0",
        ],
        "docs": [
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=1.3.0",
            "sphinx-autodoc-typehints>=1.24.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "enhanced-interplm-train=enhanced_interplm.train_enhanced_sae:main",
            "enhanced-interplm-extract=enhanced_interplm.esm.extract_embeddings_cli:main",
            "enhanced-interplm-analyze=enhanced_interplm.analysis.analyze_features_cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "enhanced_interplm": [
            "configs/*.yaml",
            "configs/*.yml",
            "data/motif_library/*.json",
        ],
    },
)