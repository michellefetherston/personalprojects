from setuptools import setup, find_packages

setup(
    name="transaction-eda",
    version="0.1.0",
    description="EDA and spending pattern analysis with fuzzy merchant grouping",
    author="Michelle Fetherston",
    packages=find_packages(),
    
    # Core dependencies
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.23.0",
        "matplotlib>=3.6.0",
        "seaborn>=0.12.0",
        "rapidfuzz>=3.0.0",
        
        # Optional but recommended for scaling / clustering
        "scikit-learn>=1.2.0"
    ],
    
    python_requires=">=3.9",
    
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)