from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="optionclass-api",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="API for predicting student academic tracks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/optionclass-api",
    packages=find_packages(exclude=["tests*"]),
    include_package_data=True,
    install_requires=[
        "fastapi>=0.68.0",
        "uvicorn>=0.15.0",
        "pydantic>=1.8.0",
        "python-dotenv>=0.19.0",
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "pycaret>=3.0.0",
        "scikit-learn>=1.0.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Education",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires='>=3.8',
    entry_points={
        'console_scripts': [
            'optionclass-api=run:main',
        ],
    },
)
