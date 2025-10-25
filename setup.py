from setuptools import setup, find_packages

setup(
    name="ecg_pipeline",
    version="0.1.0",
    description="High-Performance 12-Lead ECG Image to Time-Series Deep Learning Pipeline",
    author="Tahleel1611",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "opencv-python",
        "scipy",
        "scikit-image",
        "torch",
        "torchvision",
        "matplotlib",
        "seaborn",
        "tqdm"
    ],
    python_requires=">=3.8",
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)