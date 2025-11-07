from setuptools import setup, find_packages

setup(
    name="sign-language-detection",
    version="1.0.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "opencv-python-headless==4.8.0.74",
        "numpy==1.25.2",
        "scikit-learn==1.3.2",
        "mediapipe==0.8.11",
        "streamlit==1.28.1",
        "Pillow==10.1.0",
        "protobuf==4.24.4",
        "six==1.16.0",
        "python-dotenv==1.0.0",
        "matplotlib==3.8.1",
        "setuptools>=68.0.0",
        "wheel>=0.41.0",
    ],
    python_requires=">=3.11,<3.12",
)