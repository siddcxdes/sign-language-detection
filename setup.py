from setuptools import setup, find_packages

setup(
    name="sign-language-detection",
    version="1.0.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "streamlit==1.24.0",
        "opencv-python-headless==4.7.0.72",
        "numpy==1.23.5",
        "mediapipe==0.10.3",
        "scikit-learn==1.2.2",
        "protobuf==3.20.3",
        "Pillow==9.5.0",
        "matplotlib==3.7.1",
        "python-dotenv==0.21.1",
        "setuptools==67.8.0",
        "wheel==0.40.0",
    ],
    python_requires=">=3.10,<3.11",
)