from setuptools import setup, find_packages

setup(
    name="sign-language-detection",
    version="1.0.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "streamlit==1.22.0",
        "opencv-python-headless==4.6.0.66",
        "numpy==1.21.6",
        "mediapipe==0.9.0.1",
        "scikit-learn==1.0.2",
        "protobuf==3.20.3",
        "Pillow==9.3.0",
        "matplotlib==3.5.3",
        "python-dotenv==0.21.0",
    ],
    python_requires=">=3.9,<3.10",
)