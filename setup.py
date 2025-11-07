from setuptools import setup, find_packages

setup(
    name="sign-language-detection",
    version="1.0.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "opencv-python-headless==4.8.0.74",
        "numpy==1.24.3",
        "scikit-learn==1.3.0",
        "mediapipe==0.8.11",
        "streamlit>=1.24.0,<2.0.0",
        "Pillow>=9.5.0,<10.0.0",
        "protobuf>=3.20.0,<4.0.0",
        "six>=1.16.0",
        "python-dotenv>=0.19.0",
    ],
    python_requires=">=3.9,<3.10",
)