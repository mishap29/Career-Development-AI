from setuptools import setup, find_packages

setup(
    name="career_development_ai",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "streamlit==1.32.0",
        "openai==1.12.0",
        "python-dotenv==1.0.0",
        "pandas==2.2.0",
        "numpy==1.26.3",
        "Pillow==10.2.0",
    ],
) 