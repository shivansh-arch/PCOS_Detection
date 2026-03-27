from setuptools import setup, find_packages,setup
from typing import List

def get_requirements(requirements_path:str)->List[str]:
    with open(requirements_path) as requirements_file:
        return requirements_file.readlines().remove('-e .')

setup(
    name='PCOS_Detection',
    version='0.1',
    author='Shivansh Gupta',
    author_email="Shivanshg36@gmial.com",
    description='A machine learning model to detect PCOS based on various health parameters.',
    packages=find_packages(),
    install_requires=get_requirements(requirements_path='requirements.txt'),
)