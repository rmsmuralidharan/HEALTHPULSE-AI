from setuptools import setup, find_packages
from typing import List

HYPHEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    """
    Reads requirements.txt and returns a list of dependencies.
    """

    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()

        requirements = [req.replace('\n', '') for req in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

            return requirements



setup(
    name = "HealthPulse-AI",
    version='0.0.1',
    author = 'Muralidharan RMS',
    author_email='rmsmuralidaran@gmail.com',
    description="ECG-based Myocardial Infarction detection using Machine Learning and Deep Learning",
     long_description=open('README.md', encoding='utf-8').read() if __name__ == '__main__' else '',
    long_description_content_type='text/markdown',
    url="https://github.com/rmsmuralidharan/HealthPulse-AI",
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    license="MIT"
)