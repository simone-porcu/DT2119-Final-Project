from setuptools import setup, find_packages

with open('README.md') as f:
    long_description = f.read()

setup(
    name='dualstudent-asr',
    version='0.1.2',
    long_description=long_description,
    long_description_content_type='text/markdown',
    description='Dual Student for Automatic Speech Recognition',
    author='Simone Porcu, Andrea Caraffa, Kevin Dalla Torre Castillo, Franco Ruggeri',
    author_email='porcu@kth.se, caraffa@kth.se, kevindt@kth.se, fruggeri@kth.se',
    license='GPL',
    packages=find_packages(include=['dualstudent', 'dualstudent.*']),
    include_package_data=True,
    url='https://github.com/simone-porcu/DT2119-Final-Project'
)
