from setuptools import setup, find_packages

setup(
    name='dualstudent-asr',
    version='0.1.0',
    description='Dual Student training for Automatic Speech Recognition',
    author='Andrea Caraffa, Kevin Dalla Torre Castillo, Simone Porcu, Franco Ruggeri',
    author_email='caraffa@kth.se, kevindt@kth.se, porcu@kth.se, fruggeri@kth.se',
    license='GPL',
    packages=find_packages(include=['dualstudent', 'dualstudent.*']),
    include_package_data=True
)
