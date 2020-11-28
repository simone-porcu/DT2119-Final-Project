from setuptools import setup, find_packages

with open('README.md') as f:
    long_description = f.read()

setup(
    name='dualstudent-asr',
    version='0.1.2',
    long_description=long_description,
    long_description_content_type='text/markdown',
    description='Dual Student for Automatic Speech Recognition',
    author='Franco Ruggeri, Andrea Caraffa, Kevin Dalla Torre Castillo, Simone Porcu',
    author_email='fruggeri@kth.se, caraffa@kth.se, kevindt@kth.se, porcu@kth.se',
    license='GPL',
    packages=find_packages(include=['dualstudent', 'dualstudent.*']),
    include_package_data=True,
    url='https://github.com/franco-ruggeri/dt2119-dualstudent-asr'
)
