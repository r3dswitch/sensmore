from setuptools import setup, find_packages

setup(
    name='Sensmore Case Study',
    version='0.0.1',
    packages=find_packages(),
    install_requires=[
    ],
    author='Soumya Mondal',
    author_email='soumya.mondal@tum.de',
    description='Small VLA POC for Autonomous Trucking',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/r3dswitch/sensmore',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
)
