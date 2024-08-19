from setuptools import setup, find_packages


setup(
    name='dll',
    version='0.1.0',
    packages = find_packages(),
    install_requires=[],
    author = 'alex',
    author_email='xwan0454@student.monash.edu',
    description='This is a deep learning library',
    long_description=open('README.md').read(),
    long_description_content_type='text\markdown',
    url = 'https://github.com/mmy360/dll',
        classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)