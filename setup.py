from setuptools import setup, find_packages

setup(
    name='ehr-classification-with-bert',
    version='0.1.0',
    packages=find_packages(),
    url='https://github.com/jernejvivod/ehr-classification-with-bert',
    license='MIT',
    author='Jernej Vivod',
    author_email='vivod.jernej@gmail.com',
    description='EHR data classification with BERT',
    entry_points={
        'console_scripts': [
            'ehr-classification-with-bert=ehr_classification_with_bert.__main__:main',
        ]
    },
)
