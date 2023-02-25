from setuptools import setup, find_packages

setup(
    name='text-classification-with-embeddings',
    version='0.1.0',
    packages=find_packages(),
    url='https://github.com/jernejvivod/classification-with-embeddings',
    license='MIT',
    author='Jernej Vivod',
    author_email='vivod.jernej@gmail.com',
    description='Evaluation of entity embedding methods on classification tasks',
    entry_points={
        'console_scripts': [
            'classification_with_embeddings=classification_with_embeddings.__main__:main',
        ]
    },
)
