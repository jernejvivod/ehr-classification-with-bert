from setuptools import setup

setup(
    name='text-classification-with-embeddings',
    version='0.1.0',
    packages=['classification_with_embeddings', 'classification_with_embeddings.util', 'classification_with_embeddings.embedding', 'classification_with_embeddings.evaluation'],
    url='https://github.com/jernejvivod/classification-with-embeddings',
    license='MIT',
    author='Jernej Vivod',
    author_email='vivod.jernej@gmail.com',
    description='Evaluation of entity embedding methods on classification tasks'
)
