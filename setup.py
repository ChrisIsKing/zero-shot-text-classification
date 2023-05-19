from setuptools import setup, find_packages

VERSION = '0.1.1'
DESCRIPTION = """
code and data for the Findings of ACL'23 paper Label Agnostic Pre-training for Zero-shot Text Classification 
by Christopher Clarke, Yuzhao Heng, Yiping Kang, Krisztian Flautner, Lingjia Tang and Jason Mars
 """

setup(
    name='zeroshot-classifier',
    version=VERSION,
    license='MIT',
    author='Christopher Clarke & Yuzhao Heng',
    author_email='csclarke@umich.edu',
    description=DESCRIPTION,
    long_description=DESCRIPTION,
    url='https://github.com/ChrisIsKing/zero-shot-text-classification',
    download_url='https://github.com/ChrisIsKing/zero-shot-text-classification/archive/refs/tags/v0.1.0.tar.gz',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'gdown==4.5.4', 'openai==0.25.0', 'requests==2.28.1', 'tenacity==8.1.0',
        'spacy==3.2.2', 'nltk==3.7', 'scikit-learn==1.1.3',
        'torch==1.12.0', 'sentence-transformers==2.2.0', 'transformers==4.16.2',
        'datasets==1.18.3',
        'stefutils==0.22.2'
    ],
    keywords=['python', 'nlp', 'machine-learning', 'deep-learning', 'text-classification', 'zero-shot-classification'],
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Environment :: GPU :: NVIDIA CUDA',
        'Environment :: GPU :: NVIDIA CUDA :: 11.6',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Operating System :: MacOS',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Visualization'
    ]
)
