from setuptools import setup
from setuptools import find_packages

setup(name='pl',
      version='1.0',
      description='Effective Peel Learning for Small Data with Structured Features',
      author='Anonymous Author',
      author_email='abc@gmail.com',
      url='https://github.com/nips-2019-pl/nips-2019-7311',
      download_url='https://github.com/nips-2019-pl/nips-2019-7311',
      license='MIT',
      install_requires=['numpy',
                        'pytorch',
                        'sklearn',
                        'pandas',
                        
                        ],
      package_data={'pl': ['README.md']},
      packages=find_packages())