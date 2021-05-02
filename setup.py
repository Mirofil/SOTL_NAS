from setuptools import setup, find_packages

setup(name='sotl_nas',
      packages=find_packages(where='.', exclude=['tests']),
      package_dir={'sotl_nas':'sotl_nas'}
     )
