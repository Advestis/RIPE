#!/usr/bin/env python

from distutils.core import setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()
    
setup(name='RIPE',
      version='0.1',
      description='RIPE algorithm',
      author='Vincent Margot',
      author_email='vincent.margot@hotmail.fr',
      url='',
      packages=['RIPE'],
      install_requires=requirements,
      )
