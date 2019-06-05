#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 14:28:35 2019

@author: abhijithneilabraham
"""

from setuptools import setup

setup(name='cdc',
      version='1.0',
      description='Image processing for cat and dog',
      author='Abhijith Neil Abraham',
      author_email='abhijithneilabrahampk@gmail.com',
      url='https://github.com/abhijithneilabraham/SoapClassification/tree/master/sample_classification',
      packages=['sample_classification'],
      install_requires = [
        "keras",
        "tensorflow",
        "opencv-python",
        "numpy",
      
    ],

     )
