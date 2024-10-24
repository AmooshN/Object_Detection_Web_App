# setup.py
from setuptools import setup
setup(
    name='my_app',
    install_requires=[
        'git+https://github.com/facebookresearch/detectron2.git'
    ]
)
