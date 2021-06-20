import os
from setuptools import setup


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
README_PATH = os.path.join(ROOT_DIR, 'README.md')

setup(
    name='calc4ap',
    author='devbruce',
    author_email='bruce93k@gmail.com',
    description='Easy AP Calculator with Python',
    long_description=open(README_PATH, encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    license='MIT',
    version='1.0.1',
    url='https://github.com/DevBruce/calc4ap',
    packages=['calc4ap'],
    package_data={
        'calc4ap': ['libs/*.py', 'utils/*.py'],
    },
    keywords=[
        'ap',
        'map',
        'object_detection',
        'pascal_voc',
        'coco',
        'deep_learning',
        'evaluation',
    ],
    python_requires='>=3.6.5',
    install_requires=[],
)
