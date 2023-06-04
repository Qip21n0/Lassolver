from setuptools import setup, find_packages

setup(
	name = 'lassolver',
	version = '0.3.2-ssp',
	author = 'Ken Hisanaga',
	description='Lassolver is a Python package for Compressed Sensing and Distributed Compressed Sensing',
	install_requires = ['numpy','scipy','matplotlib','networkx',],
	packages = find_packages(),
	url = 'https://github.com/Qip21n0/Lassolver',
	license = 'MIT',
)