from setuptools import setup, find_packages

# Read README content
with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

config = {
    'description': 'An open-source Python toolkit offering a collection of efficient, easy-to-use functions for seismic data analysis.',
    'author': 'Gabriele Paoletti',
    'url': 'https://github.com/gabrielepaoletti/seismutils',
    'download_url': 'https://github.com/gabrielepaoletti/seismutils',
    'author_email': 'gabriele.paoletti@uniroma1.it',
    'version': '0.1.0-alpha',
    'python_requires': '>=3.6',
    'install_requires': ['matplotlib', 'numpy', 'pandas', 'pyproj'],
    'packages': find_packages(),
    'name': 'seismutils',
    'license': 'MIT',
    'keywords': 'seismology earthquake geophysics data-analysis'
}

setup(**config)