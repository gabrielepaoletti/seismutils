from setuptools import setup, find_packages

config = {
    'description': 'A private collection of essential functions and models for streamlined machine learning experiments.',
    'author': 'Gabriele Paoletti',
    'url': 'https://github.com/gabrielepaoletti/seismutils',
    'download_url': 'https://github.com/gabrielepaoletti/seismutils',
    'author_email': 'gabriele.paoletti@uniroma1.it',
    'version': '0.1',
    'python_requires': '>=3.6',
    'install_requires': ['matplotlib', 'numpy', 'pandas', 'pyproj'],
    'packages': find_packages(),
    'name': 'seismutils',
    'license': 'MIT',
    'keywords': 'seismology earthquake geophysics data-analysis'
}

setup(**config)