try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'description': 'A private collection of essential functions and models for streamlined machine learning experiments.',
    'author': 'Gabriele Paoletti',
    'url': 'https://github.com/gabrielepaoletti/seismutils',
    'download_url': 'https://github.com/gabrielepaoletti/seismutils',
    'author_email': 'gabriele.paoletti@uniroma1.it',
    'version': '0.1',
    'install_requires': ['matplotlib', 'numpy', 'pandas', 'pyproj'],
    'packages': ['seismutils'],
    'name': 'seismutils'
}

setup(**config)