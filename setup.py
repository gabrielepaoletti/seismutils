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
    'version': '0.3.0',
    'python_requires': '>=3.11',
    'install_requires': ['matplotlib', 'numpy', 'pandas', 'pyproj', 'scipy'],
    'packages': find_packages(),
    'name': 'seismutils',
    'license': 'MIT',
    'keywords': 'seismology earthquake geophysics data-analysis',
    'long_description': long_description,
    'long_description_content_type': 'text/markdown',
}

setup(**config)