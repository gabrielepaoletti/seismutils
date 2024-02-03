Installation
======================================

This document provides detailed instructions for installing the ``seismutils`` package, a Python toolkit designed for seismic data analysis. The installation process has been streamlined to ensure compatibility and ease of integration into various scientific computing environments.

Prerequisites
-------------
Before proceeding with the installation, ensure that you have a compatible Python environment. ``seismutils`` requires Python >= 3.6. 

It's strongly recommended creating a virtual environment to maintain a clean workspace and avoid dependency conflicts. Tools such as ``conda`` or ``venv`` are suitable for creating isolated Python environments.

Stable release
---------------------------
The ``seismutils`` package is conveniently available for installation via ``pip``, the Python package manager. To install the latest stable release from the Python Package Index (PyPI), execute the following command in your terminal or command prompt:

.. code-block:: bash

    pip install seismutils

This command fetches and installs the most recent stable version of ``seismutils``, along with its required dependencies.

Development version
--------------------------------
For those interested in accessing the latest features and developments, the cutting-edge version of ``seismutils`` can be installed directly from the source code. Begin by cloning the project repository from GitHub with:

.. code-block:: bash

    git clone https://github.com/gabrielepaoletti/seismutils.git

Once the repository is cloned, navigate to the project's root directory:

.. code-block:: bash

    cd seismutils

Complete the installation by running:

.. code-block:: bash

    pip install .

This sequence of commands installs the current development version of ``seismutils`` from the cloned repository into your Python environment.

Should you encounter any issues during the installation process, feel free to submit an issue on the GitHub repository.