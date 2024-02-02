Installation
======================================

``seismutils`` can be installed in two primary ways. For both methods, it's recommended to install the module within a virtual environment to avoid conflicts with other package dependencies and system-wide Python settings. You might use virtual environment tools like ``conda`` or ``venv`` for this purpose.

The simplest and most straightforward method is to use ``pip``, the Python package installer. Just run the following command:

.. code-block:: bash

    pip install seismutils

This will install the latest stable version of Seismutils from the Python Package Index (PyPI).

If you prefer to install the latest development version or if you'd like to modify the code, you can install it from source. First, clone the repository from GitHub:

.. code-block:: bash

    git clone https://github.com/gabrielepaoletti/seismutils.git

After cloning, navigate to the root directory of the repository:

.. code-block:: bash

    cd seismutils

Then, install the package using:

.. code-block:: bash

    pip install .

This command will install ``seismutils`` into your current Python environment directly from the source code.