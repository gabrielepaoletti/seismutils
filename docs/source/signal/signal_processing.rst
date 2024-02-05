Signal processing
======================================

.. autofunction:: seismutils.signal.filter

**Parameter options**

The ``filter()`` function offers several parameters for fine-tuning signal processing, including the choice of filter type and the application of a tapering window. Below is an overview of the parameters related to filter type and tapering windows, with a brief explanation of each:

- ``type``: Specifies the type of filter to be applied. The choice of filter type affects which frequencies are allowed to pass through:

  - ``'lowpass'``: Attenuates frequencies above a specified cutoff, allowing only lower frequencies to pass. Useful for removing high-frequency noise.
  - ``'highpass'``: Attenuates frequencies below a specified cutoff, allowing only higher frequencies to pass. Useful for removing low-frequency drift.
  - ``'bandpass'``: Allows frequencies within a specified range (between two cutoff frequencies) to pass, attenuating frequencies outside this range. Useful for isolating a band of interest.
  - ``'bandstop'``: Attenuates frequencies within a specified range, allowing frequencies outside this range to pass. Useful for removing a specific band of noise.

.. raw:: html

   <br>

- ``taper_window``: Specifies the tapering window applied to the signal before filtering, which helps reduce edge effects and leakage in the frequency domain. Tapering involves gradually reducing the signal to zero at its beginning and end. Common taper windows used in seismology include:

  - ``'hann'``: Applies a Hann window, which has a sinusoidal shape and is known for its smooth tapering, reducing the amplitude of the signal to zero at both ends. This window is effective in minimizing the leakage effect in the frequency domain.
  - ``'hamming'``: Similar to the Hann window but with a slightly different shape, providing stronger attenuation of side lobes. It's widely used for its balance between main lobe width and side lobe attenuation.
  - ``'blackman'``: Offers higher side lobe attenuation than both Hann and Hamming windows. It's particularly useful for applications requiring minimal leakage at the expense of main lobe width.
  - ``'bartlett'``: Also known as a triangular window, it linearly decreases towards zero at the edges and is effective for spectral analysis where minimal side lobe content is desired.
  - ``'tukey'``: Combines a cosine taper and a constant section in the middle. The proportion of the cosine taper is adjustable, offering a compromise between time and frequency resolution.
  - ``None``: Applies no tapering, leaving the signal unmodified at the edges.

.. note::
    Additionally, if you wish to explore more tapering window options beyond those listed, consult the ``scipy.signal.get_window`` available at `SciPy Docs <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.get_window.html>`_.
    
Many of the window types supported by ``scipy.signal`` can be directly applied within the ``filter()`` function, offering extensive flexibility for signal processing tasks.

.. autofunction:: seismutils.signal.envelope
