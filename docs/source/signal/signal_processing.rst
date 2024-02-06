Signal processing
======================================
..
  #--------------------------------------------------------------------------------------------
  # seismutils.signal.filter
  #--------------------------------------------------------------------------------------------

.. autofunction:: seismutils.signal.filter

**Parameter options**

The ``filter()`` function offers several parameters for fine-tuning signal processing, including the choice of filter type and the application of a tapering window. Below is an overview of the parameters related to filter type and tapering windows, with a brief explanation of each:

- ``filter_type``: Specifies the type of filter to be applied. The choice of filter type affects which frequencies are allowed to pass through:

  - ``'lowpass'``: Attenuates frequencies above a specified cutoff, allowing only lower frequencies to pass. Useful for removing high-frequency noise.
  - ``'highpass'``: Attenuates frequencies below a specified cutoff, allowing only higher frequencies to pass. Useful for removing low-frequency drift.
  - ``'bandpass'``: Allows frequencies within a specified range (between two cutoff frequencies) to pass, attenuating frequencies outside this range. Useful for isolating a band of interest.
  - ``'bandstop'``: Attenuates frequencies within a specified range, allowing frequencies outside this range to pass. Useful for removing a specific band of noise.

.. raw:: html
   <br>

- ``filter_mode``: Specifies the mode of filtering applied to the signal, determining how the filter affects the signal's phase and frequency characteristics. Available modes include:

  - ``'butterworth'``: This mode employs the Butterworth filter, known for its smooth frequency response. The design prioritizes a flat passband to eliminate ripples, ensuring consistent amplitude within this range. Its gradual transition to the stopband makes it ideal for scenarios where a uniform amplitude response is crucial.

  - ``'zero_phase'``: Implements a bidirectional filtering technique to achieve a zero-phase effect, counteracting any phase shifts introduced during filtering. By processing the signal forwards and then backwards, it retains the signal's original phase structure, making it indispensable for analyses where phase accuracy is paramount, such as seismic data interpretation or feature extraction from phase-sensitive signals.

  .. note::
    The choice of filter mode can significantly impact the analysis, especially in applications sensitive to phase shifts or where preserving the original phase information is crucial. 

.. raw:: html
   <br>

- ``taper_window``: Specifies the tapering window applied to the signal before filtering, which helps reduce edge effects and leakage in the frequency domain. Tapering involves gradually reducing the signal to zero at its beginning and end. Common taper windows used in seismology include:

  - ``'hann'``: Applies a Hann window, which has a sinusoidal shape and is known for its smooth tapering, reducing the amplitude of the signal to zero at both ends. This window is effective in minimizing the leakage effect in the frequency domain.
  - ``'hamming'``: Similar to the Hann window but with a slightly different shape, providing stronger attenuation of side lobes. It's widely used for its balance between main lobe width and side lobe attenuation.
  - ``'blackman'``: Offers higher side lobe attenuation than both Hann and Hamming windows. It's particularly useful for applications requiring minimal leakage at the expense of main lobe width.
  - ``'bartlett'``: Also known as a triangular window, it linearly decreases towards zero at the edges and is effective for spectral analysis where minimal side lobe content is desired.
  - ``'tukey'``: Combines a cosine taper and a constant section in the middle. The proportion of the cosine taper is adjustable, offering a compromise between time and frequency resolution.
  - ``None``: Applies no tapering, leaving the signal unmodified at the edges.

Additionally, if you wish to explore more tapering window options beyond those listed, consult the ``scipy.signal.get_window`` available at `SciPy Docs <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.get_window.html>`_.

.. note:: 
    Many of the window types supported by ``scipy.signal`` can be directly applied within the ``filter()`` function, offering extensive flexibility for signal processing tasks.

**Example usage**

This example demonstrates how to use the ``filter()`` function to apply a zero-phase shift band-pass filter to a seismic signal sampled at 100 Hz. The signal is then tapered using a Hann window before the filtering process.

.. code-block:: python
    
  import seismutils.signal as sus

  # Assuming waveform is an np.adarray containing aplitude values

  filtered_waveform = sus.filter(
      signals=waveform,
      sampling_rate=100,
      filter_type='bandpass',
      cutoff=(1, 20),
      taper_window='hann',
      filter_mode='zero_phase'
  )

.. image:: https://imgur.com/bLdbHjF.png
   :align: center
   :target: signal_processing.html#seismutils.signal.filter

In this example, the ``filter()`` function is used to isolate the frequency components of the signal between 1 Hz and 20 Hz range by applying an band-pass filter. 

The Hann window tapering is applied to the signal before filtering to reduce spectral leakage. The result is a smoother signal with reduced noise outside the desired frequency band.

.. note::
    If a multidimensional array is passed as the ``signals`` parameter, the filter is applied to each waveform independently. Ensure that the signals are organized such that each row represents a different signal for consistent application of the filter across all waveforms.

..
  #--------------------------------------------------------------------------------------------
  # seismutils.signal.envelope
  #--------------------------------------------------------------------------------------------

.. autofunction:: seismutils.signal.envelope

**Parameter options**

The ``envelope()`` function provides the ability to compute different types of envelopes for a given signal. The following parameter allows you to specify the type of envelope you wish to calculate:

- ``envelope_type``: Determines which envelope is calculated. Can be:

  - ``'positive'``: Computes the positive envelope.
  - ``'negative'``: Computes the negative envelope.
  - ``'both'``: Computes both the positive and negative envelopes, returning a tuple with two elements.

**Example usage**

This example demonstrates how to use the ``envelope()`` function to calculate both the positive and negative envelopes of a seismic signal. The signal envelopes are also plotted for visualization.

.. code-block:: python
    
  import seismutils.signal as sus

  # Assuming filtered_waveform is an np.adarray containing aplitude values

  pos_envelove, neg_envelope = sus.envelope(
      signals=filtered_waveform,
      type='both',
      plot=True,
  )

.. image:: https://imgur.com/1gwgsPg.png
   :align: center
   :target: signal_processing.html#seismutils.signal.envelope

In this example, the ``envelope()`` function is used to compute the envelopes of a seismic waveform. The plot flag is set to True, allowing the visualization of both the original signal and its envelopes.

.. note::
    If a multidimensional array is passed as the ``signals`` parameter, the envelope is calculated for each waveform independently. Ensure that the signals are organized such that each row represents a different signal for consistent computation of the envelope across all waveforms.