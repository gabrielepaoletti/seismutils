Spectral analysis
======================================
..
  #--------------------------------------------------------------------------------------------
  # seismutils.signal.fourier_transform
  #--------------------------------------------------------------------------------------------

.. autofunction:: seismutils.signal.fourier_transform

**Example usage**

This example demonstrates how to use the ``fourier_transform()`` function to compute the Fourier Transform of a seismic signal. The function generates a spectrum representation of the signal, which is crucial for frequency domain analysis. Optionally, it plots the signal alongside its Fourier Transform for a comprehensive visualization.

.. code-block:: python
    
  import seismutils.signal as sus

  # Assuming waveform is an np.adarray containing aplitude values

  fft = sus.fourier_transform(
        signals=waveform,
        sampling_rate=100,
        log_scale=True,
        plot=True,
  )

.. image:: https://imgur.com/njtQn5s.png
   :align: center
   :target: signal_processing.html#seismutils.signal.envelope

In this example, the ``fourier_transform()`` function is used to analyze a seismic waveform. By ``setting plot=True``, the function not only computes the Fourier Transform but also visualizes both the original waveform and its spectrum on a logarithmic scale, if ``log_scale=True``. This dual-visualization aids in understanding the frequency content and characteristics of the seismic signal.

.. note::
    If a multidimensional array is passed as the ``signals`` parameter, the fourier transform is calculated for each waveform independently. Ensure that the signals are organized such that each row represents a different signal for consistent computation of the fft across all waveforms.