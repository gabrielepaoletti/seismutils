import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import butter, sosfilt, get_window

def filter(signals, sampling_rate, type, cutoff, order=5, taper_window=None, taper_params=None):
    '''
    Filter a signal with optional tapering, using specified filter parameters.
    
    :param signal: The input signal as a 1D numpy array or list.
    :param sampling_rate: The sampling frequency of the signal in Hz.
    :param type: The type of the filter ('lowpass', 'highpass', 'bandpass', 'bandstop'). Defaults to 'lowpass'.
    :param cutoff: The cutoff frequency or frequencies. For 'lowpass' and 'highpass', this is a single value. For 'bandpass' and 'bandstop', this is a tuple of two values (low cutoff, high cutoff).
    :param order: The order of the filter. Higher order means a steeper filter slope but can lead to instability or ringing. Defaults to 5.
    :param taper_window: The type of the tapering window ('hamming', 'hanning', 'blackman', 'tukey', etc.) to apply before filtering. If None, no tapering is applied. Defaults to None.
    :param taper_params: A dictionary of parameters for the tapering window, such as {'alpha': 0.5} for the Tukey window. Ignored if `taper_window` is None. Defaults to None.
    :return: The filtered signal as a 1D numpy array.
    
    The function applies a digital filter to the input signal. The filter type can be one of 'lowpass', 'highpass', 'bandpass', or 'bandstop'. Before filtering, an optional tapering window can be applied to the signal to reduce edge effects. The tapering window type and parameters can be customized.

    **Parameter Details:**

    - **type** (`str`): Specifies the type of filter to apply. Available options are:
        - ``lowpass``: Allows frequencies below the cutoff to pass through and attenuates frequencies above the cutoff.
        - ``highpass``: Allows frequencies above the cutoff to pass through and attenuates frequencies below the cutoff.
        - ``bandpass``: Allows frequencies within a specified range (cutoff tuple) to pass through and attenuates frequencies outside this range.
        - ``bandstop``: Attenuates frequencies within a specified range (cutoff tuple) while allowing frequencies outside this range to pass through.

    - **taper_window** (`str` or `None`): Determines the type of window used to taper the signal before filtering to address edge effects and spectral leakage, common issues in seismological analysis. Available options include:
        - ``hamming``: A symmetric window with a bell shape. Its smooth tapering minimizes spectral leakage, making it suitable for analyzing seismic waves where frequency content is important.
        - ``hanning``: Offers a similar bell shape with slightly different tapering properties compared to Hamming. It's effective for general seismic data processing, providing a good balance between time and frequency resolution.
        - ``blackman``: Features a wider main lobe than Hamming or Hanning, offering better sidelobe attenuation. This window is beneficial when the analysis requires minimizing spectral leakage, such as in high-resolution frequency analysis.
        - ``tukey``: The proportion of the window that is tapered can be adjusted, allowing control over the trade-off between time and frequency resolution. This flexibility is useful in seismology for windowing seismic signals before applying Fourier transforms.
        - ``bartlett``: A triangular window that linearly decreases to zero at both ends. It is suitable for smoothing seismic data, helping to reduce short-term spikes without significantly distorting the signal.
        - ``kaiser``: Allows adjusting the beta parameter to fine-tune the window's properties. It's particularly useful in filter design and spectral analysis tasks in seismology, where control over the sidelobe levels and main lobe width is crucial.
        - `None`: Applies no tapering to the signal. This option might be preferred in situations where the raw signal's characteristics need to be preserved for analysis.

    - **taper_params** (`dict`): Parameters for the tapering window, if applicable. This enables customization of the window's properties for specific seismological applications. For example:
        - For ``tukey``, `{'alpha': value}` controls the taper proportion, offering a way to adjust the balance between sidelobe attenuation and main lobe width, critical in spectral analysis of seismic events.
        - For ``kaiser``, `{'beta': value}` adjusts the shape of the window, allowing for precise control over spectral leakage and resolution, beneficial in seismological filter design.

    Additional parameters for these and other windows can be found in the ``scipy.signal.get_window`` documentation. It's important to note that not all windows require additional parameters.

    .. note::
       In seismology, the effectiveness of a tapering window depends on the nature of the seismic signals and the objectives of the analysis. Experimenting with different windows and parameters is advised to achieve optimal results in tasks such as noise reduction, signal enhancement, and spectral analysis.
    '''
    
    def butter_filter(order, cutoff, sampling_rate, btype):
            nyq = 0.5 * sampling_rate
            if btype in ['lowpass', 'highpass']:
                norm_cutoff = cutoff / nyq
            else:
                norm_cutoff = [c / nyq for c in cutoff]
            sos = butter(order, norm_cutoff, btype=btype, analog=False, output='sos')
            return sos
    
    # Apply tapering if requested
    def apply_taper(signal, window, params):
        if window is not None:
            if params is not None:
                window = get_window((window, *params.values()), len(signal))
            else:
                window = get_window(window, len(signal))
            return signal * window
        return signal
    
    sos = butter_filter(order, cutoff, sampling_rate, type)
    
    # Check if signals is a 2D array (multiple signals)
    if signals.ndim == 1:
        signals = np.array([signals])  # Convert to 2D array for consistency
    
    filtered_signals = []
    for signal in signals:
        tapered_signal = apply_taper(signal, taper_window, taper_params)
        filtered_signal = sosfilt(sos, tapered_signal)
        filtered_signals.append(filtered_signal)
    
    return np.array(filtered_signals) if len(filtered_signals) > 1 else filtered_signals[0]

def fourier_transform(signals, sampling_rate, plot=True, log_scale=True):
    '''
    Computes the Fourier Transform of the given signal(s) and optionally plots the spectra.

    :param signal: The input signal as a single waveform (1D numpy array) or multiple waveforms (2D numpy array where each row represents a different waveform).
    :param sampling_rate: The sampling rate of the signal(s) in Hz. Defaults to 1.0.
    :param plot: If True, plots the spectra of the waveform(s). Defaults to False.
    :param log_scale: If True and plot is True, plots the frequency axis on a logarithmic scale. Defaults to False.
    :return: The Fourier Transform of the signal(s) as a numpy array.

    This function computes the Fourier Transform using the Fast Fourier Transform (FFT) algorithm. It can process a single waveform or multiple waveforms simultaneously. When processing multiple waveforms, each waveform should be represented as a row in a 2D numpy array. The function returns the complex Fourier Transform results, which can be used to analyze the frequency components of the signal(s).
    '''
    # Compute the Fourier Transform
    ft = np.fft.fft(signals, axis=0 if signals.ndim == 1 else 1)
    freq = np.fft.fftfreq(signals.shape[-1], d=1/sampling_rate)

    if plot:
        # Plotting
        plt.figure(figsize=(10, 5))
        plt.title('Fourier Transform Spectrum', fontsize=14, fontweight='bold')

        if signals.ndim == 1:
            if log_scale:
                plt.loglog(freq[:len(freq)//2], np.abs(ft)[:len(freq)//2], color='black', linewidth=0.75)
            else:
                plt.plot(freq[:len(freq)//2], np.abs(ft)[:len(freq)//2], color='black', linewidth=0.75)
        else:
            for i in range(signals.shape[0]):
                if log_scale:
                    plt.loglog(freq[:len(freq)//2], np.abs(ft[i, :])[:len(freq)//2], linewidth=0.75, label=f'Waveform {i+1}')
                else:
                    plt.plot(freq[:len(freq)//2], np.abs(ft[i, :])[:len(freq)//2], linewidth=0.75, label=f'Waveform {i+1}')
            plt.legend(loc='best', frameon=False, fontsize=10)
        
        plt.xlabel('Frequency [Hz]', fontsize=12)
        plt.ylabel('Amplitude', fontsize=12)
        plt.grid(True, alpha=0.25, linestyle=':')
        plt.show()
    
    return ft