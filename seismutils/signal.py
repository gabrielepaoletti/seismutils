import numpy as np
from scipy.signal import butter, sosfilt, get_window

def filter(signal, fs, type, cutoff, order=5, taper_window=None, taper_params=None):
    '''
    Filter a signal with optional tapering, using specified filter parameters.
    
    :param signal: The input signal as a 1D numpy array or list.
    :param fs: The sampling frequency of the signal in Hz.
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
    
    # Define the filter design function
    def butter_filter(order, cutoff, fs, btype):
        nyq = 0.5 * fs  # Nyquist frequency
        if btype in ['lowpass', 'highpass']:
            norm_cutoff = cutoff / nyq  # Normalize the frequency
        else:  # For 'bandpass' and 'bandstop', normalize the frequency range
            norm_cutoff = [c / nyq for c in cutoff]
        sos = butter(order, norm_cutoff, btype=btype, analog=False, output='sos')  # Generate filter coefficients
        return sos
    
    # Apply tapering if requested
    if taper_window is not None:
        if taper_params is not None:  # If taper parameters are provided, use them
            window = get_window((taper_window, *taper_params.values()), len(signal))
        else:  # Otherwise, generate the window without additional parameters
            window = get_window(taper_window, len(signal))
        signal *= window  # Apply the window to the signal
    
    # Filter the signal
    sos = butter_filter(order, cutoff, fs, type)  # Get the filter coefficients
    filtered_signal = sosfilt(sos, signal)  # Apply the filter
    
    return filtered_signal
