import os
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec

from scipy.interpolate import interp1d
from scipy.signal import butter, sosfilt, sosfiltfilt, get_window, hilbert, stft, find_peaks
from matplotlib.colors import Normalize

def envelope(signals: np.ndarray,
             envelope_type='positive',
             plot=False,
             max_plots: int=10,
             save_figure: bool=False,
             save_name: str='envelope',
             save_extension: str='jpg'):
    '''
    Computes and optionally plots the envelope of one or more signals using the Hilbert transform. This function supports generating positive, negative, or both envelopes for the provided input signal(s). It is capable of processing a single signal or multiple signals batched in a multidimensional array, returning the computed envelope(s) accordingly.

    Parameters
    ----------
    signals : np.ndarray
        The input signal(s). This can be a 1D array for a single signal or a 2D array with each row representing a distinct signal.

    envelope_type : str, optional
        The type of envelope to compute. Options are 'positive' for the upper envelope, 'negative' for the lower envelope, and 'both' for both envelopes. Defaults to 'positive'.

    plot : bool, optional
        If True, generates plots for the input signal(s) alongside their computed envelope(s). Useful for visual analysis and verification of the envelope computation. Defaults to True.

    max_plots : int, optional
        Specifies the maximum number of signals to plot when `plot` is True and multiple signals are provided. This limit prevents excessive plotting when dealing with large datasets. Defaults to 10.

    save_figure : bool, optional
        If set to True, the function saves the generated plots using the provided base name and file extension. The default is False.

    save_name : str, optional
        The base name used for saving figures when `save_figure` is True. It serves as the prefix for file names. The default base name is 'envelope'.

    save_extension : str, optional
        The file extension to use when saving figures, such as 'jpg', 'png', etc... The default extension is 'jpg'.

    Returns
    -------
    np.ndarray or tuple(np.ndarray, np.ndarray)
        The computed envelope(s) of the input signal(s). Returns a single numpy array if 'positive' or 'negative' is chosen for ``envelope_type``, or a tuple of two numpy arrays if 'both' is selected.

    Examples
    --------
    .. code-block:: python

        import seismutils.signal as sus

        # Example usage with a numpy array containing signal amplitude values

        pos_envelope, neg_envelope = sus.envelope(
            signals=filtered_waveform,
            envelope_type='both',
            plot=True,
        )

    .. image:: https://i.imgur.com/HySfbQk.png
        :align: center
        :target: signal_processing.html#seismutils.signal.envelope
    '''
    analytical_signal = hilbert(signals, axis=-1)
    positive_envelope = np.abs(analytical_signal)
    negative_envelope = -positive_envelope
    
    if plot:
        num_signals_to_plot = signals.shape[0] if signals.ndim > 1 else 1
        for i in range(min(num_signals_to_plot, max_plots)):
            plt.figure(figsize=(10, 4))
            plt.title(f'Envelope {i+1}' if num_signals_to_plot > 1 else 'Envelope', fontsize=14, fontweight='bold')
            
            signal_to_plot = signals[i, :] if signals.ndim > 1 else signals
            pos_envelope_plot = positive_envelope[i, :] if signals.ndim > 1 else positive_envelope
            neg_envelope_plot = negative_envelope[i, :] if signals.ndim > 1 else negative_envelope
            
            plt.plot(signal_to_plot, color='black', linewidth=0.75, label='Signal')
            
            if envelope_type in ['positive', 'both']:
                plt.plot(pos_envelope_plot, color='red', linewidth=0.75, label='Positive envelope')
            
            if envelope_type in ['negative', 'both']:
                plt.plot(neg_envelope_plot, color='blue', linewidth=0.75, label='Negative envelope')
        
            plt.xlabel('Samples [#]', fontsize=12)
            # plt.ylabel('Amplitude', fontsize=12)
            plt.xlim(0, len(signal_to_plot))
            plt.grid(True, alpha=0.25, axis='x', linestyle=':')
            plt.legend(loc='best', frameon=False, fontsize=12)
            
            if save_figure:
                os.makedirs('./seismutils_figures', exist_ok=True)
                fig_name = os.path.join('./seismutils_figures', f'{save_name}_{i+1}.{save_extension}')
                plt.savefig(fig_name, dpi=300, bbox_inches='tight', facecolor=None)
            
            plt.show()
            if i >= (max_plots - 1):
                break
    
    if envelope_type == 'positive':
        return positive_envelope
    elif envelope_type == 'negative':
        return negative_envelope
    else:  # 'both'
        return positive_envelope, negative_envelope

def filter(signals: np.ndarray, 
           sampling_rate: int,
           filter_type: str,
           cutoff: int, 
           order: int=5, 
           taper_window: str=None,
           taper_params: dict=None, 
           filter_mode: str='zero-phase'):
    '''
    Applies a digital filter to input signal(s), offering optional tapering to minimize edge effects.

    Parameters
    ----------
    signals : np.ndarray
        The input signal(s). This can be a 1D array for a single signal or a 2D array with each row representing a distinct signal.

    sampling_rate : int
        The sampling frequency of the signal(s) in Hz, determining the temporal resolution of the data.

    filter_type : str
        The type of filter to apply. Options include 'lowpass', 'highpass', 'bandpass', and 'bandstop'.

    cutoff : float or tuple(float, float)
        The cutoff frequency(ies) for the filter. Use a single value for 'lowpass'/'highpass' filters and a tuple for 'bandpass'/'bandstop' filters.

    order : int, optional
        The order of the filter, which affects the sharpness of the frequency cutoff. Higher order filters have a steeper rolloff but can introduce phase distortions or ringing artifacts.

    taper_window : str, optional
        The type of tapering window to apply before filtering. Options include 'hann', 'hamming', 'blackman', 'bartlett', 'tukey', or None for no tapering.

    taper_params : dict, optional
        Parameters for the tapering window, applicable only if `taper_window` is not None. These parameters are specific to the chosen window type.

    filter_mode : str, optional
        The filtering mode, which can be 'butterworth' for a smooth frequency response or 'zero-phase' for filtering without introducing phase shifts.

    Returns
    -------
    filtered_signals : np.ndarray
        The filtered signal(s), presented in the same format as the input coordinates.

    Notes
    -----
    - **Filter Type**: Determines the operational frequency range of the filter:
        - ``'lowpass'``: Attenuates frequencies above the cutoff while preserving lower frequencies.
        - ``'highpass'``: Attenuates frequencies below the cutoff while preserving higher frequencies.
        - ``'bandpass'``: Only frequencies within a specified range are preserved, with others attenuated.
        - ``'bandstop'``: Frequencies within a specified range are attenuated, while those outside are preserved.

    - **Filter Mode**: Influences the signal's phase characteristics post-filtering:
        - ``'butterworth'``: Known for a flat frequency response in the passband, minimizing amplitude distortion.
        - ``'zero-phase'``: Employs forward and reverse filtering to negate phase shifts, maintaining the original signal phase.

        .. html::
            <br>
        
        .. image:: https://i.imgur.com/ixHUpSN.png
            :align: center
            :target: signal_processing.html#seismutils.signal.filter
        
    - **Taper Window**: Reduces edge effects by applying a windowing function to the signal before filtering:
        - Options like ``'hann'``, ``'hamming'``, ``'blackman'``, ``'bartlett'``, and ``'tukey'`` each offer unique trade-offs between frequency leakage and resolution.
        - A ``None`` value skips tapering, leaving signal edges unaltered.

    If you wish to explore more tapering window options beyond those listed, consult the ``scipy.signal.get_window`` available at `SciPy Docs <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.get_window.html>`_.
    
    Examples
    --------
    .. code-block:: python

        import seismutils.signal as sus

        # Assume waveform is an np.ndarray containing amplitude values

        filtered_signal = sus.filter(
            signals=waveform,
            sampling_rate=100,
            filter_type='highpass',
            cutoff=2,
            taper_window='hann',
            filter_mode='zero-phase',
        )
    
    .. image:: https://imgur.com/Z99MknB.png
        :align: center
        :target: signal_processing.html#seismutils.signal.filter
    '''
    def butter_filter(order, cutoff, sampling_rate, filter_type, mode):
        # Normalize the cutoff frequency with respect to Nyquist frequency
        nyq = 0.5 * sampling_rate
        if isinstance(cutoff, (list, tuple)):  # For bandpass and bandstop filters
            norm_cutoff = [c / nyq for c in cutoff]
        else:  # For lowpass and highpass filters
            norm_cutoff = cutoff / nyq
        
        # Create second-order sections for the Butterworth filter
        sos = butter(order, norm_cutoff, btype=filter_type, analog=False, output='sos')
        
        # Return a function that applies the filter to a signal x
        if mode == 'zero_phase':
            return lambda x: sosfiltfilt(sos, x)
        else:
            return lambda x: sosfilt(sos, x)

    def apply_taper(signal, window_type, params):
        # Apply a taper to the signal if requested
        if window_type is not None:
            # Generate the window with the given parameters
            window = get_window((window_type, *params), len(signal)) if params else get_window(window_type, len(signal))
            return signal * window
        return signal
    
    # Ensure signals is a 2D array for consistency
    signals = np.atleast_2d(signals)
    
    # Create the filter function
    filter_func = butter_filter(order, cutoff, sampling_rate, filter_type, filter_mode)
    
    # Apply the filter and taper to each signal
    filtered_signals = []
    for signal in signals:
        tapered_signal = apply_taper(signal, taper_window, taper_params)
        filtered_signal = filter_func(tapered_signal)
        filtered_signals.append(filtered_signal)
    
    # Return the filtered signals in their original shape
    return np.squeeze(filtered_signals)

def fourier_transform(signals: np.ndarray, sampling_rate: int, log_scale=True, plot=True, plot_waveform=True, max_plots=10, save_figure=False, save_name: str='fourier_transform', save_extension: str='jpg'):
    '''
    Performs a Fourier Transform on a set of signals and optionally plots the amplitude spectrum.

    This function utilizes NumPy's efficient Fast Fourier Transform (FFT) to analyze the frequency content of signals. It supports visualization of both the original waveform and its corresponding amplitude spectrum, with options for logarithmic amplitude scaling, selective plotting, and figure saving.

    :param signals: Input signal(s), either a single signal as a 1D array or multiple signals as a 2D array where each row represents a signal.
    :type signals: np.ndarray
    :param sampling_rate: Sampling frequency of the signal(s) in Hz, used to calculate the frequency axis for the spectrum.
    :type sampling_rate: int
    :param log_scale: If True, the amplitude spectrum is plotted on a logarithmic scale for better visibility of wide amplitude ranges. Defaults to True.
    :type log_scale: bool
    :param plot: If True, generates plots for the Fourier Transform results and, optionally, the waveform. Defaults to True.
    :type plot: bool
    :param plot_waveform: If True, plots the original waveform(s) above their corresponding amplitude spectrum. Defaults to True.
    :type plot_waveform: bool
    :param max_plots: Limits the number of signal plots to this number, useful for datasets with many signals. Defaults to 10.
    :type max_plots: int
    :param save_figure: If True, saves the generated plot(s) as image files in a specified directory. Defaults to False.
    :type save_figure: bool
    :param save_name: The base filename for saved plots, to which a numeric suffix will be appended for each signal when multiple signals are plotted. Defaults to 'fourier_transform'.
    :type save_name: str
    :param save_extension: File extension for the saved plot images (e.g., 'jpg', 'png'). Defaults to 'jpg'.
    :type save_extension: str
    :return: The Fourier Transform result up to the Nyquist frequency, providing the positive frequency components of the signal(s) and frequency bins.
    :rtype: np.ndarray

    **Usage example**

    .. code-block:: python

        import seismutils.signal as sus

        # Assuming waveform is an np.ndarray containing amplitude values
        
        fft = sus.fourier_transform(
              signals=waveform,
              sampling_rate=100,
              log_scale=True,
              plot=True
        )

    .. image:: https://imgur.com/njtQn5s.png
       :align: center
       :target: spectral_analysis.html#seismutils.signal.fourier_transform

    The output is a numpy array containing the Fourier Transform of the input signal(s). If ``plot=True``, the function also generates a plot (or plots) illustrating the signal(s) and their frequency spectrum.
    '''

    # Adjust for multidimensional input
    if signals.ndim == 1:
        signals = signals[np.newaxis, :]  # Make 1D array 2D for uniform processing
    multiple_waveforms = signals.shape[0] > 1

    # Initialize a list to hold Fourier Transform results
    ft_results, freqs = [], []

    for index, signal in enumerate(signals):
        # Limit plotting to max_plots
        if index >= max_plots:
            break
        
        # Compute the Fourier Transform using NumPy's FFT
        ft = np.fft.fft(signal)
        freq = np.fft.fftfreq(signal.size, d=1/sampling_rate)
        N = signal.size
        ft = ft / N  # Normalize the amplitudes
        ft_results.append(ft[:N // 2])  # Store positive frequency components
        freqs.append(freq)

        if plot:
            # Plot setup
            if plot_waveform:
                fig, axs = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [1, 2]})
            else:
                fig, axs = plt.subplots(1, 1, figsize=(10, 6))
                axs = [axs]  # Make it iterable for the upcoming loop

            # Determine titles based on the number of waveforms
            waveform_title = f'Waveform {index+1}' if multiple_waveforms else 'Waveform'
            transform_title = f'Fourier Transform {index+1}' if multiple_waveforms else 'Fourier Transform'

            # Plot the waveform
            if plot_waveform:
                axs[0].plot(signal, linewidth=0.75, color='k')
                axs[0].set_title(waveform_title, fontsize=14, fontweight='bold')
                axs[0].set_xlabel('Samples', fontsize=12)
                axs[0].set_ylabel('Amplitude', fontsize=12)
                axs[0].set_xlim(0, len(signal))
                axs[0].grid(True, alpha=0.25, linestyle=':')

            # Plot the Fourier Transform
            ax = axs[-1]
            half_point = N // 2
            freq = freq[:half_point]
            amplitude_spectrum = np.abs(ft)[:half_point] * 2  # Multiply by 2 to account for symmetrical nature of FFT output

            if log_scale:
                ax.loglog(freq, amplitude_spectrum, color='black', linewidth=0.75)
            else:
                ax.plot(freq, amplitude_spectrum, color='black', linewidth=0.75)

            ax.set_title(transform_title, fontsize=14, fontweight='bold')
            ax.set_xlabel('Frequency [Hz]', fontsize=12)
            ax.set_ylabel('Amplitude', fontsize=12)
            ax.grid(True, alpha=0.25, which='both', linestyle=':')

            plt.tight_layout()
            if save_figure and index < max_plots:
                os.makedirs('./seismutils_figures', exist_ok=True)
                fig_name = f'./seismutils_figures/{save_name}_{index+1}.{save_extension}'
                plt.savefig(fig_name, dpi=300, bbox_inches='tight')

            plt.show()

    return np.array(ft_results), np.array(freqs) if multiple_waveforms else ft_results[0], freqs[0]

def spectrogram(signals: np.ndarray, sampling_rate: int, nperseg: int=128, noverlap: float=None, log_scale: bool=False, zero_padding_factor: int=8, return_data: bool=False, plot_waveform: bool=True, max_plots: int=10, colorbar: bool=False, cmap: str='jet', save_figure: bool=False, save_name: str='spectrogram', save_extension: str='jpg'):
    '''
    Generates and optionally plots spectrograms for provided signals, with an option to display the waveform.

    This function computes the Short-Time Fourier Transform (STFT) to generate spectrograms for a set of signals. It supports various configurations, including log-scale amplitude, zero padding for higher frequency resolution, and customizable plotting options.

    .. note::
        When provided with a multidimensional array containing multiple waveforms (one per row), this function processes each waveform independently. It generates and plots a spectrogram for each waveform, adhering to the specified parameters.

    :param signals: Input signals, a 2D numpy array where each row represents a signal.
    :type signals: np.ndarray
    :param sampling_rate: Sampling frequency of the signals in Hz.
    :type sampling_rate: int
    :param nperseg: Length of each segment for the STFT, in samples. Defaults to 128.
    :type nperseg: int, optional
    :param noverlap: Number of points to overlap between segments. If None, defaults to 75% of `nperseg`.
    :type noverlap: float, optional
    :param log_scale: If True, plots the spectrogram's amplitude in logarithmic scale.
    :type log_scale: bool, optional
    :param zero_padding_factor: Factor to increase the FFT points via zero-padding, enhancing frequency resolution.
    :type zero_padding_factor: int, optional
    :param return_data: If True, returns a list containing the spectrogram data, frequencies, and times for each signal. Defaults to False.
    :type return_data: bool, optional
    :param plot_waveform: If True, plots the waveform above its corresponding spectrogram.
    :type plot_waveform: bool, optional
    :param max_plots: Maximum number of spectrograms to plot. Useful for large datasets. Defaults to 10.
    :type max_plots: int, optional
    :param colorbar: If True, adds a colorbar indicating amplitude or power scale next to the spectrogram.
    :type colorbar: bool, optional
    :param cmap: Color map for plotting the spectrogram. Defaults to ``'jet'``.
    :type cmap: str, optional
    :param save_figure: If True, saves the generated plots in a directory.
    :type save_figure: bool, optional
    :param save_name: Name under which the figure will be saved. Default ``'fourier_transform'``
    :type save_name: str, optional
    :param save_extension: Extension with which the image will be saved. Default ``'jpg'``
    :type save_extension: str, optional
    :return: None. The function directly plots the spectrogram(s) and optionally saves them as files.

    **Usage Example**

    .. code-block:: python

        import seismutils.signal as sus

        # Assume waveform is an np.ndarray containing amplitude values

        spectrogram_data = sus.spectrogram(
            signals=waveform,
            sampling_rate=100,
            plot_waveform=True,
            return_data=True
        )

    .. image:: https://imgur.com/0J4VGQC.png
       :align: center
       :target: spectral_analysis.html#seismutils.signal.spectrogram
    '''

    spectrogram_data = []

    # If signals is 1D array, convert it to 2D array with one row
    if signals.ndim == 1:
        signals = np.array([signals])

    # Limit the number of signals to plot
    num_signals = min(len(signals), max_plots)

    for i, signal in enumerate(signals[:num_signals]):
        # Normalize the signal by subtracting the mean
        signal -= np.mean(signal)

        # If noverlap is not set, set it to 75% of nperseg
        if noverlap is None:
            noverlap = int(nperseg * 0.75)

        # Calculate the Short-Time Fourier Transform (STFT) with zero padding
        nfft = nperseg * zero_padding_factor
        frequencies, times, Zxx = stft(signal, fs=sampling_rate, window='hann', nperseg=nperseg, noverlap=noverlap, nfft=nfft)
        time = np.arange(0, len(signal)) / sampling_rate

        # Calculate the squared magnitude of the STFT (spectrogram)
        spectrogram = np.abs(Zxx)**2

        # Convert to decibels if log scale is requested
        if log_scale:
            spectrogram = 10 * np.log10(spectrogram)
            vmin, vmax = np.percentile(spectrogram, [5, 95])
        else:
            spectrogram = np.sqrt(spectrogram)*2
            vmin, vmax = np.min(spectrogram), np.max(spectrogram)

        spectrogram_data.append((spectrogram, frequencies, times))

        # Plot configuration
        fig = plt.figure(figsize=(10, 7))  # Adjust the figure size as needed
        gs = gridspec.GridSpec(2, 2, width_ratios=[25, 1], height_ratios=[1, 3], wspace=0.05, hspace=0.2)
        
        if plot_waveform:
            # Plot the waveform
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.plot(time, signal, color='k', linewidth=0.75)
            ax1.set_title('Waveform' if len(signals) == 1 else f'Waveform {i+1}', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Amplitude', fontsize=12)
            ax1.set_xlim(0, round(time[-1]))
            ax1.grid(True, alpha=0.25, axis='x', linestyle=':')
            ax1.set_xticklabels([])

        # Plot the spectrogram
        ax2 = fig.add_subplot(gs[1, 0])
        pcm = ax2.pcolormesh(times, frequencies, spectrogram, shading='gouraud', norm=Normalize(vmin, vmax), cmap=cmap)
        ax2.set_title('Spectrogram' if len(signals) == 1 else f'Spectrogram {i+1}', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Frequency [Hz]', fontsize=12)
        ax2.set_xlabel('Time [s]', fontsize=12)
        ax2.set_xlim(0, round(time[-1]))

        # Colorbar setup
        if colorbar:
            # Create an axis for colorbar. It spans all rows but is in the second column.
            cax = fig.add_subplot(gs[1, -1])
            cbar = plt.colorbar(pcm, cax=cax)
            cbar.set_label('Power spectral density [dB]' if log_scale else 'Amplitude', fontsize=12)
            cbar.ax.tick_params(labelsize=12)

        if save_figure:
            os.makedirs('./seismutils_figures', exist_ok=True)
            fig_name = os.path.join('./seismutils_figures', f'{save_name}_{i+1 if len(signals) > 1 else ""}.{save_extension}')
            plt.savefig(fig_name, dpi=300, bbox_inches='tight', facecolor=None)

        plt.show()
    
    if return_data:
        return spectrogram_data