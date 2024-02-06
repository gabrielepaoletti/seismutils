import os
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec

from scipy.signal import butter, sosfilt, sosfiltfilt, get_window, hilbert, stft
from matplotlib.colors import Normalize

def envelope(signals: np.ndarray, envelope_type='positive', plot=False, max_plots: int=10, save_figure: bool=False, save_name: str='envelope', save_extension: str='jpg'):
    '''
    Computes the envelope of a signal using the Hilbert transform. This function can generate the positive, negative, or both envelopes of the input signal(s) and optionally plot the results.

    The Hilbert transform is used to compute the analytical signal, from which the envelope is derived. The envelope can be useful for amplitude modulation analysis, feature extraction, and signal analysis tasks.

    .. note::
        When provided with a multidimensional array containing multiple waveforms (one per row), this function processes each waveform independently. It computes and returns the envelope(s) for each waveform, maintaining the input structure.

    :param signals: Input signal(s) as a numpy array. Can be a single signal (1D array) or multiple signals (2D array).
    :type signals: np.ndarray
    :param envelope_type: Specifies the type of envelope to compute and return (``positive``, ``negative``, or ``both``).
    :type envelope_type: str, optional
    :param plot: If True, plots the input signal(s) along with their computed envelope(s).
    :type plot: bool, optional
    :param max_plots: The maximum number of plots to generate when handling multiple signals. Default is 10.
    :type max_plots: int, optional
    :param save_figure: If True, saves the generated plots in a directory.
    :type save_figure: bool, optional
    :param save_name: Name under which the figure will be saved. Default ``'fourier_transform'``
    :type save_name: str, optional
    :param save_extension: Extension with which the image will be saved. Default ``'jpg'``
    :type save_extension: str, optional
    :return: The computed envelope(s) of the input signal(s). Returns a single array if ``positive`` or ``negative`` is chosen, or two arrays if ``both`` is selected.
    :rtype: np.ndarray or tuple(np.ndarray, np.ndarray)

    **Usage example**

    .. code-block:: python

        import seismutils.signal as sus

        # Assuming filtered_waveform is an np.ndarray containing amplitude values
        
        pos_envelope, neg_envelope = sus.envelope(
            signals=filtered_waveform,
            envelope_type='both',
            plot=True,
        )

    .. image:: https://imgur.com/1gwgsPg.png
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
        
            plt.xlabel('Sample', fontsize=12)
            plt.ylabel('Amplitude', fontsize=12)
            plt.xlim(0, len(signal_to_plot))
            plt.grid(True, alpha=0.25, linestyle=':')
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

def filter(signals: np.ndarray, sampling_rate: int, filter_type: str, cutoff: float, order=5, taper_window=None, taper_params=None, filter_mode='zero-phase'):
    '''
    Applies a digital filter to the input signal(s) with optional tapering. The function supports various filter types, cutoff frequencies, and filtering modes for flexible signal processing.

    This function allows for precise manipulation of signal frequencies through filtering, with the capability to apply a tapering window beforehand to minimize edge effects. It supports multiple filtering modes, including a zero-phase filtering option to preserve the phase characteristics of the signal.
    
    .. note::
        When provided with a multidimensional array containing multiple waveforms (one per row), the function processes each waveform independently, applying the specified filtering and tapering operations. The returned array will contain all the filtered waveforms, preserving the input structure.

    :param signals: Input signal(s) as a numpy array. Can be a single signal (1D array) or multiple signals (2D array).
    :type signals: np.ndarray
    :param sampling_rate: Sampling frequency of the signal(s) in Hz.
    :type sampling_rate: int
    :param filter_type: Type of filter ('lowpass', 'highpass', 'bandpass', 'bandstop').
    :type filter_type: str
    :param cutoff: Cutoff frequency(ies). Single value for 'lowpass'/'highpass'; tuple for 'bandpass'/'bandstop'.
    :type cutoff: float or tuple(float, float)
    :param order: Order of the filter. A higher order results in a sharper frequency cutoff but may introduce artifacts.
    :type order: int, optional
    :param taper_window: Type of tapering window to apply ('hann', 'hamming', 'blackman', 'bartlett', 'tukey', or None).
    :type taper_window: str, optional
    :param taper_params: Parameters for the tapering window. Ignored if `taper_window` is None.
    :type taper_params: dict, optional
    :param filter_mode: Filtering mode ('butterworth' or 'zero-phase').
    :type filter_mode: str, optional
    :return: Filtered signal(s) as a numpy array.
    :rtype: np.ndarray

    **Parameter details**

    - ``filter_type``: Dictates the filter's operational frequency range, affecting which frequencies are attenuated or preserved. Choices include:
        - ``'lowpass'``: Suppresses frequencies above the cutoff, allowing lower frequencies to pass through.
        - ``'highpass'``: Suppresses frequencies below the cutoff, allowing higher frequencies to pass through.
        - ``'bandpass'``: Only allows frequencies within a specific range to pass, attenuating those outside.
        - ``'bandstop'``: Attenuates frequencies within a specific range, allowing those outside to pass.

    - ``filter_mode``: Determines the filter's impact on the signal's phase characteristics:
        - ``'butterworth'``: Offers a smooth, flat frequency response in the passband and is designed to maintain a uniform amplitude.
        - ``'zero-phase'``: Utilizes forward and reverse filtering to eliminate phase shifts, preserving the original phase structure of the signal.

    - ``taper_window``: Applies a windowing function to the signal before filtering, reducing potential edge effects:
        - Options include classic window functions like ``'hann'``, ``'hamming'``, ``'blackman'``, ``'bartlett'``, and ``'tukey'``, each with unique characteristics affecting the signal's frequency leakage and resolution.
        - Setting to ``None`` bypasses tapering, leaving the signal edges unaltered.
        
    Additionally, if you wish to explore more tapering window options beyond those listed, consult the ``scipy.signal.get_window`` available at `SciPy Docs <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.get_window.html>`_.
        
    .. note:: 
        Many of the window types supported by ``scipy.signal`` can be directly applied within the ``filter()`` function, offering extensive flexibility for signal processing tasks.

    **Usage example**

    .. code-block:: python

        import seismutils.signal as sus

        # Assuming waveform is an np.adarray containing amplitude values

        fft = sus.fourier_transform(
            signals=waveform,
            sampling_rate=100,
            log_scale=True,
            plot=True,
        )

    .. image:: https://imgur.com/bLdbHjF.png
       :align: center
       :target: signal_processing.html#seismutils.signal.filter
    '''
    
    def butter_filter(order, cutoff, sampling_rate, filter_type, mode):
            nyq = 0.5 * sampling_rate
            if filter_type in ['lowpass', 'highpass']:
                norm_cutoff = cutoff / nyq
            else:
                norm_cutoff = [c / nyq for c in cutoff]
            sos = butter(order, norm_cutoff, btype=filter_type, analog=False, output='sos')
            if mode == 'zero_phase':
                return lambda x: sosfiltfilt(sos, x)
            else:
                return lambda x: sosfilt(sos, x)
    
    # Apply tapering if requested
    def apply_taper(signal, window, params):
        if window is not None:
            if params is not None:
                window = get_window((window, *params.values()), len(signal))
            else:
                window = get_window(window, len(signal))
            return signal * window
        return signal
    
    filter_func = butter_filter(order, cutoff, sampling_rate, filter_type, filter_mode)
    
    # Check if signals is a 2D array (multiple signals)
    if signals.ndim == 1:
        signals = np.array([signals])  # Convert to 2D array for consistency
    
    filtered_signals = []
    for signal in signals:
        tapered_signal = apply_taper(signal, taper_window, taper_params)
        filtered_signal = filter_func(tapered_signal)
        filtered_signals.append(filtered_signal)
    
    return np.array(filtered_signals) if len(filtered_signals) > 1 else filtered_signals[0]

def fourier_transform(signals: np.ndarray, sampling_rate: int, log_scale=True, plot=True, plot_waveform=True, max_plots=10, save_figure=False, 
                      save_name: str='fourier_transform', save_extension: str='jpg'):
    '''
    Performs a Fourier Transform on a set of signals and optionally plots the results.

    This function computes the Fourier Transform using numpy's FFT implementation and can generate plots for both individual and multiple signals. It supports plotting in logarithmic scale for better visualization of the frequency components.

    .. note::
        If ``plot`` is set to ``True``, the function can handle plotting single or multiple waveforms depending on the input signal's dimensionality. Plots can be saved to a specified directory.

    :param signals: Input signal(s) as a numpy array. Can be a single signal (1D array) or multiple signals (2D array where each row represents a signal).
    :type signals: np.ndarray
    :param sampling_rate: The sampling rate of the signal(s) in Hz.
    :type sampling_rate: int
    :param log_scale: If True, plots the Fourier Transform in logarithmic scale.
    :type log_scale: bool, optional
    :param plot: Whether to plot the Fourier Transform and the original signal(s).
    :type plot: bool, optional
    :param plot_waveform: If False, the waveform will not be plotted above the spectrum plot.
    :type plot_waveform: bool, optional
    :param max_plots: The maximum number of plots to generate when handling multiple signals. Default is 10.
    :type max_plots: int, optional
    :param save_figure: If True, saves the generated plots in a directory.
    :type save_figure: bool, optional
    :param save_name: Name under which the figure will be saved. Default ``'fourier_transform'``
    :type save_name: str, optional
    :param save_extension: Extension with which the image will be saved. Default ``'jpg'``
    :type save_extension: str, optional
    :return: The Fourier Transform of the input signal(s).
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
    # Compute the Fourier Transform
    ft = np.fft.fft(signals, axis=0 if signals.ndim == 1 else 1)
    freq = np.fft.fftfreq(signals.shape[-1], d=1/sampling_rate)

    if plot:
        # Handle single waveform case
        if signals.ndim == 1:
            if plot_waveform:
                fig = plt.figure(figsize=(10, 8))
                gs = gridspec.GridSpec(2, 1, height_ratios=[1, 2])
            else:
                fig = plt.figure(figsize=(10, 6))
                gs = gridspec.GridSpec(1, 1)

            if plot_waveform:
                ax1 = plt.subplot(gs[0])
                ax1.plot(signals, linewidth=0.75, color='k')
                ax1.set_title('Waveform', fontsize=14, fontweight='bold')
                ax1.set_xlabel('Samples', fontsize=12)
                ax1.set_ylabel('Amplitude', fontsize=12)
                ax1.set_xlim(0, len(signals))
                ax1.grid(True, alpha=0.25, axis='x', linestyle=':')

            ax2 = plt.subplot(gs[-1])
            if log_scale:
                ax2.loglog(freq[:len(freq)//2], np.abs(ft)[:len(freq)//2], color='black', linewidth=0.75)
            else:
                ax2.plot(freq[:len(freq)//2], np.abs(ft)[:len(freq)//2], color='black', linewidth=0.75)
            ax2.set_title('Fourier Transform', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Frequency [Hz]', fontsize=12)
            ax2.set_ylabel('Amplitude', fontsize=12)
            ax2.grid(True, alpha=0.25, which='both', linestyle=':')

            plt.tight_layout()
            if save_figure:
                os.makedirs('./seismutils_figures', exist_ok=True)
                fig_name = os.path.join('./seismutils_figures', f'{save_name}.{save_extension}')
                plt.savefig(fig_name, dpi=300, bbox_inches='tight', facecolor=None)
            plt.show()

        # Handle multiple waveforms case
        else:
            num_plots = min(signals.shape[0], max_plots)
            for i in range(num_plots):
                if plot_waveform:
                    fig = plt.figure(figsize=(10, 8))
                    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 2])
                else:
                    fig = plt.figure(figsize=(10, 6))
                    gs = gridspec.GridSpec(1, 1)

                if plot_waveform:
                    ax1 = plt.subplot(gs[0])
                    ax1.plot(signals[i, :], linewidth=0.75, color='k', label=f'Waveform {i+1}')
                    ax1.set_title(f'Waveform {i+1}', fontsize=14, fontweight='bold')
                    ax1.set_xlabel('Samples', fontsize=12)
                    ax1.set_ylabel('Amplitude', fontsize=12)
                    ax1.set_xlim(0, len(signals[i, :]))
                    ax1.grid(True, alpha=0.25, axis='x', linestyle=':')
                    ax1.legend(loc='upper right', frameon=False, fontsize=10)

                ax2 = plt.subplot(gs[-1])
                if log_scale:
                    ax2.loglog(freq[:len(freq)//2], np.abs(ft[i, :])[:len(freq)//2], color='black', linewidth=0.75, label='Spectrum')
                else:
                    ax2.plot(freq[:len(freq)//2], np.abs(ft[i, :])[:len(freq)//2], color='black', linewidth=0.75, label='Spectrum')
                ax2.set_title(f'Fourier Transform {i+1}', fontsize=14, fontweight='bold')
                ax2.set_xlabel('Frequency [Hz]', fontsize=12)
                ax2.set_ylabel('Amplitude', fontsize=12)
                ax2.grid(True, alpha=0.25, which='both', linestyle=':')
                ax2.legend(loc='upper right', frameon=False, fontsize=10)

                plt.tight_layout()
                if save_figure:
                    os.makedirs('./seismutils_figures', exist_ok=True)
                    fig_name = os.path.join('./seismutils_figures', f'{save_name}_{i+1}.{save_extension}')
                    plt.savefig(fig_name, dpi=300, bbox_inches='tight', facecolor=None)
                plt.show()

    return ft

def spectrogram(signals: np.ndarray, sampling_rate: int, nperseg: int=128, noverlap: float=None, log_scale: bool=False, zero_padding_factor: int=8, return_data: bool=False,
                plot_waveform: bool=True, max_plots: int=10, colorbar: bool=False, cmap: str='jet', save_figure: bool=False, save_name: str='spectrogram', save_extension: str='jpg'):
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
            spectrogram = np.sqrt(spectrogram)
            vmin, vmax = np.min(spectrogram), np.max(spectrogram)

        spectrogram_data.append((spectrogram, frequencies, times))

        # Plot configuration
        if plot_waveform:
            fig = plt.figure(figsize=(10, 8))
            gs = gridspec.GridSpec(2, 1, height_ratios=[1, 3], hspace=0.2)

            # Plot the waveform
            ax1 = fig.add_subplot(gs[0])
            ax1.plot(time, signal, color='k', linewidth=0.75)
            ax1.set_title('Waveform' if len(signals) == 1 else f'Waveform {i+1}', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Amplitude', fontsize=12)
            ax1.set_xlim(0, round(time[-1]))
            ax1.grid(True, alpha=0.25, axis='x', linestyle=':')
            ax1.set_xticklabels([])

            # Plot the spectrogram
            ax2 = fig.add_subplot(gs[1])
        else:
            fig, ax2 = plt.subplots(figsize=(10, 5))

        pcm = ax2.pcolormesh(times, frequencies, spectrogram, shading='gouraud', norm=Normalize(vmin, vmax), cmap=cmap)
        ax2.set_title('Spectrogram' if len(signals) == 1 else f'Spectrogram {i+1}', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Frequency [Hz]', fontsize=12)
        ax2.set_xlabel('Time [s]', fontsize=12)
        ax2.set_xlim(0, round(time[-1]))

        if colorbar:
            cbar = plt.colorbar(pcm, ax=ax2, pad=0.01)
            cbar.set_label('Log Amplitude [dB]' if log_scale else 'Amplitude', fontsize=12)
            cbar.ax.tick_params(labelsize=12)

        if save_figure:
            os.makedirs('./seismutils_figures', exist_ok=True)
            fig_name = os.path.join('./seismutils_figures', f'{save_name}_{i+1 if len(signals) > 1 else ""}.{save_extension}')
            plt.savefig(fig_name, dpi=300, bbox_inches='tight', facecolor=None)

        plt.show()
    
    if return_data:
        return spectrogram_data