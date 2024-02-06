import os
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec

from scipy.signal import butter, sosfilt, get_window, hilbert

def envelope(signals: np.ndarray, plot=False, envelope_type='positive'):
    '''
    Calculates and optionally plots the envelope of a given signal or multiple signals using the Hilbert transform.
    Allows selection between calculating the positive envelope, negative envelope, or both.

    :param np.ndarray signals: A single waveform (1D numpy array) or multiple waveforms (2D numpy array, each row represents a different waveform).
    :param bool plot: If True, plots the signal(s) along with their envelopes. Defaults to False.
    :param str envelope_type: Specifies the type of envelope to calculate: 'positive', 'negative', or 'both'. Defaults to 'positive'.
    :return: The envelope(s) of the signal(s) as a numpy array of the same shape as the input. If 'both' is selected, returns a tuple of two numpy arrays.
    :rtype: np.ndarray or tuple
    '''
    analytical_signal = hilbert(signals, axis=-1)
    positive_envelope = np.abs(analytical_signal)
    negative_envelope = -positive_envelope
    
    if plot:        
        for i, signal_to_plot in enumerate([signals] if signals.ndim == 1 else signals):
            plt.figure(figsize=(10, 4))
            plt.title('Envelope', fontsize=14, fontweight='bold')
            
            plt.plot(signal_to_plot, color='black', linewidth=0.75, label='Signal')
            
            if envelope_type in ['positive', 'both']:
                plt.plot(positive_envelope[i] if signals.ndim > 1 else positive_envelope, color='red', linewidth=0.75, label='Pos.envelope')
            
            if envelope_type in ['negative', 'both']:
                plt.plot(negative_envelope[i] if signals.ndim > 1 else negative_envelope, color='blue', linewidth=0.75, label='Neg. envelope')
        
            plt.xlabel('Sample', fontsize=12)
            plt.ylabel('Amplitude', fontsize=12)
            plt.xlim(0, len(signal_to_plot))
            plt.grid(True, alpha=0.25, axis='x', linestyle=':')
            plt.legend(loc='best', frameon=False, fontsize=12)
            plt.tight_layout()
            plt.show()
    
    if envelope_type == 'positive':
        return positive_envelope
    
    elif envelope_type == 'negative':
        return negative_envelope
    
    else: # 'both'
        return positive_envelope, negative_envelope

import numpy as np
from scipy.signal import butter, sosfilt, sosfiltfilt, get_window

def filter(signals: np.ndarray, sampling_rate: int, filter_type: str, cutoff: float, order=5, taper_window=None, taper_params=None, filter_mode='zero-phase'):
    '''
    Filter a signal with optional tapering and filtering modes, using specified filter parameters.
    
    :param np.ndarray signals: The input signal as a 1D numpy array or list, or a 2D array for multiple signals.
    :param int sampling_rate: The sampling frequency of the signal in Hz.
    :param str filter_type: The type of the filter ('lowpass', 'highpass', 'bandpass', 'bandstop').
    :param float cutoff: The cutoff frequency or frequencies. For 'lowpass' and 'highpass', this is a single value. For 'bandpass' and 'bandstop', this is a tuple of two values (low cutoff, high cutoff).
    :param int order: The order of the filter. Higher order means a steeper filter slope but can lead to instability or ringing. Defaults to 5.
    :param str taper_window: The type of the tapering window to apply before filtering. If None, no tapering is applied. Defaults to None.
    :param dict taper_params: A dictionary of parameters for the tapering window. Ignored if `taper_window` is None. Defaults to None.
    :param str filter_mode: The mode of filtering ('butterworth' for standard filtering or 'zero-phase' for zero-phase filtering). Defaults to 'zero-phase'.
    :return: The filtered signal as a 1D numpy array or a 2D array for multiple signals.
    :rtype: np.ndarray
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

def fourier_transform(signals: np.ndarray, sampling_rate: int, plot=True, log_scale=True, max_plots=10, save_figures=False):
    '''
    Performs a Fourier Transform on a set of signals and optionally plots the results.

    This function computes the Fourier Transform using numpy's FFT implementation and can generate plots for both individual and multiple signals. It supports plotting in logarithmic scale for better visualization of the frequency components.

    .. note::
        If `plot` is set to True, the function can handle plotting single or multiple waveforms depending on the input signal's dimensionality. Plots can be saved to a specified directory.

    :param signals: Input signal(s) as a numpy array. Can be a single signal (1D array) or multiple signals (2D array where each row represents a signal).
    :type signals: np.ndarray
    :param sampling_rate: The sampling rate of the signal(s) in Hz.
    :type sampling_rate: int
    :param plot: Whether to plot the Fourier Transform and the original signal(s).
    :type plot: bool, optional
    :param log_scale: If True, plots the Fourier Transform in logarithmic scale.
    :type log_scale: bool, optional
    :param max_plots: The maximum number of plots to generate when handling multiple signals. Default is 10.
    :type max_plots: int, optional
    :param save_figures: If True, saves the generated plots in a directory.
    :type save_figures: bool, optional
    :return: The Fourier Transform of the input signal(s).
    :rtype: np.ndarray

    **Parameter details**

    - ``plot``: Enable or disable the visualization of the signal(s) and their Fourier Transform. Useful for analysis and verification of signal properties.
    
    - ``log_scale``: Enhances the visibility of the spectrum's details, especially useful for signals with a wide range of frequency components.
    
    - ``save_figures``: Enables saving the generated figures for documentation or further analysis. Figures are saved in the ``'./seismutils_figures'`` directory.

    **Usage example**

    .. code-block:: python

        import seismutils.signal as sus
        import numpy as np

        # Assuming waveform is an np.ndarray containing amplitude values
        waveform = np.random.randn(1024)  # Example waveform
        fft = sus.fourier_transform(
              signals=waveform,
              sampling_rate=100,
              log_scale=True,
              plot=True,
        )

    .. image:: https://imgur.com/njtQn5s.png
       :align: center
       :target: spectral_analysis.html#seismutils.signal.fourier_transform

    The output is a numpy array containing the Fourier Transform of the input signal(s). If ``plot`` is ``True``, the function also generates a plot (or plots) illustrating the signal(s) and their frequency spectrum.
    '''
    # Compute the Fourier Transform
    ft = np.fft.fft(signals, axis=0 if signals.ndim == 1 else 1)
    freq = np.fft.fftfreq(signals.shape[-1], d=1/sampling_rate)

    # Ensure the save directory exists
    if save_figures:
        os.makedirs('./seismutils_figures', exist_ok=True)

    if plot:
        # Handle single waveform case
        if signals.ndim == 1:
            fig = plt.figure(figsize=(10, 8))
            gs = gridspec.GridSpec(2, 1, height_ratios=[1, 2])
            
            ax1 = plt.subplot(gs[0])
            ax1.plot(signals, linewidth=0.75, color='k')
            ax1.set_title('Waveform', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Samples', fontsize=12)
            ax1.set_ylabel('Amplitude', fontsize=12)
            ax1.set_xlim(0, len(signals))
            ax1.grid(True, alpha=0.25, axis='x', linestyle=':')
            
            ax2 = plt.subplot(gs[1])
            if log_scale:
                ax2.loglog(freq[:len(freq)//2], np.abs(ft)[:len(freq)//2], color='black', linewidth=0.75)
            else:
                ax2.plot(freq[:len(freq)//2], np.abs(ft)[:len(freq)//2], color='black', linewidth=0.75)
            ax2.set_title('Fourier Transform', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Frequency [Hz]', fontsize=12)
            ax2.set_ylabel('Amplitude', fontsize=12)
            ax2.grid(True, alpha=0.25, which='both', linestyle=':')
            
            plt.tight_layout()
            if save_figures:
                fig_name = os.path.join('./seismutils_figures', 'fourier_transform.png')
                plt.savefig(fig_name, dpi=300)
            plt.show()
            
        # Handle multiple waveforms case
        else:
            num_plots = min(signals.shape[0], max_plots)
            for i in range(num_plots):
                fig = plt.figure(figsize=(10, 8))
                gs = gridspec.GridSpec(2, 1, height_ratios=[1, 2])

                ax1 = plt.subplot(gs[0])
                ax1.plot(signals[i, :], linewidth=0.75, color='k', label=f'Waveform {i+1}')
                ax1.set_title(f'Waveform {i+1}', fontsize=14, fontweight='bold')
                ax1.set_xlabel('Samples', fontsize=12)
                ax1.set_ylabel('Amplitude', fontsize=12)
                ax1.set_xlim(0, len(signals[i, :]))
                ax1.grid(True, alpha=0.25, axis='x', linestyle=':')
                ax1.legend(loc='upper right', frameon=False, fontsize=10)
                
                ax2 = plt.subplot(gs[1])
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
                if save_figures:
                    fig_name = os.path.join('./seismutils_figures', f'fourier_transform{i+1}.png')
                    plt.savefig(fig_name, dpi=300)
                plt.show()

    return ft