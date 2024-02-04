import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import butter, sosfilt, get_window, hilbert

def envelope(signals: np.ndarray, plot=False):
    '''
    Calculates and optionally plots the envelope of a given signal or multiple signals.

    :param np.ndarray signals: A single waveform (1D numpy array) or multiple waveforms (2D numpy array, each row represents a different waveform).
    :param bool plot: If True, plots the signal(s) along with their envelopes. Defaults to False.
    :return: The envelope of the signal(s) as a numpy array of the same shape as the input.
    :rtype: np.ndarray

    This function computes the envelope of the signal(s) using the Hilbert transform. If plot is True, it also plots the original signal(s) with their envelopes. For multiple signals, each is plotted in a separate subplot.
    '''
    envelopes = np.abs(hilbert(signals, axis=-1))
    
    if plot:
        plt.figure(figsize=(10, 4))
        plt.title('Envelope', fontsize=14, fontweight='bold')

        # Determine if signals is a single signal or multiple, and select appropriately
        signal_to_plot = signals if signals.ndim == 1 else signals[0]
        envelope_to_plot = envelopes if signals.ndim == 1 else envelopes[0]
        
        # Plotting signal with envelope
        plt.plot(signal_to_plot, color='black', linewidth=0.75, label='Signal')
        plt.plot(envelope_to_plot, color='red', linewidth=0.75, linestyle='--', label='Envelope')
        
        plt.xlabel('Sample', fontsize=12)
        plt.ylabel('Amplitude', fontsize=12)
        plt.xlim(0, len(signal_to_plot))

        plt.grid(True, alpha=0.25, axis='x', linestyle=':')
        plt.legend(loc='best', frameon=False, fontsize=12)
        plt.tight_layout()
        plt.show()
    
    return envelopes

def filter(signals: np.ndarray, sampling_rate: int, type: str, cutoff: float, order=5, taper_window=None, taper_params=None):
    '''
    Filter a signal with optional tapering, using specified filter parameters.
    
    :param np.ndarray signal: The input signal as a 1D numpy array or list.
    :param int sampling_rate: The sampling frequency of the signal in Hz.
    :param str type: The type of the filter ('lowpass', 'highpass', 'bandpass', 'bandstop'). Defaults to 'lowpass'.
    :param float cutoff: The cutoff frequency or frequencies. For 'lowpass' and 'highpass', this is a single value. For 'bandpass' and 'bandstop', this is a tuple of two values (low cutoff, high cutoff).
    :param int order: The order of the filter. Higher order means a steeper filter slope but can lead to instability or ringing. Defaults to 5.
    :param str taper_window: The type of the tapering window ('hamming', 'hanning', 'blackman', 'tukey', etc.) to apply before filtering. If None, no tapering is applied. Defaults to None.
    :param dict taper_params: A dictionary of parameters for the tapering window, such as {'alpha': 0.5} for the Tukey window. Ignored if `taper_window` is None. Defaults to None.
    :return: The filtered signal as a 1D numpy array.
    :rtype: np.ndarray
    
    The function applies a digital filter to the input signal. The filter type can be one of 'lowpass', 'highpass', 'bandpass', or 'bandstop'. Before filtering, an optional tapering window can be applied to the signal to reduce edge effects. The tapering window type and parameters can be customized.
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

def fourier_transform(signals: np.ndarray, sampling_rate: int, plot=True, log_scale=True):
    '''
    Computes the Fourier Transform of the given signal(s) and optionally plots the spectra.

    :param np.ndarray signal: The input signal as a single waveform (1D numpy array) or multiple waveforms (2D numpy array where each row represents a different waveform).
    :param int sampling_rate: The sampling rate of the signal(s) in Hz. Defaults to 1.0.
    :param bool plot: If True, plots the spectra of the waveform(s). Defaults to False.
    :param bool log_scale: If True and plot is True, plots the frequency axis on a logarithmic scale. Defaults to False.
    :return: The Fourier Transform of the signal(s) as a numpy array.
    :rtype: np.ndarray

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