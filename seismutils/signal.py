import numpy as np
import matplotlib.pyplot as plt

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

def filter_signal(signals: np.ndarray, sampling_rate: int, filter_type: str, cutoff: float, order=5, taper_window=None, taper_params=None, filter_mode='butterworth'):
    '''
    Filter a signal with optional tapering and zero-phase shift filtering, using specified filter parameters.
    
    :param np.ndarray signals: The input signal as a 1D numpy array or list, or a 2D array for multiple signals.
    :param int sampling_rate: The sampling frequency of the signal in Hz.
    :param str filter_type: The type of the filter ('lowpass', 'highpass', 'bandpass', 'bandstop').
    :param float cutoff: The cutoff frequency or frequencies. For 'lowpass' and 'highpass', this is a single value. For 'bandpass' and 'bandstop', this is a tuple of two values (low cutoff, high cutoff).
    :param int order: The order of the filter. Higher order means a steeper filter slope but can lead to instability or ringing. Defaults to 5.
    :param str taper_window: The type of the tapering window to apply before filtering. If None, no tapering is applied. Defaults to None.
    :param dict taper_params: A dictionary of parameters for the tapering window. Ignored if `taper_window` is None. Defaults to None.
    :param str filter_mode: The mode of filtering ('butterworth' for standard filtering or 'zero_phase' for zero-phase filtering). Defaults to 'butterworth'.
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

def fourier_transform(signals: np.ndarray, sampling_rate: int, plot=True, log_scale=True):
    '''
    Computes the Fourier Transform of the given signal(s) and optionally plots the spectra using the Fast Fourier Transform (FFT) algorithm.

    :param np.ndarray signals: The input signal as a single waveform (1D numpy array) or multiple waveforms (2D numpy array where each row represents a different waveform).
    :param int sampling_rate: The sampling rate of the signal(s) in Hz.
    :param bool plot: If True, plots the spectra of the waveform(s). Defaults to True.
    :param bool log_scale: If True and plot is True, plots the frequency axis on a logarithmic scale. Defaults to True.
    :return: The Fourier Transform of the signal(s) as a numpy array.
    :rtype: np.ndarray

    It can process a single waveform or multiple waveforms simultaneously. When processing multiple waveforms, each waveform should be represented as a row in a 2D numpy array.
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