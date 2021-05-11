import math

import numpy as np
import librosa.display
import matplotlib.pyplot as plt


def conv1d(x, y):
    '''
    conv1d calculate convolution of x and y (1d array)
    x * y

    Parameters
    ----------
    x: 1d array
    y: 1d array

    Returns
    -------
    conv: ndarray
    '''

    length = len(x) + len(y) - 1

    x_spec = np.fft.rfft(x, n=length)
    y_spec = np.fft.rfft(y, n=length)
    conv_spec = x_spec * y_spec
    conv = np.fft.irfft(conv_spec)

    return conv


def plot_phase_and_frequency_characteristic(data, sr, is_show=False, is_save=True, result_path=None):
    '''
    plotting phase characteristic and frequency characteristic

    Parameters
    ----------
    data: array
    '''

    if is_save:
        assert (result_path != None)

    fig, ax = plt.subplots(nrows=2)
    plt.subplots_adjust(wspace=0.4, hspace=0.6)

    freq = np.fft.rfft(data, sr)
    amp = np.abs(freq)
    phase = np.unwrap(np.angle(freq))

    ax[0].plot(amp)
    ax[0].set(title="amplitude characteristic",
              xlabel="Frequency[Hz]", ylabel="Amplitude")

    ax[1].plot(phase)
    ax[1].set(title="phase characteristic",
              xlabel="Frequency[Hz]", ylabel="Phase[rad]")

    # save result
    if is_save:
        plt.savefig(result_path)

    # show
    if is_show:
        plt.show()


def plot_wave_and_spec(wave_data, spec_data, sr, is_show=False, is_save=True, result_path=None):
    '''
    plotting wave data and spectrogram

    Parameters
    ----------
    wave_data: 1d array
        wave data
    spec_data: 2d array
        spectrogram data
    sr: int
        sampling rate
    is_show: boolean
        whether show plot
    is_save: boolean
        whether save plot image
    result_path: pathlib.Path
        path to save result data
    '''

    if is_save:
        assert (result_path != None)

    fig, ax = plt.subplots(nrows=2)

    librosa.display.waveplot(wave_data, sr=sr, x_axis='time', ax=ax[0])
    ax[0].set(title='Original wave', xlabel="Time [s]", ylabel="Magnitude")
    ax[0].label_outer()

    spec_img_1 = librosa.display.specshow(
        spec_data, sr=sr, x_axis='time', y_axis='log', ax=ax[1])
    fig.colorbar(spec_img_1, ax=ax[1])
    ax[1].set(title='Spectrum', xlabel="Time [s]", ylabel="Frequency [Hz]")
    ax[1].yaxis.set_ticks([0, 128, 512, 2048, 8192])
    ax[1].label_outer()

    ax_pos_0 = ax[0].get_position()
    ax_pos_1 = ax[1].get_position()
    ax[0].set_position(
        [ax_pos_0.x0, ax_pos_0.y0, ax_pos_1.width, ax_pos_1.height])

    # save result
    if is_save:
        plt.savefig(result_path)

    # show
    if is_show:
        plt.show()
