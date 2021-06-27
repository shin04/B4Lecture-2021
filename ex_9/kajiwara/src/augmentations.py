import numpy as np
# import sklearn
import torch


def add_gaussian_noise(input, max_noise_amplitude=0.1):
    noise_amplitude = np.random.uniform(0.0, max_noise_amplitude)
    noise = np.random.randn(len(input))
    augmented = (input + noise * noise_amplitude).astype(input.dtype)

    return augmented


def gaussian_noise_snr(input, min_snr=5.0, max_snr=20.0):
    snr = np.random.uniform(min_snr, max_snr)
    a_signal = np.sqrt(input ** 2).max()
    a_noise = a_signal / (10 ** (snr / 20))

    white_noise = np.random.randn(len(input))
    a_white = np.sqrt(white_noise ** 2).max()
    augmented = (input + white_noise * 1 / a_white * a_noise).astype(input.dtype)

    return augmented


def time_shift(input, sr, max_shift_second=0.2, padding_mode='replace'):
    shift = np.random.randint(-sr*max_shift_second, sr*max_shift_second)
    augmented = np.roll(input, shift)
    if padding_mode == "zero":
        if shift > 0:
            augmented[:shift] = 0
        else:
            augmented[shift:] = 0

    return augmented


def volume_control(input, db_lim=20, mode='sine'):
    """
    mode must be one of 'uniform', 'fade', 'cosine', 'sine'
    """

    db = np.random.uniform(-db_lim, db_lim)
    if mode == "uniform":
        db_translated = 10 ** (db / 20)
    elif mode == "fade":
        lin = np.arange(len(input))[::-1] / (len(input) - 1)
        db_translated = 10 ** (db * lin / 20)
    elif mode == "cosine":
        cosine = np.cos(np.arange(len(input)) / len(input) * np.pi * 2)
        db_translated = 10 ** (db * cosine / 20)
    else:
        sine = np.sin(np.arange(len(input)) / len(input) * np.pi * 2)
        db_translated = 10 ** (db * sine / 20)

    augmented = input * db_translated

    return augmented


def mixup(x, y, alpha=0.2, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''

    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam
