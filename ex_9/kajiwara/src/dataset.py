from pathlib import Path

import numpy as np
import pandas as pd
import librosa
from torch.utils.data import Dataset

from omegaconf import DictConfig

import augmentations


def mel_spec(input: np.ndarray, sr: int, win_size: int, hop_len: int, n_mels: int, fmax: int = 4096) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=input, sr=sr, n_mels=n_mels, fmax=fmax, n_fft=win_size, win_length=win_size, hop_length=hop_len)
    log_mel = librosa.amplitude_to_db(mel)

    # reverse
    log_mel = log_mel[::-1] - np.zeros_like(log_mel)

    return log_mel


class FSDDataset(Dataset):
    def __init__(
        self, audio_path: str, metadata_path: str,
        win_size_rate: float, overlap: float, n_mels: int,
        aug_cfg: DictConfig = None, training: bool = True,
        n_channels: int = 1
    ):
        self.training = training

        if aug_cfg is not None:
            self.gns_cfg = aug_cfg['gaussian_noise']
            self.pn_cfg = aug_cfg['pink_noise']
            self.ps_cfg = aug_cfg['pitch_shift']
            self.ts_cfg = aug_cfg['time_shift']
            self.vc_cfg = aug_cfg['volume_control']
        else:
            self.gns_cfg = {'using': False}
            self.pn_cfg = {'using': False}
            self.ps_cfg = {'using': False}
            self.ts_cfg = {'using': False}
            self.vc_cfg = {'using': False}

        self.audio_path = Path(audio_path)

        self.win_size_rate = win_size_rate
        self.overlap = overlap
        self.n_mels = n_mels

        self.n_channels = n_channels

        df = pd.read_csv(Path(metadata_path))
        self.audio_names = df['path'].values
        if training:
            self.labels = df['label'].values

    def __len__(self):
        return len(self.audio_names)

    def __getitem__(self, idx):
        data_path = self.audio_names[idx]

        waveform, sr = librosa.load(data_path)
        if len(waveform) <= 1.0*sr:
            waveform = np.append(waveform, np.array(
                [0] * (int(1.0*sr) - len(waveform))))
        else:
            waveform = waveform[:int(1.0*sr)]

        if self.ps_cfg['using']:
            waveform = augmentations.pitch_shift(
                waveform, sr, self.ps_cfg['max_steps'])

        if self.ts_cfg['using']:
            waveform = augmentations.time_shift(
                waveform, sr, self.ts_cfg['max_shift_sec'], self.ts_cfg['padding_mode'])

        if self.vc_cfg['using']:
            waveform = augmentations.volume_control(
                waveform, self.vc_cfg['db_lim'], self.vc_cfg['mode'])

        if self.gns_cfg['using']:
            waveform = augmentations.gaussian_noise_snr(
                waveform, self.gns_cfg['min_snr'], self.gns_cfg['max_snr'])

        if self.pn_cfg['using']:
            waveform = augmentations.pink_noise_snr(
                waveform, self.pn_cfg['min_snr'], self.pn_cfg['max_snr'])

        win_size = int(self.win_size_rate * sr)
        feature = mel_spec(waveform, sr, win_size,
                           int(win_size*self.overlap), self.n_mels)

        if self.n_channels == 1:
            feature = np.float32(feature[np.newaxis, :, :])
        else:
            # feature = np.stack([feature, feature, feature])
            feature = mono_to_color(feature)
            feature = np.float32(feature)

        if self.training:
            return feature, self.labels[idx]
        else:
            return feature


class EXFSD_Dataset(Dataset):
    def __init__(
        self, data_path: str, label_path: str,
        win_size_rate: float, overlap: float, n_mels: int, training: bool = True,
    ):
        self.data = np.array([])
        for p in list(Path(data_path).glob('*.npy')):
            data = np.load(p)
            if len(self.data) == 0:
                self.data = data
            else:
                self.data = np.vstack([self.data, data])

        self.labels = np.array([])
        for p in list(Path(label_path).glob('*.npy')):
            label = np.load(p)
            if len(self.labels) == 0:
                self.labels = label
            else:
                self.labels = np.hstack([self.labels, label])

        self.training = training

        self.win_size_rate = win_size_rate
        self.overlap = overlap
        self.n_mels = n_mels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        waveform = self.data[idx]
        sr = 22050

        win_size = int(self.win_size_rate * sr)
        feature = mel_spec(waveform, sr, win_size,
                           int(win_size*self.overlap), self.n_mels)

        if self.training:
            return np.float32(feature[np.newaxis, :, :]), self.labels[idx]
        else:
            return np.float32(feature[np.newaxis, :, :])


def mono_to_color(input: np.ndarray, eps=1e-6):
    X = np.stack([input, input, input])

    X = X - X.mean()
    X_std = X / (X.std() + eps)
    norm_max = X_std.max()
    norm_min = X_std.min()
    if (norm_max - norm_min) > eps:
        V = X_std
        V[V < norm_min] = norm_min
        V[V > norm_max] = norm_max
        V = 255 * (V - norm_min) / (norm_max - norm_min)
        V = V.astype(np.uint8)
    else:
        V = np.zeros_like(X_std, dtype=np.uint8)
    return V


if __name__ == '__main__':
    dataset = FSDDataset(
        audio_path='/work/dataset',
        metadata_path='/work/meta/training.csv',
        win_size_rate=0.025,
        overlap=0.5,
        n_mels=32,
        training=True,
        n_channels=1,
    )

    print(len(dataset))
    print(dataset[0][0].shape)
    print(dataset[0][1])

    # ex_dataset = EXFSD_Dataset(
    #     data_path='/work/ex_dataset/audio',
    #     label_path='/work/ex_dataset/meta',
    #     win_size_rate=0.025,
    #     overlap=0.5,
    #     n_mels=32
    # )

    # print(len(ex_dataset))
    # print(ex_dataset[0][0].shape)
