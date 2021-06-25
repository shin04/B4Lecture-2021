from pathlib import Path

import numpy as np
import pandas as pd
import librosa
from torch.utils.data import Dataset

import augmentations


def mel_spec(input: np.ndarray, sr: int, win_size: int, hop_len: int, n_mels: int, ) -> np.ndarray:
    # spec = librosa.stft(
    #     y=input, n_fft=win_size, win_length=win_size, hop_length=hop_len)
    # spec = np.abs(spec) ** 2.0

    # mel_filter_bank = librosa.filters.mel(sr=sr, n_fft=win_size, n_mels=n_mels)
    # mel = np.dot(mel_filter_bank, spec)

    mel = librosa.feature.melspectrogram(
        y=input, sr=sr, n_mels=n_mels, n_fft=win_size, win_length=win_size, hop_length=hop_len)
    log_mel = librosa.amplitude_to_db(mel)

    return log_mel


class FSDDataset(Dataset):
    def __init__(
        self, audio_path: str, metadata_path: str,
        win_size_rate: float, overlap: float, n_mels: int, training: bool = True,
        gaussian_noise_snr: bool = True,
        time_shift: bool = True,
        volume_control: bool = True
    ):
        self.training = training
        self.gaussian_noise_snr = gaussian_noise_snr
        self.time_shift = time_shift
        self.volume_control = volume_control

        self.audio_path = Path(audio_path)

        self.win_size_rate = win_size_rate
        self.overlap = overlap
        self.n_mels = n_mels

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

        if self.gaussian_noise_snr:
            augmentations.gaussian_noise_snr(waveform)

        if self.time_shift:
            augmentations.time_shift(waveform, sr)

        if self.volume_control:
            augmentations.volume_control(waveform)

        win_size = int(self.win_size_rate * sr)
        feature = mel_spec(waveform, sr, win_size,
                           int(win_size*self.overlap), self.n_mels)

        if self.training:
            return np.float32(feature[np.newaxis, :, :]), self.labels[idx]
        else:
            return np.float32(feature[np.newaxis, :, :])


class EXFSD_Dataset(Dataset):
    def __init__(
        self, data_path: str, label_path: str,
        win_size_rate: float, overlap: float, n_mels: int, training: bool = True,
    ):
        self.data = np.array([])
        for p in list(Path(data_path).glob('**/*.npy')):
            data = np.load(p)
            if len(self.data) == 0:
                self.data = data
            else:
                self.data = np.vstack([self.data, data])

        self.labels = np.array([])
        for p in list(Path(label_path).glob('**/*.npy')):
            label = np.load(p)
            if len(self.labels) == 0:
                self.labels = label
            else:
                self.labels = np.hstack([self.labels, label])

        print(self.data.shape)
        print(self.labels.shape)

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


if __name__ == '__main__':
    dataset = FSDDataset(
        audio_path='/work/dataset',
        metadata_path='/work/meta/training.csv',
        win_size_rate=0.025,
        overlap=0.5,
        n_mels=32
    )

    print(len(dataset))
    print(dataset[0][0].shape)

    ex_dataset = EXFSD_Dataset(
        data_path='/work/ex_dataset/audio',
        label_path='/work/ex_dataset/meta',
        win_size_rate=0.025,
        overlap=0.5,
        n_mels=32
    )

    print(len(ex_dataset))
    print(ex_dataset[0][0].shape)
