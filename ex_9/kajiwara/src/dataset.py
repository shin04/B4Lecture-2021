from pathlib import Path

import numpy as np
import pandas as pd
import librosa
from torch.utils.data import Dataset
# import torchaudio


def mel_spec(input: np.ndarray, sr: int, win_size: int, hop_len: int, n_mels: int, ) -> np.ndarray:
    spec = librosa.stft(
        y=input, n_fft=win_size, win_length=win_size, hop_length=hop_len)
    spec = np.abs(spec) ** 2.0

    mel_filter_bank = librosa.filters.mel(sr=sr, n_fft=win_size, n_mels=n_mels)
    mel = np.dot(mel_filter_bank, spec)

    # mel = librosa.feature.melspectrogram(
    #     y=crop_data, sr=sr, n_mels=80, n_fft=win_size, win_length=win_size, hop_length=hop_len)
    # log_mel = np.log(mel)

    return mel


class FSDDataset(Dataset):
    def __init__(self, audio_path: str, metadata_path: str, training=True):
        self.training = training
        self.audio_path = Path(audio_path)

        df = pd.read_csv(Path(metadata_path))
        self.audio_names = df['path'].values
        if training:
            self.labels = df['label'].values

    def __len__(self):
        return len(self.audio_names)

    def __getitem__(self, idx):
        data_path = self.audio_names[idx]

        # waveform, _ = torchaudio.load(data_path)
        waveform, sr = librosa.load(data_path)
        if len(waveform) <= 1.0*sr:
            waveform = np.append(waveform, np.array(
                [0] * (int(1.0*sr) - len(waveform))))
        else:
            waveform = waveform[:int(1.0*sr)]
        win_size = int(sr*0.025)
        overlap = 0.4
        n_mels = 64
        feature = mel_spec(waveform, sr, win_size,
                           int(win_size*overlap), n_mels)

        if self.training:
            return np.float32(feature.T[np.newaxis, :, :]), self.labels[idx]
            # return np.float32(waveform), self.labels[idx]
        else:
            return np.float32(feature.T[np.newaxis, :, :])
            # return np.float32(waveform)


if __name__ == '__main__':
    dataset = FSDDataset(
        audio_path='/work/dataset',
        metadata_path='/work/training.csv',
    )

    print(len(dataset))
    print(dataset[0])
