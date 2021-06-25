from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

import augmentations


@ hydra.main(config_path='../config', config_name='train')
def main(cfg):
    pathes = cfg['path']

    ex_audio_dir = Path(pathes['ex_audio'])
    base_meta_path = Path(pathes['meta'])
    ex_label_path = Path(pathes['ex_label'])

    base_df = pd.read_csv(base_meta_path)
    base_meta = base_df['path'].values
    base_label = base_df['label'].values

    """base audio"""
    _, _sr = sf.read(base_meta[0])
    base_wave_list = np.zeros((len(base_meta), int(_sr*1.0)))
    for i, p in tqdm(enumerate(base_meta)):
        p = Path(p)
        waveform, sr = sf.read(p)
        if len(waveform) <= 1.0*sr:
            waveform = np.append(waveform, np.array(
                [0] * (int(1.0*sr) - len(waveform))))
        else:
            waveform = waveform[:int(1.0*sr)]
        base_wave_list[i] = waveform
    np.save(ex_audio_dir/'base', base_wave_list)
    np.save(ex_label_path/'base', base_label)
    print('base', str(ex_audio_dir/'base'))

    """gaussian noise snr"""
    gns_wave_list = np.zeros((len(base_meta), int(_sr*1.0)))
    for i, wave in tqdm(enumerate(base_wave_list)):
        auged = augmentations.gaussian_noise_snr(wave)
        gns_wave_list[i] = auged
    np.save(ex_audio_dir/'gaussian_noise_snr', gns_wave_list)
    np.save(ex_label_path/'gaussian_noise_snr', base_label)
    print('gaussian noise snr', str(ex_audio_dir/'gaussian_noise_snr'))

    """gaussian noise snr"""
    ts_wave_list = np.zeros((len(base_meta), int(_sr*1.0)))
    for i, wave in tqdm(enumerate(base_wave_list)):
        auged = augmentations.time_shift(wave, _sr)
        ts_wave_list[i] = auged
    np.save(ex_audio_dir/'time_shift', ts_wave_list)
    np.save(ex_label_path/'time_shift', base_label)
    print('time shift', str(ex_audio_dir/'time_shift'))

    """volume control"""
    vc_wave_list = np.zeros((len(base_meta), int(_sr*1.0)))
    for i, wave in tqdm(enumerate(base_wave_list)):
        auged = augmentations.volume_control(wave)
        vc_wave_list[i] = auged
    np.save(ex_audio_dir/'volume_control', vc_wave_list)
    np.save(ex_label_path/'volume_control', base_label)
    print('volume control', str(ex_audio_dir/'volume_control'))

    """full augmentation"""
    full_aug_wave_list = np.zeros((len(base_meta), int(_sr*1.0)))
    for i, wave in tqdm(enumerate(base_wave_list)):
        gns = augmentations.gaussian_noise_snr(wave)
        ts = augmentations.time_shift(gns, _sr)
        vc = augmentations.volume_control(ts)

        full_aug_wave_list[i] = vc
    np.save(ex_audio_dir/'full_aug', full_aug_wave_list)
    np.save(ex_label_path/'full_aug', base_label)
    print('full augmentation', str(ex_audio_dir/'full_aug'))


if __name__ == '__main__':
    main()
