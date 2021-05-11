import argparse
from pathlib import Path
from datetime import datetime

import soundfile
import librosa

from modules.spectrogram import spectrogram
from modules.filters import *
from modules.utils import *

TIME_TEMPLATE = '%Y%m%d%H%M%S'


def main(args):
    data_path = Path(args.data_path)
    result_path = Path(args.save_path)

    assert data_path.exists(), '{} is not exist'.format(data_path)
    assert result_path.exists(), '{} is not exist'.format(result_path)

    # loda data
    wave_data, sr = librosa.load(data_path)
    spec = spectrogram(wave_data)

    # filtering
    l_cutoff, h_cutoff = 1000, 3000
    bpf = band_pass_filter(sr, h_cutoff, l_cutoff, 255)
    wave_data_filtered = conv1d(wave_data, bpf)
    spec_filtered = spectrogram(wave_data_filtered)

    # plot data
    timestamp = datetime.now().strftime(TIME_TEMPLATE)
    res_path = result_path.joinpath(timestamp+'-original-result.png')
    plot_wave_and_spec(wave_data, spec, sr, result_path=res_path)
    res_path = result_path.joinpath(timestamp+'-filtered-result.png')
    plot_wave_and_spec(wave_data_filtered, spec_filtered, sr,
                       result_path=res_path)
    res_path = result_path.joinpath(timestamp+'-result.png')
    plot_phase_and_frequency_characteristic(bpf, sr, result_path=res_path)

    # save wav
    soundfile.write(result_path.joinpath(
        timestamp+'-result.wav'), wave_data_filtered, sr)


if __name__ == "__main__":
    description = 'Example: python main.py ./sample.wav'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('data_path', help='path of data')
    parser.add_argument('-s', '--save_path', default='./',
                        help='path to save the result')

    args = parser.parse_args()

    main(args)
