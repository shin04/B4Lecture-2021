import argparse
import time
from pathlib import Path
from datetime import datetime

import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score

from hmm import HMM

TIME_TEMPLATE = '%Y%m%d%H%M%S'


def main(args):
    if args.timestamp is None:
        timestamp = datetime.now().strftime(TIME_TEMPLATE)
    else:
        timestamp = args.timestamp

    result_path = Path(args.result_path)
    result_path = result_path/timestamp
    if not result_path.exists():
        result_path.mkdir(parents=True)

    data_path = Path(args.data)
    data = np.load(data_path, allow_pickle=True)

    outputs = np.array(data['output'])
    models = data['models']
    answer_models = np.array(data['answer_models'])

    PIs = np.array(models['PI'])
    As = np.array(models['A'])
    Bs = np.array(models['B'])

    n_models = PIs.shape[0]
    n_samples = outputs.shape[0]

    hmms = []
    for n in range(n_models):
        pi = np.transpose(PIs[n], axes=(1, 0))[0]
        A = As[n]
        B = Bs[n]
        hmms.append(HMM(pi, A, B))

    forward_prob_by_model = np.zeros((n_models, n_samples))
    viterbi_prob_by_model = np.zeros((n_models, n_samples))
    forward_pro_times = []
    viterbi_pro_times = []
    for i, hmm in enumerate(hmms):
        start = time.perf_counter()
        forward_prob = hmm.forward(outputs)
        forward_pro_times.append(time.perf_counter() - start)

        start = time.perf_counter()
        viterbi_prob = hmm.viterbi(outputs)
        viterbi_pro_times.append(time.perf_counter() - start)

        forward_prob_by_model[i] = forward_prob
        viterbi_prob_by_model[i] = viterbi_prob

    print(f'process time (forward): {sum(forward_pro_times)/len(forward_pro_times)}')
    print(f'process time (viterbi): {sum(viterbi_pro_times)/len(viterbi_pro_times)}')

    forward_pred = np.argmax(forward_prob_by_model, axis=0)
    viterbi_pred = np.argmax(viterbi_prob_by_model, axis=0)

    forward_acc = accuracy_score(answer_models, forward_pred)
    viterbi_acc = accuracy_score(answer_models, viterbi_pred)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f"{data_path.stem}", fontsize=20)
    forward_cm = confusion_matrix(answer_models, forward_pred)
    viterbi_cm = confusion_matrix(answer_models, viterbi_pred)
    sns.heatmap(forward_cm, square=True, cbar=True, annot=True, cmap='Blues', ax=ax1)
    ax1.set_title(f"{data_path.stem} forward\n acc: {forward_acc}", fontsize=15)
    ax1.set_xlabel("pred", fontsize=15)
    ax1.set_ylabel("answer", fontsize=15)
    sns.heatmap(viterbi_cm, square=True, cbar=True, annot=True, cmap='Blues', ax=ax2)
    ax2.set_title(f"{data_path.stem} viterbi\n acc: {viterbi_acc}", fontsize=15)
    ax2.set_xlabel("pred", fontsize=15)
    ax2.set_ylabel("answer", fontsize=15)
    plt.savefig(result_path / f'{data_path.stem}.png')
    plt.close()
    plt.clf()


if __name__ == "__main__":
    description = 'Example: python main.py data.pickle -rs ./result'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('data', help='path to data (pickle format only)')
    parser.add_argument('-rs', '--result-path', default='./result', help='path to save the result')
    parser.add_argument('-ts', '--timestamp', default=None, help='timestamp')

    args = parser.parse_args()

    main(args)
