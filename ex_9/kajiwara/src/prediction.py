from typing import Any
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from model import ConformerModel, GRUModel, ResNet, CRNN
from dataset import FSDDataset


def predict(model: Any, testloader: DataLoader, device: Any) -> np.ndarray:
    preds = []
    for data in testloader:
        data = data.to(device)
        outputs = model(data)
        _, pred = torch.max(outputs.data, 1)
        preds += list(pred.to('cpu').detach().numpy().copy())

    return np.array(preds)


def run(model_name: str, weight_path: str, testloader: DataLoader, device: Any) -> np.array:
    """load model"""
    if model_name == 'ConformerModel':
        model = ConformerModel().cuda()
    elif model_name == 'ResNet':
        model = ResNet('resnet18').cuda()
    elif model_name == 'CRNN':
        model = CRNN().cuda()
    else:
        model = GRUModel().cuda()

    model.load_state_dict(torch.load(weight_path))
    model.eval()

    """predict test data"""
    preds_list = []
    for i in range(5):

        preds_list.append(predict(model, testloader, device))

    pred = np.array(preds_list).sum(axis=0) // 5

    return pred


def plot_confusion_matrix(ts: str, result_path: Path, predict, ground_truth, title=None, cmap=plt.cm.Blues):
    cm = confusion_matrix(predict, ground_truth)
    plt.figure()
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.ylabel("Predicted")
    plt.xlabel("Ground truth")
    plt.savefig(result_path / 'pred.png')


def generate_confusion_matrix(ts: str, result_path: Path, pred_by_fold: np.ndarray, truth_path: str):
    test_truth = pd.read_csv(truth_path)

    pred = pred_by_fold.sum(axis=0) // 3

    truth_values = test_truth['label'].values
    title = f'{ts}, acc: {str(accuracy_score(truth_values, pred))}'
    plot_confusion_matrix(ts, result_path, pred, truth_values, title=title)


if __name__ == '__main__':
    testset = FSDDataset(
        audio_path='/work/dataset',
        metadata_path='/work/meta/test.csv',
        win_size_rate=0.025,
        overlap=0.5,
        n_mels=32,
        training=False,
        n_channels=1,
    )
    testloader = DataLoader(testset, batch_size=16, pin_memory=True)
    device = torch.device('cuda')
    ts = 20210627092226
    result_path = Path('./')
    truth_path = Path('/work/meta/test_truth.csv')

    preds_by_fold = []
    for i in range(3):
        preds_by_fold.append(
            run(
                'ConformerModel',
                '/work/results/20210627092226/fold0-best.pt',
                testloader, device
            )
        )

    preds_by_fold = np.array(preds_by_fold)
    generate_confusion_matrix(
        ts, result_path, preds_by_fold, truth_path
    )
