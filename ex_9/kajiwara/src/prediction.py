from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from model import ConformerModel, GRUModel
from dataset import FSDDataset


def predict(model: Any, testloader: DataLoader, device: Any) -> np.ndarray:
    preds = []
    for data in testloader:
        data = data.to(device)
        outputs = model(data)
        _, pred = torch.max(outputs.data, 1)
        preds += list(pred.to('cpu').detach().numpy().copy())

    return np.array(preds)


def run(model_name: str, weight_path: str, testset: Dataset, device: Any) -> np.array:
    """load model"""
    if model_name == 'GRUModel':
        model = GRUModel().cuda()
    else:
        model = ConformerModel().cuda()
    model.load_state_dict(torch.load(weight_path))
    model.eval()

    """predict test data"""
    testloader = DataLoader(testset, batch_size=16, pin_memory=True)
    preds_list = []
    for i in range(5):
        preds_list.append(predict(model, testloader, device))

    predict = np.array(preds_list).sum(axis=0) // 5

    return predict


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
    device = torch.device('cuda')
    predict = run(
        'Conformer', '/work/results/20210627081148/fold0-best.pt', testset, device)
    print(predictt)
