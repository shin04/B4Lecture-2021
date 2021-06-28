from datetime import datetime
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from dataset import FSDDataset
from model import ConformerModel, GRUModel, ResNet, CRNN
from traininig import train, valid
import prediction


TIME_TEMPLATE = '%Y%m%d%H%M%S'


@ hydra.main(config_path='../config', config_name='param')
def run(cfg):
    """set config"""
    path_cfg = cfg['path']
    audio_cfg = cfg['audio']
    aug_cfg = cfg['augmentation']
    train_cfg = cfg['training']

    """set path"""
    ts = datetime.now().strftime(TIME_TEMPLATE)
    print(f"TIMESTAMP: {ts}")

    audio_path = Path(path_cfg['audio'])
    meta_path = Path(path_cfg['meta'])
    test_metadata_path = Path(path_cfg['test_meta'])
    ex_audio_path = Path(path_cfg['ex_audio'])
    ex_label_path = Path(path_cfg['ex_label'])
    truth_path = Path(path_cfg['truth'])

    log_path = Path(path_cfg['tensorboard']) / ts
    if not log_path.exists():
        log_path.mkdir(parents=True)

    result_path = Path(path_cfg['result']) / ts
    if not result_path.exists():
        result_path.mkdir(parents=True)

    print('PATH')
    print(f'audio: {audio_path}')
    print(f'meta: {meta_path}')
    print(f'test meta: {test_metadata_path}')
    print(f'ex_data: {ex_audio_path}')
    print(f'ex_label: {ex_label_path}')
    print(f'tensorboard: {log_path}')
    print(f'result: {result_path}')
    print(f'truth_path: {truth_path}')

    """set parameters"""
    device = torch.device(cfg['device'])

    data_type = train_cfg['data']
    model_name = train_cfg['model']
    kfold = train_cfg['kfold']
    n_epoch = train_cfg['n_epoch']
    batch_size = train_cfg['batch_size']
    lr = train_cfg['lr']

    win_size_rate = audio_cfg['win_size_rate']
    overlap = audio_cfg['overlap']
    n_mels = audio_cfg['n_mels']
    n_channels = audio_cfg['n_channels']

    writer = SummaryWriter(log_dir=log_path)

    meta_df = pd.read_csv(Path(meta_path))
    num_data = len(meta_df)

    is_mixup = aug_cfg['mixup']['using']
    mixup_mode = aug_cfg['mixup']['mode']

    print('PARAMETERS')
    print(f'device: {device}')
    print(f'data: {data_type}')
    print(f'model: {model_name}')
    print(f'kfold: {kfold}')
    print(f'n_epoch: {n_epoch}')
    print(f'batch_size: {batch_size}')
    print(f'lr: {lr}')
    print(f'win_size_rate: {win_size_rate}')
    print(f'overlap: {overlap}')
    print(f'n_mels: {n_mels}')
    print(f'n_channels: {n_channels}')
    print(f'mixup: {is_mixup}')

    """training and validation"""
    trainset = FSDDataset(
        audio_path=audio_path,
        metadata_path=meta_path,
        win_size_rate=win_size_rate,
        overlap=overlap,
        n_mels=n_mels,
        aug_cfg=aug_cfg,
        training=True,
        n_channels=n_channels,
    )

    validset = FSDDataset(
        audio_path=audio_path,
        metadata_path=meta_path,
        win_size_rate=win_size_rate,
        overlap=overlap,
        n_mels=n_mels,
        training=True,
        n_channels=n_channels,
    )

    testset = FSDDataset(
        audio_path=audio_path,
        metadata_path=test_metadata_path,
        win_size_rate=win_size_rate,
        overlap=overlap,
        n_mels=n_mels,
        training=False,
        n_channels=n_channels,
    )

    idxes = [i for i in range(num_data)]
    kf = KFold(n_splits=kfold, shuffle=True)
    preds_by_fold = []

    for k_fold, (tr_idx, val_idx) in enumerate(kf.split(idxes)):
        print('='*10)
        print(f'===== fold: {k_fold}')

        # path to save weight
        weight_path = result_path / f'fold{k_fold}-best.pt'

        """prepare dataset"""
        train_subset = Subset(trainset, tr_idx)
        trainloader = DataLoader(
            train_subset, batch_size=batch_size, shuffle=True, pin_memory=True)
        valid_subset = Subset(validset, val_idx)
        validloader = DataLoader(
            valid_subset, batch_size=batch_size, shuffle=True, pin_memory=True)

        """prepare model"""
        if model_name == 'ConformerModel':
            model = ConformerModel().cuda()
        elif 'resnet' in model_name:
            model = ResNet(model_name).cuda()
        elif model_name == 'CRNN':
            model = CRNN().cuda()
        else:
            model = GRUModel().cuda()

        """prepare optimizer and loss function"""
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        """training and validation"""
        best_loss = 10000
        train_global_step = 0
        for epoch in range(n_epoch):
            print(f'===== epoch: {epoch}')

            train_global_step, train_loss, train_acc = train(
                trainloader, optimizer, device, train_global_step, model,
                criterion, writer, k_fold, is_mixup, mixup_mode)
            valid_loss, valid_acc = valid(
                validloader, device, model, criterion, is_mixup, mixup_mode)

            writer.add_scalar(f"{k_fold}/train/loss/epoch", train_loss, epoch)
            writer.add_scalar(f"{k_fold}/train/acc/epoch", train_acc, epoch)

            writer.add_scalar(f"{k_fold}/valid/loss/epoch", valid_loss, epoch)
            writer.add_scalar(f"{k_fold}/valid/acc/epoch", valid_acc, epoch)

            print(
                f'epoch: {epoch}/{n_epoch}, train loss: {train_loss}, train acc: {train_acc}')
            print(
                f'epoch: {epoch}/{n_epoch}, val loss: {valid_loss}, val acc: {valid_acc}')

            if best_loss > train_loss:
                best_loss = train_loss
                with open(weight_path, 'wb') as f:
                    torch.save(model.state_dict(), f)

        """prediction"""
        model.eval()
        testloader = DataLoader(testset, batch_size=16, pin_memory=True)
        pred = prediction.run(model_name, weight_path, testloader, device)
        preds_by_fold.append(pred)

    """render confusion matrix and save pred"""
    preds_by_fold = np.array(preds_by_fold)
    prediction.generate_confusion_matrix(
        ts, result_path, preds_by_fold, truth_path
    )
    np.save(result_path/'pred', preds_by_fold)

    writer.close()

    print('complete!!')


if __name__ == '__main__':
    run()
