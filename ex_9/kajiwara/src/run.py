from datetime import datetime
from pathlib import Path

import hydra
import numpy as np
from sklearn.model_selection import KFold

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from dataset import FSDDataset
from model import BaseModel

TIME_TEMPLATE = '%Y%m%d%H%M%S'


def train(trainloader, optimizer, device, global_step,  model, criterion, writer, fold):
    model.train()

    n_batch = len(trainloader)
    train_loss = 0
    train_acc = 0
    total = 0
    for batch_num, (t_data, labels) in enumerate(trainloader):
        optimizer.zero_grad()
        t_data = t_data.to(device)
        labels = labels.to(device)

        outputs = model(t_data)

        optimizer.zero_grad()

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        _, predict = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct = (predict == labels).sum()
        train_acc += correct.item()

        writer.add_scalar(f"{fold}/loss", loss.item(), global_step)
        writer.add_scalar(f"{fold}/acc", correct.item() /
                          labels.size(0), global_step)
        print(
            f'batch: {batch_num}/{n_batch}, '
            f'loss: {loss.item()}, train loss: {train_loss/(batch_num+1)}, '
            f'acc: {correct.item()/labels.size(0)}, train acc: {train_acc/total} ')
        global_step += 1

    train_loss /= n_batch
    train_acc /= len(trainloader.dataset)

    return model, global_step, train_loss, train_acc


def valid(validloader, device, model, criterion):
    model.eval()

    valid_loss = 0
    valid_acc = 0
    total = 0

    with torch.no_grad():
        for i, (t_data, labels) in enumerate(validloader):
            t_data = t_data.to(device)
            labels = labels.to(device)

            outputs = model(t_data)

            loss = criterion(outputs, labels)
            valid_loss += loss.item()

            _, predict = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct = (predict == labels).sum()
            valid_acc += correct.item()

        valid_loss /= len(validloader)
        valid_acc /= total

    print(f'val loss: {valid_loss}, val acc: {valid_acc}')


@ hydra.main(config_path='../config', config_name='train')
def run(cfg):
    """set config"""
    path_cfg = cfg['path']
    # preprocess_cfg = cfg['preprocess']
    train_cfg = cfg['training']

    """set path"""
    ts = datetime.now().strftime(TIME_TEMPLATE)
    print(f"TIMESTAMP: {ts}")

    audio_path = Path(path_cfg['audio'])
    meta_path = Path(path_cfg['meta'])
    test_metadata_path = Path(path_cfg['test_meta'])

    log_path = Path(path_cfg['tensorboard']) / ts
    if not log_path.exists():
        log_path.mkdir(parents=True)

    print('PATH')
    print(f'audio: {audio_path}')
    print(f'meta: {meta_path}')
    print(f'tensorboard: {log_path}')

    """set parameters"""
    device = torch.device(cfg['device'])
    n_epoch = train_cfg['n_epoch']
    batch_size = train_cfg['batch_size']
    lr = train_cfg['lr']

    writer = SummaryWriter(log_dir=log_path)

    print('PARAMETERS')
    print(f'device: {device}')
    print(f'n_epoch: {n_epoch}')
    print(f'batch_size: {batch_size}')
    print(f'lr: {lr}')

    """training and validation"""
    dataset = FSDDataset(
        audio_path=audio_path,
        metadata_path=meta_path,
    )
    idxes = [i for i in range(len(dataset))]
    kf = KFold(n_splits=3)
    preds_by_fold = []

    for k_fold, (tr_idx, val_idx) in enumerate(kf.split(idxes)):
        print(f'===== fold: {k_fold}')

        """prepare dataset"""
        trainset = Subset(dataset, tr_idx)
        trainloader = DataLoader(
            trainset, batch_size=batch_size, shuffle=True, pin_memory=True)
        validset = Subset(dataset, val_idx)
        validloader = DataLoader(
            validset, batch_size=batch_size, shuffle=True, pin_memory=True)

        """prepare model"""
        model = BaseModel().cuda()

        """prepare optimizer and loss function"""
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        """training and validation"""
        train_global_step = 0
        for epoch in range(n_epoch):
            model, train_global_step, train_loss, train_acc = train(
                trainloader, optimizer, device, train_global_step, model, criterion, writer, k_fold)
            valid(validloader, device, model, criterion)

            writer.add_scalar(f"{k_fold}/loss/epoch", train_loss, epoch)
            writer.add_scalar(f"{k_fold}/acc/epoch", train_acc, epoch)

            print(
                f'epoch: {epoch}/{n_epoch}, train loss: {train_loss}, train acc: {train_acc}')

        """prediction"""
        model.eval()
        testset = FSDDataset(
            audio_path=audio_path,
            metadata_path=test_metadata_path,
            training=False
        )
        testloader = DataLoader(
            testset, batch_size=16, pin_memory=True)
        preds = []
        for data in testloader:
            data = data.to(device)
            outputs = model(data)
            _, pred = torch.max(outputs.data, 1)
            preds += list(pred.to('cpu').detach().numpy().copy())
        preds_by_fold.append(np.array(preds))

    preds_by_fold = np.array(preds_by_fold)
    np.save('./predict', preds_by_fold)

    writer.close()


if __name__ == '__main__':
    run()
