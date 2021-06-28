import torch
import torch.nn as nn
from augmentations import mixup


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def bceloss(pred, target):
    m = nn.Sigmoid()
    cr = nn.BCELoss()

    return cr(m(pred), target)


def train(
        trainloader, optimizer, device, global_step,
        model, criterion, writer, fold, is_mixup, mixup_mode=None):
    model.train()

    n_batch = len(trainloader)
    train_loss = 0
    train_acc = 0
    total = 0
    for batch_num, (t_data, labels) in enumerate(trainloader):
        optimizer.zero_grad()
        t_data = t_data.to(device)
        labels = labels.to(device)

        if is_mixup:
            t_data, label_a, label_b, lam = mixup(t_data, labels)

        outputs = model(t_data)

        optimizer.zero_grad()

        if is_mixup and (mixup_mode == 'loss_mixup'):
            loss = mixup_criterion(criterion, outputs, label_a, label_b, lam)
        elif is_mixup and (mixup_mode == 'label_mixup'):
            loss = bceloss(outputs, labels)
        else:
            loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        _, predict = torch.max(outputs.data, 1)
        total += labels.size(0)

        if is_mixup:
            correct = lam * ((predict == label_a).sum()) + \
                (1-lam) * ((predict == label_b).sum())
        else:
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

    return global_step, train_loss, train_acc


def valid(validloader, device, model, criterion, is_mixup, mixup_mode=None):
    model.eval()

    valid_loss = 0
    valid_acc = 0
    total = 0

    with torch.no_grad():
        for i, (t_data, labels) in enumerate(validloader):
            t_data = t_data.to(device)
            labels = labels.to(device)

            outputs = model(t_data)

            # loss = criterion(outputs, labels)
            if is_mixup and (mixup_mode == 'label_mixup'):
                loss = bceloss(outputs, labels)
            else:
                loss = criterion(outputs, labels)

            valid_loss += loss.item()

            _, predict = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct = (predict == labels).sum()
            valid_acc += correct.item()

        valid_loss /= len(validloader)
        valid_acc /= total

    return valid_loss, valid_acc
