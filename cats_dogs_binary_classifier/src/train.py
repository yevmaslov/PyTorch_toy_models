import torch
import torch.nn as nn

import os

import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from src.models.dataset import CustomDataset
from src.models.simple_cnn import SimpleCnn

from sklearn.model_selection import train_test_split

import json

DATA_PATH = '../data'
TRAIN_PATH = os.path.join(DATA_PATH, 'train')
TEST_PATH = os.path.join(DATA_PATH, 'test')

IMG_SIZE = 224
BATCH_SIZE = 64

MODEL_PATH = '../models'


def train_fn(model, train_loader, criterion, optimizer, device):
    epoch_loss = 0
    epoch_acc = 0

    for data, label in train_loader:
        data = data.to(device)
        label = label.to(device)

        output = model(data)
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = ((output.argmax(dim=1) == label).float().mean())
        epoch_acc += acc / len(train_loader)
        epoch_loss += loss / len(train_loader)

    return epoch_acc, epoch_loss


def valid_fn(model, valid_loader, criterion, device):
    epoch_valid_acc = 0
    epoch_valid_loss = 0

    with torch.no_grad():
        for data, label in valid_loader:
            data = data.to(device)
            label = label.to(device)

            val_output = model(data)
            val_loss = criterion(val_output, label)

            acc = ((val_output.argmax(dim=1) == label).float().mean())
            epoch_valid_acc += acc / len(valid_loader)
            epoch_valid_loss += val_loss / len(valid_loader)

    return epoch_valid_acc, epoch_valid_loss


def get_transforms():
    train_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomResizedCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    valid_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])

    return train_transforms, valid_transforms


def train(model, train_loader, valid_loader, criterion, optimizer, device, model_save_path, n_epochs=10):
    epochs = n_epochs

    history = {
        'train_loss': [],
        'valid_loss': [],
        'train_acc': [],
        'valid_acc': []
               }
    for epoch in range(epochs):

        epoch_acc, epoch_loss = train_fn(model, train_loader, criterion, optimizer, device)
        print('Epoch : {}, train accuracy : {}, train loss : {}'.format(epoch + 1, epoch_acc, epoch_loss))

        epoch_valid_acc, epoch_valid_loss = valid_fn(model, valid_loader, criterion, device)
        print('Epoch : {}, val_accuracy : {}, val_loss : {}'.format(epoch + 1, epoch_valid_acc, epoch_valid_loss))

        history['train_loss'].append(epoch_loss.item())
        history['valid_loss'].append(epoch_valid_loss.item())
        history['train_acc'].append(epoch_acc.item())
        history['valid_acc'].append(epoch_valid_acc.item())

    model_weights_file = os.path.join(model_save_path, 'weights.pth')
    torch.save(model.state_dict(), model_weights_file)
    return history


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_img_list, valid_img_list = train_test_split(os.listdir(TRAIN_PATH),
                                                      test_size=0.2,
                                                      random_state=42,
                                                      shuffle=True)

    train_transforms, valid_transforms = get_transforms()

    train_ds = CustomDataset(TRAIN_PATH, train_img_list, transform=train_transforms)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    valid_ds = CustomDataset(TRAIN_PATH, valid_img_list, transform=valid_transforms)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=True)

    model = SimpleCnn().to(device)
    model_name = 'SimpleCnn'
    model_save_path = os.path.join(MODEL_PATH, model_name)

    if not os.path.isdir(model_save_path):
        os.mkdir(model_save_path)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=model.parameters(), lr=0.001)

    history = train(model, train_loader, valid_loader, criterion, optimizer, device, model_save_path)

    model_history_file = os.path.join(model_save_path, 'model_history.json')
    with open(model_history_file, "w") as outfile:
        json.dump(history, outfile)


if __name__ == '__main__':
    main()
