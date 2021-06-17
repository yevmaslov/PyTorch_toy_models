import torch
import torch.nn as nn
from torch import optim
from torchtext.vocab import Vocab
import pandas as pd
from collections import Counter
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from src.models.LSTM.model import RNN, LRScheduler, EarlyStopping
from pickle import dump
import json


class CustomVocabulary:
    def __init__(self, df_path):
        self.df = pd.read_csv(df_path).dropna(subset=['comment_preprocessed_str'])
        self.vocab = self.get_vocabulary()

    def get_vocabulary(self):
        counter = Counter()
        for text in self.df.comment_preprocessed_str.values:
            for word in text.split(' '):
                counter[word] += 1

        vocab = Vocab(counter, max_size=10000, specials=['<unk>', '<pad>', '<sos>', '<eos>'])
        return vocab

    def sentence_to_indices(self, sentence):
        indices = [self.vocab.stoi[w] for w in sentence.split(' ')]
        return indices


class CustomDataset(Dataset):
    def __init__(self, df_path, vocab):
        self.df = pd.read_csv(df_path).dropna(subset=['comment_preprocessed_str'])
        self.vocab = vocab

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        x = self.df.iloc[index].comment_preprocessed_str
        y = self.df.iloc[index].toxic

        x_ind = self.vocab.sentence_to_indices(x)

        x_out = [self.vocab.vocab.stoi['<sos>']]
        x_out += x_ind
        x_out += [self.vocab.vocab.stoi['<eos>']]

        return torch.tensor(x_out), y


class CustomCollate:
    def __init__(self, pad_idx, max_len=None):
        self.pad_idx = pad_idx
        self.max_len = max_len

    def __call__(self, batch):
        text = [item[0] for item in batch]
        if self.max_len:
            text = [item[:self.max_len] for item in text]
        text = pad_sequence(text, batch_first=False, padding_value=self.pad_idx)
        target = [item[1] for item in batch]
        return text, torch.tensor(target)


def get_dataloader(df_path, vocab, batch_size=64, shuffle=True):

    train_dataset = CustomDataset(df_path, vocab)
    pad_idx = train_dataset.vocab.vocab.stoi['<pad>']

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=CustomCollate(pad_idx=pad_idx)
    )

    return train_loader, train_dataset


def binary_accuracy(predictions, y):
    rounded_predictions = torch.round(torch.sigmoid(predictions))
    correct = (rounded_predictions == y).float()
    acc = correct.sum() / len(correct)
    return acc


def train_fn(model, data_loader, criterion, optimizer, device):
    epoch_loss = 0
    epoch_acc = 0

    model.train()
    for batch_idx, (data, target) in enumerate(data_loader):
        data = data.to(device)
        target = target.to(device)
        text_lengths = torch.tensor([data.size(0)] * data.size(1))

        optimizer.zero_grad()

        predictions = model(data, text_lengths)
        predictions = predictions.squeeze(1)

        target = target.type_as(predictions)

        loss = criterion(predictions, target)

        loss.backward()
        optimizer.step()

        acc = binary_accuracy(predictions, target)
        epoch_acc += acc.item()
        epoch_loss += loss.item()

    epoch_loss = epoch_loss / len(data_loader)
    epoch_acc = epoch_acc / len(data_loader)

    return epoch_loss, epoch_acc


def valid_fn(model, data_loader, criterion, device):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            data = data.to(device)
            target = target.to(device)

            text_lengths = torch.tensor([data.size(0)] * data.size(1))
            predictions = model(data, text_lengths)
            predictions = predictions.squeeze(1)
            target = target.type_as(predictions)

            loss = criterion(predictions, target)

            acc = binary_accuracy(predictions, target)
            epoch_acc += acc.item()
            epoch_loss += loss.item()

    epoch_loss = epoch_loss / len(data_loader)
    epoch_acc = epoch_acc / len(data_loader)

    return epoch_loss, epoch_acc


def train(model,
          train_loader, val_loader,
          criterion, optimizer,
          lr_scheduler, early_stopping,
          num_epochs, device):

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    for epoch in range(num_epochs):

        tr_epoch_loss, tr_epoch_acc = train_fn(model, train_loader, criterion, optimizer, device)
        val_epoch_loss, val_epoch_acc = valid_fn(model, val_loader, criterion, device)

        print(f'Epoch {epoch+1}  train loss: {tr_epoch_loss}, train acc: {tr_epoch_acc}, '
              f'val loss: {val_epoch_loss}, val acc: {val_epoch_acc}')

        history['train_loss'].append(tr_epoch_loss)
        history['train_acc'].append(tr_epoch_acc)
        history['val_loss'].append(val_epoch_loss)
        history['val_acc'].append(val_epoch_acc)

        lr_scheduler(val_epoch_loss)
        early_stopping(val_epoch_loss)
        if early_stopping.early_stop:
            break

    return model, history


def main():
    train_path = '../../../data/train.csv'
    val_path = '../../../data/test.csv'

    vocab = CustomVocabulary(train_path)

    train_loader, train_dataset = get_dataloader(train_path, vocab)
    val_loader, val_dataset = get_dataloader(val_path, vocab)

    vocab_size = len(train_dataset.vocab.vocab.stoi)
    embedding_size = 100
    hidden_size = 128
    n_layers = 2
    dropout = 0.5
    pad_idx = train_dataset.vocab.vocab.stoi['<pad>']
    num_epochs = 20
    learning_rate = 0.001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = RNN(vocab_size=vocab_size,
                embedding_dim=embedding_size,
                hidden_dim=hidden_size,
                output_dim=1,
                n_layers=n_layers,
                bidirectional=True,
                dropout=dropout,
                pad_idx=pad_idx).to(device)

    criterion = nn.BCEWithLogitsLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    lr_scheduler = LRScheduler(optimizer)
    early_stopping = EarlyStopping()

    print('Start training.')
    model, history = train(model,
                           train_loader, val_loader,
                           criterion, optimizer,
                           lr_scheduler, early_stopping,
                           num_epochs, device)

    torch.save(model.state_dict(), 'lstm_model.pt')

    with open('word2index.json', 'w') as file:
        json.dump(dict(vocab.vocab.stoi), file)

    with open('history.json', 'w') as file:
        json.dump(history, file)


if __name__ == '__main__':
    main()
