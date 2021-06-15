import torch
import torch.nn as nn
from torch import optim
from torchtext.vocab import Vocab
import pandas as pd
from collections import Counter
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from src.models.LSTM.model import RNN


class CustomVocabulary:
    def __init__(self, df):
        self.df = df
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
    def __init__(self, df_path):
        self.df = pd.read_csv(df_path)
        self.vocab = CustomVocabulary(self.df)

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


def get_dataloader(df_path, batch_size=64, shuffle=True):

    train_dataset = CustomDataset(df_path)
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


def train_fn(model, data_loader, criterion, optimizer, num_epochs, device):

    history = []
    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_acc = 0
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
        print(f'Epoch {epoch+1} loss: {epoch_loss}, acc: {epoch_acc}')

        history.append(epoch_loss)

    return model, history


def main():
    df_path = '../../../data/train.csv'
    train_loader, train_dataset = get_dataloader(df_path)

    vocab_size = len(train_dataset.vocab.vocab.stoi)
    embedding_size = 100
    hidden_size = 256
    n_layers = 2
    dropout = 0.5
    pad_idx = train_dataset.vocab.vocab.stoi['<pad>']
    num_epochs = 10
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

    print('Start training.')
    model, history = train_fn(model, train_loader, criterion, optimizer, num_epochs, device)

    torch.save(model.state_dict(), 'lstm_model2')


if __name__ == '__main__':
    main()
