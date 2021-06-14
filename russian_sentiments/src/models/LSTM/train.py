import torch
from torchtext.vocab import Vocab
import pandas as pd
from collections import Counter
from pickle import dump
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader


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
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        text = [item[0] for item in batch]
        text = pad_sequence(text, batch_first=False, padding_value=self.pad_idx)
        target = [item[1] for item in batch]
        return text, target


def get_dataloader(df_path, batch_size=64, num_workers=4, shuffle=True, pin_memory=True):

    train_dataset = CustomDataset(df_path)
    pad_idx = train_dataset.vocab.vocab.stoi['<pad>']

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=CustomCollate(pad_idx=pad_idx)
    )

    return train_loader, train_dataset


def main():
    df_path = '../../../data/train.csv'

    train_loader, train_dataset = get_dataloader(df_path)


if __name__ == '__main__':
    main()
