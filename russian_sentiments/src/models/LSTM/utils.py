from collections import Counter
import re
import torch
import pandas as pd
from torchtext.vocab import Vocab
from torch.utils.data import Dataset


class CustomVocabulary2:
    def __init__(self, df):
        self.df = df
        self.vocab = self.get_vocabulary()

    def get_vocabulary(self):
        counter = Counter()

        for text in self.df.comment.values:
            text_prep = text.strip().lower()
            text_prep = re.sub(r'[^\w\s]', '', text_prep)
            for word in text_prep.split(' '):
                counter[word] += 1

        vocab = Vocab(counter, max_size=10000, specials=['<unk>', '<pad>', '<sos>', '<eos>'])
        return vocab


class CustomDataset2(Dataset):
    def __init__(self, df_path):
        self.df = pd.read_csv(df_path)
        self.vocab = CustomVocabulary2(self.df)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        x = self.df.iloc[index].comment
        y = self.df.iloc[index].toxic

        x = x.strip().lower()
        x = re.sub(r'[^\w\s]', '', x)

        x_ind = self.vocab.sentence_to_indices(x)

        x_out = [self.vocab.vocab.stoi['<sos>']]
        x_out += x_ind
        x_out += [self.vocab.vocab.stoi['<eos>']]

        return torch.tensor(x_out), y
