from src.models.LSTM.model import RNN
import re
import json
import torch


def load_vocab_model():
    with open('word2index.json', encoding='utf-8') as json_file:
        stoi = json.load(json_file)

    vocab_size = len(stoi)
    model = RNN(vocab_size=vocab_size)
    return stoi, model


def preprocess(text, stoi):
    text = text.strip().lower()
    text = re.sub(r'[^\w\s]', '', text)
    indexes = [stoi[w] if w in stoi else stoi['<unk>'] for w in text.split(' ')]

    length = [len(indexes)]
    tensor = torch.LongTensor(indexes)
    tensor = tensor.unsqueeze(1)
    length_tensor = torch.LongTensor(length)
    return tensor, length_tensor


def inference():
    stoi, model = load_vocab_model()
    while True:
        text = input('Enter a comment (or nothing to breek): ')
        if not text:
            break
        tensor, length_tensor = preprocess(text, stoi)

        label = 'not toxic'
        prediction = torch.sigmoid(model(tensor, length_tensor))
        if prediction > 0.5:
            label = 'toxic'
        print(f'Model prediction: {label} comment')


if __name__ == '__main__':
    inference()
