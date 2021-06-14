import pandas as pd
import spacy
import time
from sklearn.model_selection import train_test_split

nlp = spacy.load('ru_core_news_sm')


def preprocess(text):
    text = text.strip()
    tokens = nlp(text)
    filtered_tokens = []
    for token in tokens:
        if (not token.is_stop) and (not token.is_punct):
            filtered_tokens.append(token.lemma_)

    return filtered_tokens


def main():
    df = pd.read_csv('../data/labeled.csv')
    df['toxic'] = df['toxic'].astype(int)

    start = time.time()
    df['comment_preprocessed'] = df.comment.apply(preprocess)
    stop = time.time()
    print('Executing time: ', stop - start)

    df['comment_preprocessed_str'] = df.comment_preprocessed.apply(lambda x: ' '.join([i for i in x]))

    train, test = train_test_split(df, test_size=0.2, random_state=0)

    train.to_csv('../data/train.csv', index=False)
    test.to_csv('../data/test.csv', index=False)


if __name__ == '__main__':
    main()
