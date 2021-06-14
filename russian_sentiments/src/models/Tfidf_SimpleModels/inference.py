from pickle import load
from sklearn.metrics import accuracy_score
import pandas as pd


def inference(model, vectorizer):
    print('Enter nothing to break.')
    while True:
        text = input('Enter a comment (russian language): ')
        if not text:
            break
        text = [text]
        text_prep = vectorizer.transform(text)
        prediction = model.predict(text_prep)
        answer = 'toxic' if prediction == 1 else 'not toxic'
        print('Model thinking it is ', answer, ' comment.')


def main():
    test = pd.read_csv('../../../data/test.csv')
    test = test.dropna(subset=['comment_preprocessed_str'])

    model = load(open('best_model.pkl', 'rb'))
    vectorizer = load(open('tfidf.pkl', 'rb'))

    x_test = vectorizer.transform(test.comment_preprocessed_str)
    y_test = test.toxic.values

    predictions = model.predict(x_test)
    print(f'Accuracy on test set: {accuracy_score(y_test, predictions)}')

    if input('Wanna try? (y/n): ') == 'y':
        inference(model, vectorizer)


if __name__ == '__main__':
    main()
