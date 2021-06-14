from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pandas as pd
from pickle import dump
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
import numpy as np


def validate_model(estimator, x, y):
    k_fold = StratifiedKFold(shuffle=True, random_state=0)

    folds_acc = []
    folds_precision = []
    folds_recall = []

    for train_idx, val_idx in k_fold.split(x, y):
        x_train, y_train = x[train_idx, :], y[train_idx]
        x_val, y_val = x[val_idx, :], y[val_idx]

        model = estimator
        model.fit(x_train, y_train)
        predictions = model.predict(x_val)

        acc = metrics.accuracy_score(y_val, predictions)
        precision = metrics.precision_score(y_val, predictions)
        recall = metrics.recall_score(y_val, predictions)

        folds_acc.append(acc)
        folds_precision.append(precision)
        folds_recall.append(recall)

    folds_acc = np.array(folds_acc)
    folds_precision = np.array(folds_precision)
    folds_recall = np.array(folds_recall)

    return folds_acc, folds_precision, folds_recall


def main():
    train = pd.read_csv('../../../data/train.csv')

    x = train.comment_preprocessed_str
    y = train.toxic.values

    vectorizer = TfidfVectorizer()
    x = vectorizer.fit_transform(x)
    dump(vectorizer, open('tfidf.pkl', 'wb'))

    models = [
        ('LogisticRegression', LogisticRegression()),
        ('RandomForestClassifier', RandomForestClassifier()),
        ('XGBClassifier', XGBClassifier(objective='binary:logistic'))
    ]

    best_acc = 0
    for model_name, model in models:
        acc, precision, recall = validate_model(model, x, y)
        print(f'{model_name} mean fold acc: {acc.mean()}, precision: {precision.mean()}, recall: {recall.mean()}')
        model.fit(x, y)
        dump(model, open(f'{model_name}.pkl', 'wb'))

        if acc.mean() > best_acc:
            best_acc = acc.mean()
            dump(model, open(f'best_model.pkl', 'wb'))


if __name__ == '__main__':
    main()
