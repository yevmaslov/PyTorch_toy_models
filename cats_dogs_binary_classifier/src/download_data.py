import os
from kaggle.api.kaggle_api_extended import KaggleApi
from zipfile import ZipFile


DATA_PATH = './data'


def download_dataset():
    api = KaggleApi()
    api.authenticate()

    competition_name = 'dogs-vs-cats'

    if not os.path.isdir(DATA_PATH):
        os.mkdir(DATA_PATH)

    api.competition_download_files(competition_name,
                                   path=DATA_PATH)


def extract_dataset():

    main_archive_path = os.path.join(DATA_PATH, 'dogs-vs-cats.zip')
    main_archive = ZipFile(main_archive_path)
    main_archive.extractall(DATA_PATH)

    train_archive_path = os.path.join(DATA_PATH, 'train.zip')
    train_archive = ZipFile(train_archive_path)
    train_archive.extractall(DATA_PATH)

    test_archive_path = os.path.join(DATA_PATH, 'test1.zip')
    test_archive = ZipFile(test_archive_path)
    test_archive.extractall(DATA_PATH)


if __name__ == '__main__':
    download_dataset()
    print('Dataset download succeed.')
    extract_dataset()
    print('Dataset extracted succeed.')
