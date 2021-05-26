from torch.utils.data import Dataset
from PIL import Image
import os


class CustomDataset(Dataset):
    def __init__(self, path, img_list, transform=None):
        self.path = path
        self.img_list = img_list
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.path, self.img_list[idx])

        image = Image.open(img_path)
        label = 0 if self.img_list[idx].startswith('cat') else 1

        if self.transform:
            image = self.transform(image)

        return image, label
