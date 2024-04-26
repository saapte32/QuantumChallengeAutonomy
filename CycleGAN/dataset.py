from PIL import Image
import os
from torch.utils.data import Dataset
import numpy as np
import config
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
class daynightDataset(Dataset):
    def __init__(self, root_night, root_day, transform=None):
        self.root_night = root_night
        self.root_day = root_day
        self.transform = transform

        self.night_images = os.listdir(root_night)
        self.day_images = os.listdir(root_day)
        self.length_dataset = max(len(self.night_images), len(self.day_images)) # 1000, 1500
        self.night_len = len(self.night_images)
        self.day_len = len(self.day_images)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        night_img = self.night_images[index % self.night_len]
        day_img = self.day_images[index % self.day_len]

        night_path = os.path.join(self.root_night, night_img)
        day_path = os.path.join(self.root_day, day_img)

        night_img = np.array(Image.open(night_path).convert("RGB"))
        day_img = np.array(Image.open(day_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=night_img, image0=day_img)
            night_img = augmentations["image"]
            day_img = augmentations["image0"]

        return night_img, day_img




