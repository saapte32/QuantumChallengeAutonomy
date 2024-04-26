import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "data/train"
VAL_DIR = "data/val"
BATCH_SIZE = 5
LEARNING_RATE = 5e-5
LAMBDA_IDENTITY = 0.5
LAMBDA_CYCLE = 25
NUM_WORKERS = 20
NUM_EPOCHS = 20
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_GEN_D = "genh.pth.tar"
CHECKPOINT_GEN_N = "genz.pth.tar"
CHECKPOINT_CRITIC_D = "critich.pth.tar"
CHECKPOINT_CRITIC_N = "criticz.pth.tar"

transforms = A.Compose(
    [
        A.Resize(width=256, height=256),
        #A.Resize(width=512, height=512),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        #A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=511),
        ToTensorV2(),
    ],
    additional_targets={"image0": "image"},
)