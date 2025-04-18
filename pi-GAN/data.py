import os
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import torchvision
from pathlib import Path
from config import *

class ImageDataset(Dataset):
    def __init__(self, folders: tuple, image_size: int = 64, transparent: bool = False, aug_prob: float = 0., exts=None):
        super().__init__()
        print(f"Initializing ImageDataset with folders: {folders}")
        if exts is None:
            exts = ['jpg', 'jpeg', 'png']

        # Expect a tuple of two directories: (target_dir, center_dir)
        assert isinstance(folders, (list, tuple)) and len(folders) == 2, \
            "`folders` must be a tuple (target_dir, center_dir)"
        target_dir, center_dir = folders

        # Gather and sort image paths
        self.target_paths = sorted(
            p for ext in exts for p in Path(target_dir).rglob(f'*.{ext}')
        )
        self.center_paths = sorted(
            p for ext in exts for p in Path(center_dir).rglob(f'*.{ext}')
        )
        print(f"Found {len(self.target_paths)} target images and {len(self.center_paths)} center images")

        assert len(self.target_paths) == len(self.center_paths), \
            "Target and center folders must contain the same number of images"

        # Ensure filenames match one-to-one
        target_names = [p.name for p in self.target_paths]
        center_names = [p.name for p in self.center_paths]
        assert target_names == center_names, \
            "Filenames in target and center directories do not match"
        print("Image filenames verified: all match one-to-one")

        # Load PIL images
        self.target_imgs = []
        self.center_imgs = []
        for t_path, c_path in zip(self.target_paths, self.center_paths):
            try:
                target_img = Image.open(t_path).convert('RGB')
                center_img = Image.open(c_path).convert('RGB')
                self.target_imgs.append(target_img)
                self.center_imgs.append(center_img)
            except Exception as e:
                print(f"Warning: failed to load image pair {t_path}, {c_path}: {e}")
        print(f"Loaded {len(self.target_imgs)} image pairs successfully")

        # Set up transforms
        self.image_size = image_size
        self.target_transform = T.Compose([
            T.Resize(image_size),
            T.CenterCrop(image_size),
            T.ToTensor(),
        ])
        # Center images always resized to 128x128
        self.center_transform = T.Compose([
            T.Resize(128),
            T.CenterCrop(128),
            T.ToTensor(),
        ])

        # Apply transforms and stack
        self.target_img_arr = torch.stack([
            self.target_transform(img) for img in self.target_imgs
        ])  # shape: (N, 3, image_size, image_size)
        self.center_img_arr = torch.stack([
            self.center_transform(img) for img in self.center_imgs
        ])  # shape: (N, 3, 128, 128)
        print(f"Transformed images: target shape {self.target_img_arr.shape}, center shape {self.center_img_arr.shape}")

    def __len__(self):
        length = len(self.target_img_arr)
        return length

    def __getitem__(self, idx):
        return self.target_img_arr[idx], self.center_img_arr[idx]

dataset = ImageDataset((image_data_dir, center_data_dir), 16)
val_dataset = ImageDataset((val_image_data_dir, val_center_data_dir), 16)