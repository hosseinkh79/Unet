import os
import numpy as np
from PIL import Image

from going_modular import configs

from torchvision import transforms
from torch.utils.data import Dataset
import torch

class ADE20KDataset(Dataset):
    def __init__(self, images_dir, annotations_dir, transforms=None):
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.transforms = transforms
        self.image_files = sorted(os.listdir(images_dir))
        self.mask_files = sorted(os.listdir(annotations_dir))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_filename = self.image_files[idx]
        
        img_name = os.path.join(self.images_dir, image_filename)
        image = Image.open(img_name).convert("RGB")
        image = torch.tensor(np.array(image)).permute(2, 0, 1)/255
        # print(f'image : {image.shape}')
        
        # Generate the mask filename based on the image filename
        mask_filename = image_filename.replace(".jpg", ".png")
        mask_name = os.path.join(self.annotations_dir, mask_filename)

        mask = Image.open(mask_name)
        mask = torch.tensor(np.array(mask)).unsqueeze(0)
        mask[mask == 150] = 149
        # print(f'mask : {mask.shape}')

        if self.transforms:
            image = self.transforms(image)
            mask = self.transforms(mask)

        return image, mask
