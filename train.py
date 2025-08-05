import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.functional as F
from torchvision import transforms

import kagglehub
import os
from PIL import Image

from main import ConGenDetect, JepaGenDetect, PatchJEPA, PatchMoco

def generate_nonoverlapping_masks(grid_h, grid_w, enc_ratio=0.3, pred_ratio=0.2):
    total_patches = grid_h * grid_w
    idxs = torch.randperm(total_patches)
    num_enc = int(enc_ratio * total_patches)
    num_pred = int(pred_ratio * total_patches)

    enc_idxs = idxs[:num_enc]
    pred_idxs = idxs[num_enc:num_enc + num_pred]

    enc_mask = torch.zeros(total_patches, dtype=torch.bool)
    pred_mask = torch.zeros(total_patches, dtype=torch.bool)

    enc_mask[enc_idxs] = 1
    pred_mask[pred_idxs] = 1

    return enc_mask, pred_mask

class DalleDataset(Dataset):
    def __init__(self, root_dir):
        """root_dir should be the directory with all the images, with real and fake subfolders"""
        self.root_dir = root_dir
        assert(os.path.exists(root_dir), f"Root Directory {root_dir} Does not exist")

        real_dir = os.path.join(root_dir, 'real')
        assert(os.path.exists(real_dir), f"Subfolder real does not exist under {root_dir}")
        self.real_images = [os.path.join(real_dir, f) for f in os.listdir(real_dir) 
                           if f.endswith(('.png', '.jpg', '.jpeg'))]

        fake_dir = os.path.join(root_dir, "fake-v2")
        assert(os.path.exists(fake_dir), f"Subfolder fake-v2 does not exist under {root_dir}")
        self.fake_images = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir) 
                           if f.endswith(('.png', '.jpg', '.jpeg'))]

        self.imgs = self.fake_images + self.real_images
        self.labels = [0] * len(self.fake_images) + [1] * len(self.real_images)

        self.image_size = (256, 256)

        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], # Standard ImageNet normalization
                                std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, idx):
        img_path = self.all_images[idx]
        image = Image.open(img_path).convert('RGB') # Convert to RGB to ensure 3 channels

        label = self.labels[idx]
        image = self.transform(image)

        return image, label

    def downloadDaLLEData():
        path = kagglehub.dataset_download("superpotato9/dalle-recognition-dataset", path=os.path.join(os.getcwd(), "data"))
        print(f"Path to dataset files: {path}, should be in CurrentWorkingDirectory/data")

def trainJEPA(dataloader: DataLoader) -> JepaGenDetect:
    num_epochs = 400
    m = 0.996 # momentum

    image_size = 0
    patch_size = 0
    optimizer = torch.optim.Adam()

    lgen = PatchJEPA(image_size, patch_size)

    for epoch in range(num_epochs):
        for imgs in dataloader:
            # generate valid patches for encoder and target
            B, C, H, W = imgs.shape
            for _ in range(B):
                enc_mask, pred_mask = generate_nonoverlapping_masks(H, W)
                masks_enc.append(enc_mask)
                masks_pred.append(pred_mask)

            imgs = imgs.to(lgen.device)
            masks_enc = [m.to(lgen.device) for m in masks_enc]
            masks_pred = [m.to(lgen.device) for m in masks_pred]

            z, h = lgen(imgs, masks_enc, masks_pred)

            loss = lgen.loss(z, h)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # momentum update
            with torch.no_grad():
                for param_q, param_k in zip(lgen.encoder.parameters(), lgen.target_encoder.parameters()):
                    param_k.data.mul_(m).add((1.-m) * param_q.data)

        print(f"Epoch {epoch} finished: Loss {loss.item(): .4f}")

    print("JEPA finished training")

    patchdet = JepaGenDetect(lgen)

    return patchdet

def trainMOCO(dataloader: DataLoader) -> ConGenDetect:
    num_epochs = 400
    m = 0.996 # momentum

    image_size = [0, 0]
    patch_size = [0, 0]
    queue_size = 0
    optimizer = torch.optim.Adam()

    lgen = PatchMoco(image_size, patch_size, queue_size)

    for epoch in range(num_epochs):
        for imgs in dataloader:
            pred, target = lgen(imgs)

            loss = lgen.loss(pred, target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # momentum update
            with torch.no_grad():
                for param_q, param_k in zip(lgen.encoder.parameters(), lgen.target_encoder.parameters()):
                    param_k.data.mul_(m).add((1.-m) * param_q.data)

    print("MoCo finished training")

    patchdet = ConGenDetect(lgen)

    return patchdet

def validateModel(model: ConGenDetect | JepaGenDetect, dataloader):
    pass