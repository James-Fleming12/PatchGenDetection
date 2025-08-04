import torch
from torch import nn
import torch.functional as F

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

def trainJEPA() -> JepaGenDetect:
    num_epochs = 400
    m = 0.996 # momentum

    unsupervised_loader = torch.DataLoader() # ...
    image_size = 0
    patch_size = 0
    optimizer = torch.optim.Adam()

    lgen = PatchJEPA(image_size, patch_size)

    for epoch in range(num_epochs):
        for imgs in unsupervised_loader:
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

def trainMOCO() -> ConGenDetect:
    num_epochs = 400
    m = 0.996 # momentum

    unsupervised_loader = torch.DataLoader() # ...
    image_size = [0, 0]
    patch_size = [0, 0]
    queue_size = 0
    optimizer = torch.optim.Adam()

    lgen = PatchMoco(image_size, patch_size, queue_size)

    for epoch in range(num_epochs):
        for imgs in unsupervised_loader:
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