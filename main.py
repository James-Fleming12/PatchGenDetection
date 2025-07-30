import copy
import torch
import torch.functional as F
from torch import nn

from ijepa.src.helper import init_model
from ijepa.src.masks.utils import apply_masks
from ijepa.src.utils.tensors import repeat_interleave_batch

class LatentPatch(nn.Module):
    # takes some image and returns a sequence of latent tokens that describe it
    # trained contrastively so that each token is similar to the tokens of similar images
    def __init__(self, image_size, patch_size, batch_size):
        super.__init__(LatentPatch, self)
        assert(image_size[0] % patch_size[0] == 0, "Patch Size needs to divide Image Size")
        assert(image_size[1] % patch_size[1] == 0, "Patch Size needs to divide Image Size")
        assert(image_size[2] == patch_size[2], "Patch Size channels and Image Size channels need to match")
        self.batch_size = batch_size
        self.image_size = image_size # size of [C, H, W]
        self.patch_size = patch_size # size of [C, h, w]

        if not torch.cuda.is_available():
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda:0')
            torch.cuda.set_device(self.device)

        # using the design of MoCo, these can even be fully connected MLPs, as long as they produce embeddings
        self.encoder, self.predictor = init_model(
            device=self.device,
            patch_size=self.patch_size[1]*self.patch_size[2],
            crop_size=self.image_size[1]*self.image_size[2]
        )
        self.target_encoder = copy.deepcopy(self.encoder)

        self.latent_dim = self.encoder.embed_dim # size of [d]
        # has to be set to this so the code works well with the I-JEPA library

    def forward(self, x):
        pass

    def get_latent(self, x):
        with torch.no_grad():
            return self.encoder(x)

    def loss(self, x, target): # an input of size [B, S, d], or Batch-Size, Sequence Length, Dimension
        pass

class PatchGenerationDetector(nn.Module):
    def __init__(self, lgen: LatentPatch):
        super.__init__(PatchGenerationDetector, self)
        self.lgen = lgen

    def forward(self, x):
        pass

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

def main():
    num_epochs = 400
    m = 0.996 # momentum

    unsupervised_loader = torch.DataLoader() # ...
    optimizer = torch.optim.Adam()

    lgen = LatentPatch()

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

            # target encoder
            with torch.no_grad():
                h = lgen.target_encoder(imgs)
                h = F.layer_norm(h, (h.size(-1),))
                B = len(h)
                h = apply_masks(h, masks_pred)
                h = repeat_interleave_batch(h, B, repeat=len(masks_enc))

            # prediction
            z = lgen.encoder(imgs, masks_enc)
            z = lgen.predictor(z, masks_enc, masks_pred)

            loss = lgen.loss(z, h)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # momentum update
            with torch.no_grad():
                for param_q, param_k in zip(lgen.encoder.parameters(), lgen.target_encoder.parameters()):
                    param_k.data.mul_(m).add((1.-m) * param_q.data)

        print(f"Epoch {epoch} finished: Loss {loss.item(): .4f}")

    patchdet = PatchGenerationDetector(lgen)


if __name__=="__main__":
    main()