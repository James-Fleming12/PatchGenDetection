import copy

import torch
from torch import nn

from src.helper import init_model

class PatchJEPA(nn.Module):
    # takes some image and returns a sequence of latent tokens that describe it
    # trained contrastively so that each token is similar to the tokens of similar images
    def __init__(self, image_size, patch_size, batch_size):
        super.__init__(PatchJEPA, self)
        assert(image_size[0] % patch_size[0] == 0, "Patch Size needs to divide Image Size")
        assert(image_size[1] % patch_size[1] == 0, "Patch Size needs to divide Image Size")
        assert(image_size[0] == patch_size[0], "Patch Size channels and Image Size channels need to match")
        assert(image_size[1] == image_size[2], "Image Size needs to be Square")
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
            patch_size=self.patch_size[1],
            crop_size=self.image_size[1]
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

class JepaGenDetect(nn.Module):
    def __init__(self, lgen: PatchJEPA):
        super.__init__(JepaGenDetect, self)
        self.lgen = lgen

    def forward(self, x):
        pass

class PatchMoco(nn.Module):
    def __init__(self, image_size, patch_size):
        super.__init__(PatchMoco, self)
        assert(image_size[0] % patch_size[0] == 0, "Patch Size needs to divide Image Size")
        assert(image_size[1] % patch_size[1] == 0, "Patch Size needs to divide Image Size")
        assert(image_size[0] == patch_size[0], "Patch Size channels and Image Size channels need to match")
        assert(image_size[1] == image_size[2], "Image Size needs to be Square")

        self.image_size = image_size # size of [C, H, W]
        self.patch_size = patch_size # size of [C, h, w]
        if not torch.cuda.is_available():
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda:0')
            torch.cuda.set_device(self.device)

        self.encoder, _ = init_model(
            device=self.device,
            patch_size=self.patch_size[1],
            crop_size=self.image_size[1]
        )
        self.target_encoder = copy.deepcopy(self.encoder)

    def forward(self, x):
        pass

class ConGenDetect(nn.Module):
    def __init__(self, moco: PatchMoco):
        super.__init__(ConGenDetect, self)
        self.moco = moco

    def forward(self, x):
        pass