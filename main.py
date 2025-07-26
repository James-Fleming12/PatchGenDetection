import torch
from torch import nn

class LatentPatch(nn.Module):
    # takes some image and returns a sequence of latent tokens that describe it
    # trained contrastively so that each token is similar to the tokens of similar images
    def __init__(self, image_size, patch_size, batch_size, latent_dim):
        super.__init__(LatentPatch, self)
        assert(image_size[0] % patch_size[0] == 0, "Patch Size needs to divide Image Size")
        assert(image_size[1] % patch_size[1] == 0, "Patch Size needs to divide Image Size")
        assert(image_size[2] == patch_size[2], "Patch Size channels and Image Size channels need to match")
        self.batch_size = batch_size
        self.image_size = image_size # size of [H, W, C]
        self.patch_size = patch_size # size of [h, w, C]
        self.latent_dim = latent_dim # size of [d]

        # using the design of MoCo, these can even be fully connected MLPs, as long as they produce embeddings
        # self.encoder = 
        # self.keyencoder = 

    def forward(x):
        pass

    def get_latent(x):
        with torch.no_grad():
            pass

    def loss(x): # an input of size [B, S, d], or Batch-Size, Sequence Length, Dimension
        pass

class PatchGenerationDetector(nn.Module):
    def __init__(self):
        super.__init__(PatchGenerationDetector, self)

    def forward(x):
        pass

def main():
    pass

if __name__=="__main__":
    main()
