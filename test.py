from src.helper import init_model

import torch

patch_size = 10
crop_size = 100 # ...

encoder, decoder = init_model(
    device=torch.device('cpu'),
    patch_size=patch_size,
    crop_size=crop_size
)

print(encoder(torch.rand(1, 3, crop_size, crop_size)).size())