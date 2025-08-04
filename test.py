from src.helper import init_model

import torch

patch_size = 10
crop_size = 100 # ...

encoder, decoder = init_model(
    device=torch.device('cpu'),
    patch_size=patch_size,
    crop_size=crop_size
)

print(encoder(torch.rand(1, 3, crop_size, crop_size)).size()) # returns [batch_size, crop_size^2 / patch_size^2, 768]
# therefore returns [batch_size, num_tokens, emb_dim]