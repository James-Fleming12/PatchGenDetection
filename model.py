import copy

import torch
from torch import nn
import torch.functional as F

from src.helper import init_model

from src.masks.utils import apply_masks
from src.utils.tensors import repeat_interleave_batch

from src.models.vision_transformer import Transformer

class TransformerClassifier(nn.Module):
    def __init__(self, transformer: Transformer, num_classes=2):
        super().__init__(TransformerClassifier, self) # set num_tokens
        self.transformer = transformer
        self.head = nn.Linear(transformer.embed_dim, num_classes)

    def foward(self, x):
        features = self.vit(x)
        pooled = features.mean(dim=1) # global pooling over patches
        logits = self.head(pooled)
        return logits

class PatchJEPA(nn.Module):
    # takes some image and returns a sequence of latent tokens that describe it
    # trained contrastively so that each token is similar to the tokens of similar images
    def __init__(self, image_size, patch_size, batch_size):
        super.__init__(PatchJEPA, self)
        assert(image_size[1] % patch_size[1] == 0 and image_size[2] % patch_size[2] == 0, "Patch Size needs to divide Image Size")
        assert(image_size[0] == patch_size[0], "Patch Size channels and Image Size channels need to match")
        assert(image_size[1] == image_size[2], "Image Size needs to be Square")
        assert(patch_size[1] == patch_size[2], "Patch Size needs to be Square")
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

        self.latent_dim = self.encoder.embed_dim # has to be set to this so the code works well with the I-JEPA library

    def forward(self, x, masks_enc, masks_pred):
        # target encoder
        with torch.no_grad():
            h = self.target_encoder(x)
            h = F.layer_norm(h, (h.size(-1),))
            B = len(h)
            h = apply_masks(h, masks_pred)
            h = repeat_interleave_batch(h, B, repeat=len(masks_enc))

        # prediction
        z = self.encoder(x, masks_enc) # might need (imgs)
        z = self.predictor(z, masks_enc, masks_pred)

        return z, h

    def get_latent(self, x):
        with torch.no_grad():
            return self.encoder(x.unsqueeze(0))

    def loss(self, x, target): # an input of size [B, S, d], or Batch-Size, Sequence Length, Dimension
        pass

class JepaGenDetect(nn.Module):
    def __init__(self, lgen: PatchJEPA):
        super.__init__(JepaGenDetect, self)
        self.lgen = lgen
        num_tokens = self.moco.image_size[1]^2 / self.moco.patch_size[1]^2
        t = Transformer(num_tokens=num_tokens)
        self.classifier = TransformerClassifier(t)

    def forward(self, x):
        x = self.lgen(x)
        x = self.classifier(x)
        return x

class PatchMoco(nn.Module):
    def __init__(self, image_size, patch_size, queue_size):
        super.__init__(PatchMoco, self)
        assert(image_size[1] % patch_size[1] == 0 and image_size[2] % patch_size[2] == 0, "Patch Size needs to divide Image Size")
        assert(image_size[0] == patch_size[0], "Patch Size channels and Image Size channels need to match")
        assert(image_size[1] == image_size[2], "Image Size needs to be Square")
        assert(patch_size[1] == patch_size[2], "Patch Size needs to be Square")

        self.image_size = image_size # size of [C, H, W]
        self.patch_size = patch_size # size of [C, h, w]
        self.queue_size = queue_size
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

        self.emb_dim = self.encoder.emb_dim
        self.criterion = PatchMocoLoss(self.emb_dim, queue_size)

    def forward(self, x): # expecting [B, C, H, W]
        pred = self.encoder(x)
        with torch.no_grad():
            target = self.target_encoder(x)
        return pred, target

    def get_latent(self, x):
        with torch.no_grad():
            return self.encoder(x.unsqueeze(0))

    def loss(self, pred, target):
        return self.criterion(pred, target)

class PatchMocoLoss(nn.Module):
    def __init__(self, emb_size, queue_size=65536, temperature=0.07):
        super(PatchMocoLoss, self).__init__()
        self.temperature = temperature
        self.queue_size = queue_size
        self.register_buffer("queue", torch.randn(queue_size, emb_size)) # non-trainable tensor
        self.queue = F.normalize(self.queue, dim=1)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)

        # Handle cases where the update exceeds queue boundaries
        if ptr + batch_size > self.queue_size:
            # Split the update into two parts: end of queue and start of queue
            remaining = self.queue_size - ptr
            self.queue[ptr:] = keys[:remaining]
            self.queue[:batch_size - remaining] = keys[remaining:]
        else:
            # Standard case: single contiguous update
            self.queue[ptr:ptr + batch_size] = keys
        # Update pointer (modulo queue_size)
        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr

    def forward(self, pred, target):
        # Normalize the embeddings
        pred = F.normalize(pred, dim=1)  # queries [B, D]
        target = F.normalize(target, dim=1)  # keys [B, D]

        l_pos = torch.einsum('nc,nc->n', [pred, target]).unsqueeze(-1) # Positive logits: Nx1
        l_neg = torch.einsum('nc,ck->nk', [pred, self.queue.clone().detach().t()]) # Negative logits: NxK
        logits = torch.cat([l_pos, l_neg], dim=1) # Logits: Nx(1+K)

        logits /= self.temperature # Apply temperature

        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(pred.device) # Labels: positives are the first (0-th)

        self._dequeue_and_enqueue(target) # Update the queue
        return F.cross_entropy(logits, labels) # Compute cross entropy loss

class ConGenDetect(nn.Module):
    def __init__(self, moco: PatchMoco):
        super.__init__(ConGenDetect, self)
        self.moco = moco
        num_tokens = self.moco.image_size[1]^2 / self.moco.patch_size[1]^2
        t = Transformer(num_tokens=num_tokens)
        self.classifier = TransformerClassifier(t)

    def forward(self, x):
        x = self.moco(x)
        x = self.classifier(x)
        return x