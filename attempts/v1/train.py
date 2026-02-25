#!/usr/bin/env python
# coding: utf-8

# Import Dataset Class from other notebook and other libraries

# In[ ]:


from torch import nn
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

class FaceDataset(Dataset):
    def __init__(self, folder_path):
        self.image_paths = []

        for filename in os.listdir(folder_path):
            if filename.lower().endswith('.jpg'):
                full_path = os.path.join(folder_path, filename)
                self.image_paths.append(full_path)

        if len(self.image_paths) == 0:
            raise IOError("No files found")

        print(str(len(self.image_paths)) + "images imported")

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img = Image.open(self.image_paths[index]).convert('RGB')
        return self.transform(img)


# ![meme](assets/meme.jpg "Title")
# 
# For Face Swapping Architecture, we will use kind of U-net Architecture but without skips (concatenation) at first.
# 
# We will have two decoders, one for each person and a common encoder.
# 
# We will use 4 convolutional blocks in the encoder and 4 upsampling blocks in the decoder. The bottleneck will be at 512 channels.
# 
# The block is per https://arxiv.org/pdf/1505.04597 section 2 (Network Architecture) of the U-net paper is Conv2d -> ReLU -> MaxPool. However, we will use BatchNorm instead of MaxPool as this is the modern way to do it.
# 
# See also: https://ai.stackexchange.com/a/40638

# In[ ]:


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


# In[ ]:


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(3,   64,  2),
            ConvBlock(64,  128, 2),
            ConvBlock(128, 256, 2),
            ConvBlock(256, 512, 2),
            ConvBlock(512, 1024, 2),# Encoder Bottleneck
        )

    def forward(self, x):
        return self.net(x)


# In[ ]:


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBlock(1024, 512), # Decoder Bottleneck

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBlock(512, 256), # Decoder Bottleneck

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBlock(256, 128),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBlock(128, 64),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBlock(64, 32),

            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)


# In[ ]:


import time
from matplotlib import pyplot as plt

def log_progress(epoch, loss, epoch_start_time, start_time):
    elapsed = time.time() - start_time
    epoch_time = time.time() - epoch_start_time
    print(f"Epoch {epoch} - Loss: {loss:.4f} - Elapsed Time: {elapsed:.1f} seconds - Epoch Time: {epoch_time:.1f} seconds")

def denormalize(tensor):
    return tensor * 0.5 + 0.5

def save_preview(encoder, decoder_a, decoder_b, sample_a, sample_b, epoch):
    encoder.eval()
    decoder_a.eval()
    decoder_b.eval()

    with torch.no_grad():
        latent_a = encoder(sample_a)
        latent_b = encoder(sample_b)

        recon_a = decoder_a(latent_a)
        recon_b = decoder_b(latent_b)

        swap_ab = decoder_b(latent_a)
        swap_ba = decoder_a(latent_b)

    imgs = [sample_a[0], recon_a[0], swap_ab[0], sample_b[0], recon_b[0], swap_ba[0]]
    titles = ["Real A", "Recon A", "Swap A to B", "Real B", "Recon B", "Swap B to A"]

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    for ax, img, title in zip(axes, imgs, titles):
        img = denormalize(img).clamp(0, 1)
        img = img.permute(1, 2, 0).cpu().numpy()
        ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')

    plt.suptitle(f"Epoch {epoch}")
    plt.tight_layout()
    plt.savefig(f"checkpoints/{epoch}/result.png")
    plt.show()
    plt.close()

    encoder.train()
    decoder_a.train()
    decoder_b.train()


# In[1]:


import os
import torch
from torch.utils.data import DataLoader

BATCH_SIZE = 11
LR = 0.00005
NUM_WORKERS = 1

def train():
    start_time = time.time()

    dataset_a = FaceDataset("../../personA")
    dataset_b = FaceDataset("../../personB")

    loader_a = DataLoader(dataset_a, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=NUM_WORKERS)
    loader_b = DataLoader(dataset_b, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=NUM_WORKERS)

    sample_a = next(iter(loader_a)).to("cuda")
    sample_b = next(iter(loader_b)).to("cuda")

    encoder = Encoder().to("cuda")
    decoder_a = Decoder().to("cuda")
    decoder_b = Decoder().to("cuda")

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) +
        list(decoder_a.parameters()) +
        list(decoder_b.parameters()),
        lr=LR
    )

    loss_fn = nn.L1Loss()

    for epoch in range(1, 100):
        epoch_start_time = time.time()
        total_loss = 0
        steps = 0

        for img_a, img_b in zip(loader_a, loader_b):
            img_a = img_a.to("cuda", non_blocking=True)
            img_b = img_b.to("cuda", non_blocking=True)

            latent_a = encoder(img_a)
            latent_b = encoder(img_b)

            reconstructed_a = decoder_a(latent_a)
            reconstructed_b = decoder_b(latent_b)

            loss_a = loss_fn(reconstructed_a, img_a)
            loss_b = loss_fn(reconstructed_b, img_b)
            loss = loss_a + loss_b

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            steps += 1

        avg_loss = total_loss / steps
        log_progress(epoch, avg_loss, epoch_start_time, start_time)
        os.makedirs("", exist_ok=True)
        os.makedirs(f"attempts/v1/checkpoints/{epoch}", exist_ok=True)
        torch.save(encoder.state_dict(), f"attempts/v1/checkpoints/{epoch}/encoder.pth")
        torch.save(decoder_a.state_dict(), f"attempts/v1/checkpoints/{epoch}/decoder_A.pth")
        torch.save(decoder_b.state_dict(), f"attempts/v1/checkpoints/{epoch}/decoder_B.pth")
        save_preview(encoder, decoder_a, decoder_b, sample_a, sample_b, epoch)
        print("\nSaved/")


# In[ ]:


if __name__ == "__main__":
    train()

