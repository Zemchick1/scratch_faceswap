#!/usr/bin/env python
# coding: utf-8

# Import Dataset Class from other notebook and other libraries

# In[11]:


from sympy.printing.pytorch import torch
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
            transforms.Resize((256, 256)),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img = Image.open(self.image_paths[index]).convert('RGB')
        return self.transform(img)


# Now we will try implement skips on encoder and decoder

# In[49]:


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=5, stride=2, padding=2),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch//2, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        return self.block(x)


# In[102]:


class EncoderDenseBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=5, stride=1, padding=2),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch//4, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.out = nn.Conv2d(in_channels=in_ch + out_ch // 4, out_channels=out_ch, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        out1 = self.block(x)
        out2 = torch.cat([x, out1], dim=1)
        out3 = self.out(out2)
        return out3


# In[130]:


class DecoderDenseBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=4 * out_ch, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)

    def forward(self, x):
        out1 = self.block(x)
        out2 = torch.cat([x, out1], dim=1)
        out3 = self.pixel_shuffle(out2)
        return out3


# In[128]:


class DeconvolutionBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch * 4, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.PixelShuffle(upscale_factor=2)
        )

    def forward(self, x):
        return self.block(x)


# In[137]:


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Path 1
        self.path1_CB16 = ConvBlock(3, 16) # 256 x 256 x 3 -> 128 x 128 x 8
        self.path1_CB32 = ConvBlock(8, 32) # 128 x 128 x 16 -> 64 x 64 x 16
        self.path1_DB32 = EncoderDenseBlock(16, 32) # 64 x 64 x 16 -> 32 x 32 x 32
        self.path1_DB64 = EncoderDenseBlock(32, 64) # 32 x 32 x 32 -> 16 x 16 x 64
        self.path1_DB128 = EncoderDenseBlock(64, 128)  # 16 x 16 x 64 -> 8 x 8 x 128

        # Path 2
        self.path2_CB32 = ConvBlock(3,  32) # 256 x 256 x 3 -> 128 x 128 x 16
        self.path2_CB64 = ConvBlock(16,  64) # 128 x 128 x 16 -> 64 x 64 x 32
        self.path2_DB128 = EncoderDenseBlock(32, 128)
        self.path2_DB256 = EncoderDenseBlock(128, 256)
        self.path2_DB512 = EncoderDenseBlock(256, 512)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Sequential(
            nn.Linear(512 * 8 * 8, 1024),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.fc4 = nn.Sequential(
            nn.Linear(1024, 512 * 8 * 8),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self.deconv2 = DeconvolutionBlock(512, 384)
        self.deconv1 = DeconvolutionBlock(128, 128)

    def forward(self, x):
        path1 = self.path1_CB16(x)
        path1 = self.path1_CB32(path1)
        path1 = self.path1_DB32(path1)
        path1 = self.path1_DB64(path1)
        path1 = self.path1_DB128(path1)
        path1 = self.deconv1(path1)

        path2 = self.path2_CB32(x)
        path2 = self.path2_CB64(path2)
        path2 = self.path2_DB128(path2)
        path2 = self.path2_DB256(path2)
        path2 = self.path2_DB512(path2)

        path2 = self.flatten(path2)
        path2 = self.fc1(path2)
        path2 = self.fc2(path2)
        path2 = self.fc3(path2)
        path2 = self.fc4(path2)

        path2 = path2.view(-1, 512, 8, 8)
        path2 = self.deconv2(path2)
        out = torch.cat([path1, path2], dim=1)
        return out

# In[139]:


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            DeconvolutionBlock(512, 256),
            DeconvolutionBlock(256, 128),
            DecoderDenseBlock(128, 64),
            DecoderDenseBlock(96, 32),
            nn.Conv2d(in_channels=56, out_channels=3, kernel_size=5, stride=1, padding=2),
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


# In[ ]:


import os
import torch
from torch.utils.data import DataLoader

BATCH_SIZE = 50
LR = 0.00005
NUM_WORKERS = 1

def train():
    start_time = time.time()

    dataset_a = FaceDataset("../../personA_images")
    dataset_b = FaceDataset("../../personB_images")

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

    for epoch in range(1, 100000):
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
        os.makedirs("checkpoints", exist_ok=True)
        os.makedirs(f"checkpoints/{epoch}", exist_ok=True)
        torch.save(encoder.state_dict(),   f"checkpoints/{epoch}/encoder.pth")
        torch.save(decoder_a.state_dict(), f"checkpoints/{epoch}/decoder_A.pth")
        torch.save(decoder_b.state_dict(), f"checkpoints/{epoch}/decoder_B.pth")
        save_preview(encoder, decoder_a, decoder_b, sample_a, sample_b, epoch)
        print("\nSaved/")


# In[ ]:


if __name__ == "__main__":
    train()

