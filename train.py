from sympy.printing.pytorch import torch
from torch import nn
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

class FaceDataset(Dataset):
    def __init__(self, folder_path):
        self.image_paths = []

        for filename in os.listdir(folder_path):
            if filename.lower().endswith('.png'):
                full_path = os.path.join(folder_path, filename)
                self.image_paths.append(full_path)

        if len(self.image_paths) == 0:
            raise IOError("No files found")

        print(str(len(self.image_paths)) + "images imported")

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img = Image.open(self.image_paths[index]).convert('RGB')
        return self.transform(img)

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class DeconvolutionBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch * 4, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.PixelShuffle(upscale_factor=2)
        )

    def forward(self, x):
        return self.block(x)

import torchvision

class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        vgg = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT).features
        blocks = [vgg[:4], vgg[4:9]]
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target, feature_layers=[0, 1]):
        if input.shape[1] != 3:
            raise ValueError("Expected 3-channel input, got {} channels".format(input.shape[1]))
        if input.min() < 0 or target.min() < 0: # denormalization
            input = input * 0.5 + 0.5
            target = target * 0.5 + 0.5
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
        return loss

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.convPath = nn.Sequential(
            ConvBlock(3, 128),
            ConvBlock(128, 256),
            ConvBlock(256, 512),
            ConvBlock(512, 1024),
        )
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(1024 * 4 * 4, 1024)
        self.dense2 = nn.Linear(1024, 4 * 4 * 1024)
        self.deconvPath = DeconvolutionBlock(1024, 512)


    def forward(self, x):
        out = self.convPath(x)
        print(out.shape)
        out = self.flatten(out)
        out = self.dense1(out)
        out = self.dense2(out)
        out = out.view(out.size(0), 1024, 8, 8)
        out = self.deconvPath(out)
        return out

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            DeconvolutionBlock(512, 256),
            DeconvolutionBlock(256, 128),
            DeconvolutionBlock(128, 64),
            nn.Conv2d(in_channels=32, out_channels=3, kernel_size=5, stride=1, padding=2),
        )

    def forward(self, x):
        return self.net(x)

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

    fig, axes = plt.subplots(1, 6, figsize=(24, 4))
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

import os
import torch
from torch.utils.data import DataLoader

BATCH_SIZE = 27
LR = 5e-5
NUM_WORKERS = 1

def train(load = False, epoch = 0):
    start_time = time.time()

    dataset_a = FaceDataset("personA")
    dataset_b = FaceDataset("personB")

    loader_a = DataLoader(dataset_a, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=NUM_WORKERS)
    loader_b = DataLoader(dataset_b, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=NUM_WORKERS)

    sample_a = next(iter(loader_a)).to("cuda")
    sample_b = next(iter(loader_b)).to("cuda")

    encoder = Encoder().to("cuda")
    decoder_a = Decoder().to("cuda")
    decoder_b = Decoder().to("cuda")

    if load:
        encoder.load_state_dict(torch.load(f"checkpoints/{epoch}/encoder.pth"))
        decoder_a.load_state_dict(torch.load(f"checkpoints/{epoch}/decoder_A.pth"))
        decoder_b.load_state_dict(torch.load(f"checkpoints/{epoch}/decoder_B.pth"))

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) +
        list(decoder_a.parameters()) +
        list(decoder_b.parameters()),
        lr=LR
    )

    loss_fn = nn.L1Loss()

    for epoch in range(epoch + 1, 1000):
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
        torch.save(encoder.state_dict(), f"checkpoints/{epoch}/encoder.pth")
        torch.save(decoder_a.state_dict(), f"checkpoints/{epoch}/decoder_A.pth")
        torch.save(decoder_b.state_dict(), f"checkpoints/{epoch}/decoder_B.pth")
        save_preview(encoder, decoder_a, decoder_b, sample_a, sample_b, epoch)

if __name__ == "__main__":
    train(False)
