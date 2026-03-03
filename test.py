from train import Encoder, Decoder
import os
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

EPOCH = 110
def denormalize(tensor):
    return tensor * 0.5 + 0.5

def to_numpy(tensor):
    return denormalize(tensor.squeeze(0)).clamp(0, 1).permute(1, 2, 0).cpu().numpy()

def swap_first_10(folder_path):
    device = "cuda"

    encoder = Encoder().to(device)
    decoder_a = Decoder().to(device)

    encoder.load_state_dict(torch.load(f"checkpoints/{EPOCH}/encoder.pth", map_location=device))
    decoder_a.load_state_dict(torch.load(f"checkpoints/{EPOCH}/decoder_B.pth", map_location=device))

    encoder.eval()
    decoder_a.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5],
                             [0.5, 0.5, 0.5])
    ])

    image_files = sorted(os.listdir(folder_path))[1000:1010]

    originals = []
    fakes = []

    with torch.no_grad():
        for file in image_files:
            path = os.path.join(folder_path, file)

            img = Image.open(path).convert("RGB")
            sample = transform(img).unsqueeze(0).to(device)

            latent = encoder(sample)
            fake = decoder_a(latent)

            originals.append(to_numpy(sample))
            fakes.append(to_numpy(fake))

    fig, axes = plt.subplots(10, 2, figsize=(6, 20))

    for i in range(10):
        axes[i, 0].imshow(originals[i])
        axes[i, 0].axis("off")

        axes[i, 1].imshow(fakes[i])
        axes[i, 1].axis("off")

    plt.tight_layout()
    plt.show()
    plt.close()

swap_first_10("personA")