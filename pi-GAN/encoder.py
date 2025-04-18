import torch
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image

class ImageEncoder(nn.Module):
    """
    Modified image encoder that maps a 256x256 input image to a latent representation,
    and also returns a skip connection feature map.
    """
    def __init__(self, emb_dim=256):
        super(ImageEncoder, self).__init__()
        self.emb_dim = emb_dim
        
        # Define the encoder body up to the skip connection extraction:
        self.encoder_body = nn.Sequential(
            # First block: 256x256 -> 128x128
            nn.Conv2d(3, 64, kernel_size=3, padding=1),    # [B, 64, 256, 256]
            nn.ReLU(),
            nn.MaxPool2d(2),                                # [B, 64, 128, 128]
            
            # Second block: 128x128 -> 64x64
            nn.Conv2d(64, 128, kernel_size=3, padding=1),   # [B, 128, 128, 128]
            nn.ReLU(),
            nn.MaxPool2d(2),                                # [B, 128, 64, 64]
            
            # Third block: 64x64 -> 32x32
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # [B, 256, 64, 64]
            nn.ReLU(),
            nn.MaxPool2d(2),                                # [B, 256, 32, 32]
            
            # EXTRA block: deeper feature extraction at 32x32 resolution
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # [B, 256, 32, 32]
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # [B, 256, 32, 32]
            nn.ReLU(),
            
            # Projection to embedding dimension – produces skip feature map
            nn.Conv2d(256, emb_dim, kernel_size=3, padding=1),  # [B, emb_dim, 32, 32]
            nn.ReLU()
        )
        
        # Global pooling to generate latent vector from the skip feature map.
        self.avgpool = nn.AdaptiveAvgPool2d(1)  # [B, emb_dim, 1, 1]

    def forward(self, od_image, return_skip=False):
        # Run the encoder body to get the feature map (skip connection)
        x = self.encoder_body(od_image)  # shape: [B, emb_dim, 32, 32]
        # Pool to obtain the latent vector
        latent = self.avgpool(x).squeeze(-1).squeeze(-1)  # shape: [B, emb_dim]
        if return_skip:
            return latent, x  # Return both latent vector and skip connection feature map
        return latent

class ImageDecoder(nn.Module):
    """
    Decoder that reconstructs a 256x256 RGB image from a latent vector,
    incorporating a skip connection from the encoder.
    """
    def __init__(self, latent_dim=256, emb_dim=256):
        super(ImageDecoder, self).__init__()
        # Project latent vector to a feature map of shape [B, emb_dim, 32, 32]
        self.fc = nn.Linear(latent_dim, emb_dim * 32 * 32)
        
        self.deconv_layers = nn.Sequential(
            # Upsample from 32x32 -> 64x64
            nn.ConvTranspose2d(emb_dim, 128, kernel_size=4, stride=2, padding=1),  # [B, 128, 64, 64]
            nn.ReLU(),
            # Extra convolution block at resolution 64x64
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            
            # Upsample from 64x64 -> 128x128
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),       # [B, 64, 128, 128]
            nn.ReLU(),
            # Extra convolution block at resolution 128x128
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            
            # Upsample from 128x128 -> 256x256
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),        # [B, 32, 256, 256]
            nn.ReLU(),
            # Extra convolution block at resolution 256x256
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            
            # Final convolution: map to RGB channels
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()  # Use Tanh if your images are normalized to [-1, 1]
        )

    def forward(self, latent, skip_connection=None):
        # Project the latent vector back into a spatial feature map.
        x = self.fc(latent)
        x = x.view(x.size(0), -1, 32, 32)
        # Incorporate the skip connection via element-wise addition if provided.
        if skip_connection is not None:
            x = x + skip_connection
        x = self.deconv_layers(x)
        return x

class AutoEncoder(nn.Module):
    """
    Autoencoder that combines the ImageEncoder and ImageDecoder with skip connections.
    """
    def __init__(self, latent_dim=256, emb_dim=256):
        super(AutoEncoder, self).__init__()
        self.encoder = ImageEncoder(emb_dim=emb_dim)
        self.decoder = ImageDecoder(latent_dim=latent_dim, emb_dim=emb_dim)
    
    def forward(self, x):
        # Return both the latent vector and the intermediate skip feature map.
        latent, skip = self.encoder(x, return_skip=True)
        # Pass the skip connection into the decoder.
        recon = self.decoder(latent, skip_connection=skip)
        return recon

class RetinalImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        """
        Args:
            image_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.image_dir = image_dir
        # List all image files in the directory (you can adjust the extensions as needed)
        self.image_files = [os.path.join(image_dir, f)
                            for f in os.listdir(image_dir)
                            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Read the image using PIL
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image