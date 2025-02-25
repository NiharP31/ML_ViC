import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import cv2
import os
from PIL import Image
import matplotlib.pyplot as plt


class FrameDataset(Dataset):
    """Dataset for video frames"""
    
    def __init__(self, frames_dir=None, frames_list=None, transform=None):
        """
        Initialize dataset from directory or list of frames
        
        Args:
            frames_dir (str, optional): Directory containing frame images
            frames_list (list, optional): List of numpy arrays containing frames
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.transform = transform
        self.frames = []
        
        if frames_dir is not None:
            # Load frames from directory
            frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(('.jpg', '.png'))])
            for frame_file in frame_files:
                frame_path = os.path.join(frames_dir, frame_file)
                self.frames.append(Image.open(frame_path).convert('RGB'))
        
        elif frames_list is not None:
            # Use provided frames list
            for frame in frames_list:
                # Convert numpy array to PIL Image if needed
                if isinstance(frame, np.ndarray):
                    frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                self.frames.append(frame)
    
    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, idx):
        frame = self.frames[idx]
        
        if self.transform:
            frame = self.transform(frame)
            
        return frame


class Encoder(nn.Module):
    """Encoder part of the autoencoder"""
    
    def __init__(self, latent_dim=128):
        super(Encoder, self).__init__()
        
        # Simple CNN-based encoder
        self.encoder = nn.Sequential(
            # Input: 3 x 256 x 256 (assuming RGB frames resized to 256x256)
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),  # -> 32 x 128 x 128
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # -> 64 x 64 x 64
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # -> 128 x 32 x 32
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # -> 256 x 16 x 16
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            
            nn.Conv2d(256, latent_dim, kernel_size=4, stride=2, padding=1),  # -> latent_dim x 8 x 8
        )
    
    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    """Decoder part of the autoencoder"""
    
    def __init__(self, latent_dim=128):
        super(Decoder, self).__init__()
        
        # Simple CNN-based decoder (mirroring the encoder)
        self.decoder = nn.Sequential(
            # Input: latent_dim x 8 x 8
            nn.ConvTranspose2d(latent_dim, 256, kernel_size=4, stride=2, padding=1),  # -> 256 x 16 x 16
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # -> 128 x 32 x 32
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # -> 64 x 64 x 64
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # -> 32 x 128 x 128
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),  # -> 3 x 256 x 256
            nn.Sigmoid()  # Output values between 0 and 1
        )
    
    def forward(self, x):
        return self.decoder(x)


class Quantizer(nn.Module):
    """Quantizes the latent space representation"""
    
    def __init__(self, num_bits=8):
        super(Quantizer, self).__init__()
        self.num_bits = num_bits
        self.num_levels = 2 ** num_bits
    
    def forward(self, x):
        # Scale to [0, 1] range
        x_min = x.min()
        x_max = x.max()
        x_normalized = (x - x_min) / (x_max - x_min + 1e-8)
        
        # Quantize to specified number of bits
        x_quantized = torch.floor(x_normalized * (self.num_levels - 1) + 0.5) / (self.num_levels - 1)
        
        # Scale back to original range
        x_dequantized = x_quantized * (x_max - x_min) + x_min
        
        # Use straight-through estimator for gradients
        if self.training:
            return x + (x_dequantized - x).detach()
        else:
            return x_dequantized


class VideoAutoencoder(nn.Module):
    """Complete autoencoder for video frame compression"""
    
    def __init__(self, latent_dim=128, num_bits=8):
        super(VideoAutoencoder, self).__init__()
        
        self.encoder = Encoder(latent_dim)
        self.quantizer = Quantizer(num_bits)
        self.decoder = Decoder(latent_dim)
        
        self.latent_dim = latent_dim
        self.num_bits = num_bits
    
    def forward(self, x):
        # Encode
        latent = self.encoder(x)
        
        # Quantize
        latent_quantized = self.quantizer(latent)
        
        # Decode
        reconstructed = self.decoder(latent_quantized)
        
        return reconstructed, latent_quantized
    
    def compress(self, x):
        """Compress a batch of frames"""
        with torch.no_grad():
            latent = self.encoder(x)
            latent_quantized = self.quantizer(latent)
        return latent_quantized
    
    def decompress(self, latent_quantized):
        """Decompress a batch of latent representations"""
        with torch.no_grad():
            reconstructed = self.decoder(latent_quantized)
        return reconstructed
    
    def get_compression_ratio(self, frame_shape):
        """Calculate the theoretical compression ratio"""
        # Original size: H * W * 3 * 8 bits (assuming 8-bit RGB image)
        h, w = frame_shape[2], frame_shape[3]
        original_bits = h * w * 3 * 8
        
        # Compressed size: latent_shape * num_bits
        latent_h, latent_w = h // 32, w // 32  # Based on our architecture (5 layers with stride 2)
        compressed_bits = self.latent_dim * latent_h * latent_w * self.num_bits
        
        return original_bits / compressed_bits


def train_autoencoder(model, dataloader, num_epochs=10, learning_rate=0.001, device="cuda" if torch.cuda.is_available() else "cpu"):
    """Train the autoencoder model"""
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    mse_loss = nn.MSELoss()
    
    print(f"Training on device: {device}")
    print(f"Number of epochs: {num_epochs}")
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_idx, frames in enumerate(dataloader):
            frames = frames.to(device)
            
            # Forward pass
            reconstructed, _ = model(frames)
            
            # Calculate loss
            loss = mse_loss(reconstructed, frames)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.6f}")
        
        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_epoch_loss:.6f}")
    
    print("Training completed!")
    return model


def visualize_results(original_frames, reconstructed_frames, num_samples=5):
    """Visualize original vs reconstructed frames"""
    # Select random indices if we have more frames than num_samples
    if original_frames.shape[0] > num_samples:
        indices = np.random.choice(original_frames.shape[0], num_samples, replace=False)
    else:
        indices = range(original_frames.shape[0])
    
    plt.figure(figsize=(12, 4 * len(indices)))
    
    for i, idx in enumerate(indices):
        # Original frame
        plt.subplot(len(indices), 2, 2*i + 1)
        plt.imshow(original_frames[idx].cpu().permute(1, 2, 0).numpy())
        plt.title(f"Original Frame {idx}")
        plt.axis("off")
        
        # Reconstructed frame
        plt.subplot(len(indices), 2, 2*i + 2)
        plt.imshow(reconstructed_frames[idx].cpu().permute(1, 2, 0).numpy())
        plt.title(f"Reconstructed Frame {idx}")
        plt.axis("off")
    
    plt.tight_layout()
    plt.savefig("frame_reconstruction_comparison.png")
    plt.show()


def main():
    """Example usage of the frame compression autoencoder"""
    # Parameters
    frames_dir = "extracted_frames"  # From stage 1
    latent_dim = 64  # Size of the latent space
    num_bits = 8     # Number of bits for quantization
    batch_size = 8
    num_epochs = 5
    image_size = 256
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Transforms
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    
    # Create dataset and dataloader
    try:
        dataset = FrameDataset(frames_dir=frames_dir, transform=transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        print(f"Loaded {len(dataset)} frames from {frames_dir}")
        
        # Create model
        model = VideoAutoencoder(latent_dim=latent_dim, num_bits=num_bits)
        print(f"Created autoencoder with latent dimension: {latent_dim}, quantization bits: {num_bits}")
        
        # Calculate theoretical compression ratio
        example_shape = next(iter(dataloader)).shape
        compression_ratio = model.get_compression_ratio(example_shape)
        print(f"Theoretical compression ratio: {compression_ratio:.2f}x")
        
        # Train model
        model = train_autoencoder(model, dataloader, num_epochs=num_epochs, device=device)
        
        # Save model
        torch.save(model.state_dict(), "video_autoencoder.pth")
        print("Model saved to video_autoencoder.pth")
        
        # Visualize results on a few test frames
        test_loader = DataLoader(dataset, batch_size=10, shuffle=True)
        test_frames = next(iter(test_loader)).to(device)
        reconstructed_frames, _ = model(test_frames)
        
        visualize_results(test_frames, reconstructed_frames)
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()