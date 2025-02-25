import cv2
import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt
import os
import tempfile
from torch.utils.data import DataLoader
from pathlib import Path
import pandas as pd
import time
import torchvision.transforms as transforms

# Import from our previous modules
# Assuming these are saved in separate files: extract_frames.py and frame_autoencoder.py
from frame_extractor import extract_frames
from frame_encoder import VideoAutoencoder, FrameDataset


def calculate_psnr(original, compressed):
    """
    Calculate Peak Signal-to-Noise Ratio between original and compressed images
    
    Args:
        original: Original image (numpy array or torch tensor)
        compressed: Compressed image (numpy array or torch tensor)
        
    Returns:
        float: PSNR value in dB
    """
    # Convert torch tensors to numpy if needed
    if isinstance(original, torch.Tensor):
        original = original.detach().cpu().numpy()
        if original.shape[0] == 3:  # If channels first (C, H, W)
            original = np.transpose(original, (1, 2, 0))
    
    if isinstance(compressed, torch.Tensor):
        compressed = compressed.detach().cpu().numpy()
        if compressed.shape[0] == 3:  # If channels first (C, H, W)
            compressed = np.transpose(compressed, (1, 2, 0))
    
    # Ensure values are in the correct range for PSNR calculation
    if original.max() <= 1.0:
        original = original * 255.0
    if compressed.max() <= 1.0:
        compressed = compressed * 255.0
    
    # Convert to uint8 for consistent PSNR calculation
    original = original.astype(np.uint8)
    compressed = compressed.astype(np.uint8)
    
    # Calculate PSNR
    return psnr(original, compressed)


def calculate_ssim(original, compressed):
    """
    Calculate Structural Similarity Index between original and compressed images
    
    Args:
        original: Original image (numpy array or torch tensor)
        compressed: Compressed image (numpy array or torch tensor)
        
    Returns:
        float: SSIM value (between -1 and 1, where 1 means identical images)
    """
    # Convert torch tensors to numpy if needed
    if isinstance(original, torch.Tensor):
        original = original.detach().cpu().numpy()
        if original.shape[0] == 3:  # If channels first (C, H, W)
            original = np.transpose(original, (1, 2, 0))
    
    if isinstance(compressed, torch.Tensor):
        compressed = compressed.detach().cpu().numpy()
        if compressed.shape[0] == 3:  # If channels first (C, H, W)
            compressed = np.transpose(compressed, (1, 2, 0))
    
    # Ensure values are in the correct range for SSIM calculation
    if original.max() <= 1.0:
        original = original * 255.0
    if compressed.max() <= 1.0:
        compressed = compressed * 255.0
    
    # Convert to uint8 for consistent SSIM calculation
    original = original.astype(np.uint8)
    compressed = compressed.astype(np.uint8)
    
    # Calculate SSIM (multichannel for RGB images)
    return ssim(original, compressed, multichannel=True, channel_axis=2)


def compress_with_h264(frames, output_path=None, crf=23, fps=30):
    """
    Compress frames using H.264 codec
    
    Args:
        frames: List of frames (numpy arrays) or directory of frame images
        output_path: Path to save the compressed video (if None, uses a temp file)
        crf: Constant Rate Factor (quality, lower is better, 18-28 is typical)
        fps: Frames per second for the output video
        
    Returns:
        tuple: (List of compressed frames, compression ratio, file size in bytes)
    """
    # Create temp file if output_path is not provided
    if output_path is None:
        temp_dir = tempfile.gettempdir()
        output_path = os.path.join(temp_dir, "compressed_h264.mp4")
    
    # Load frames if a directory is provided
    if isinstance(frames, str) and os.path.isdir(frames):
        frame_files = sorted([os.path.join(frames, f) for f in os.listdir(frames) 
                              if f.endswith(('.jpg', '.png'))])
        loaded_frames = []
        for frame_file in frame_files:
            frame = cv2.imread(frame_file)
            if frame is not None:
                loaded_frames.append(frame)
        frames = loaded_frames
    
    if not frames:
        raise ValueError("No frames provided or found in directory")
    
    # Get frame dimensions from the first frame
    height, width = frames[0].shape[:2]
    
    # Calculate original size (uncompressed)
    original_size = sum(frame.nbytes for frame in frames)
    
    # Set up H.264 codec
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # or 'H264' depending on platform
    
    # Create video writer
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Add all frames to the video
    for frame in frames:
        out.write(frame)
    
    # Release the writer
    out.release()
    
    # Get file size of the compressed video
    compressed_size = os.path.getsize(output_path)
    compression_ratio = original_size / compressed_size
    
    # Read back the compressed video for quality comparison
    cap = cv2.VideoCapture(output_path)
    compressed_frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        compressed_frames.append(frame)
    
    cap.release()
    
    return compressed_frames, compression_ratio, compressed_size


def evaluate_autoencoder(model, dataloader, device):
    """
    Evaluate autoencoder performance on a dataset
    
    Args:
        model: Trained VideoAutoencoder model
        dataloader: DataLoader with frames to evaluate
        device: Device to run evaluation on ('cuda' or 'cpu')
        
    Returns:
        tuple: (List of (original, reconstructed) frame pairs, 
                average PSNR, average SSIM, compression ratio)
    """
    model.eval()
    model = model.to(device)
    
    all_originals = []
    all_reconstructed = []
    psnr_values = []
    ssim_values = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            batch = batch.to(device)
            
            # Get reconstructed frames
            reconstructed, latent = model(batch)
            
            # Calculate PSNR and SSIM for each frame in the batch
            for i in range(batch.size(0)):
                original = batch[i].cpu()
                recon = reconstructed[i].cpu()
                
                # Convert to numpy and appropriate format for metrics
                original_np = original.permute(1, 2, 0).numpy()
                recon_np = recon.permute(1, 2, 0).numpy()
                
                # Calculate metrics
                psnr_val = calculate_psnr(original_np, recon_np)
                ssim_val = calculate_ssim(original_np, recon_np)
                
                psnr_values.append(psnr_val)
                ssim_values.append(ssim_val)
                
                # Store frames for later visualization
                all_originals.append(original)
                all_reconstructed.append(recon)
    
    # Calculate averages
    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)
    
    # Get compression ratio
    # Handle both batch and single image shapes
    example_shape = batch[0].shape
    # For a single image with shape [channels, height, width]
    if len(example_shape) == 3:
        h, w = example_shape[1], example_shape[2]
    # For a batch with shape [batch_size, channels, height, width]
    else:
        h, w = example_shape[2], example_shape[3]
        
    # Calculate compression ratio directly here
    # Original size: H * W * 3 * 8 bits (assuming 8-bit RGB image)
    original_bits = h * w * 3 * 8
    
    # Compressed size: latent_shape * num_bits
    latent_h, latent_w = h // 32, w // 32  # Based on our architecture (5 layers with stride 2)
    compressed_bits = model.latent_dim * latent_h * latent_w * model.num_bits
    
    compression_ratio = original_bits / compressed_bits
    
    return list(zip(all_originals, all_reconstructed)), avg_psnr, avg_ssim, compression_ratio


def evaluate_h264(frames_dir, crf=23):
    """
    Evaluate H.264 compression on frames
    
    Args:
        frames_dir: Directory containing frame images
        crf: Constant Rate Factor for H.264 compression
        
    Returns:
        tuple: (List of (original, compressed) frame pairs, 
                average PSNR, average SSIM, compression ratio)
    """
    # Load original frames
    frame_files = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) 
                          if f.endswith(('.jpg', '.png'))])
    
    original_frames = []
    for frame_file in frame_files:
        frame = cv2.imread(frame_file)
        if frame is not None:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            original_frames.append(frame)
    
    # Compress frames with H.264
    compressed_frames, compression_ratio, _ = compress_with_h264(original_frames, crf=crf)
    
    # Match frames if counts differ
    min_frames = min(len(original_frames), len(compressed_frames))
    original_frames = original_frames[:min_frames]
    compressed_frames = compressed_frames[:min_frames]
    
    # Calculate metrics
    psnr_values = []
    ssim_values = []
    frame_pairs = []
    
    for orig, comp in zip(original_frames, compressed_frames):
        # Convert BGR to RGB for metrics calculation
        orig_rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
        comp_rgb = cv2.cvtColor(comp, cv2.COLOR_BGR2RGB)
        
        psnr_val = calculate_psnr(orig_rgb, comp_rgb)
        ssim_val = calculate_ssim(orig_rgb, comp_rgb)
        
        psnr_values.append(psnr_val)
        ssim_values.append(ssim_val)
        frame_pairs.append((orig_rgb, comp_rgb))
    
    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)
    
    return frame_pairs, avg_psnr, avg_ssim, compression_ratio


def visualize_comparison(autoencoder_results, h264_results, num_samples=3):
    """
    Visualize comparison between original, autoencoder, and H.264 compressed frames
    
    Args:
        autoencoder_results: Results from evaluate_autoencoder
        h264_results: Results from evaluate_h264
        num_samples: Number of frame samples to display
    """
    # Unpack results
    ae_frame_pairs, ae_psnr, ae_ssim, ae_ratio = autoencoder_results
    h264_frame_pairs, h264_psnr, h264_ssim, h264_ratio = h264_results
    
    # Select a few sample frames
    num_samples = min(num_samples, len(ae_frame_pairs), len(h264_frame_pairs))
    sample_indices = np.linspace(0, min(len(ae_frame_pairs), len(h264_frame_pairs))-1, num_samples, dtype=int)
    
    # Create figure
    plt.figure(figsize=(15, 5 * num_samples))
    
    for i, idx in enumerate(sample_indices):
        # Get sample frames
        orig_ae, recon_ae = ae_frame_pairs[idx]
        orig_h264, comp_h264 = h264_frame_pairs[idx]
        
        # Convert PyTorch tensors to numpy if needed
        if isinstance(orig_ae, torch.Tensor):
            orig_ae = orig_ae.permute(1, 2, 0).numpy()
        if isinstance(recon_ae, torch.Tensor):
            recon_ae = recon_ae.permute(1, 2, 0).numpy()
        
        # Plot original
        plt.subplot(num_samples, 3, 3*i + 1)
        plt.imshow(orig_ae)
        plt.title(f"Original")
        plt.axis('off')
        
        # Plot autoencoder reconstruction
        plt.subplot(num_samples, 3, 3*i + 2)
        plt.imshow(recon_ae)
        ae_frame_psnr = calculate_psnr(orig_ae, recon_ae)
        ae_frame_ssim = calculate_ssim(orig_ae, recon_ae)
        plt.title(f"Autoencoder\nPSNR: {ae_frame_psnr:.2f}dB, SSIM: {ae_frame_ssim:.3f}")
        plt.axis('off')
        
        # Plot H.264 compression
        plt.subplot(num_samples, 3, 3*i + 3)
        plt.imshow(comp_h264)
        h264_frame_psnr = calculate_psnr(orig_h264, comp_h264)
        h264_frame_ssim = calculate_ssim(orig_h264, comp_h264)
        plt.title(f"H.264\nPSNR: {h264_frame_psnr:.2f}dB, SSIM: {h264_frame_ssim:.3f}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig("compression_comparison.png")
    plt.close()
    
    # Create summary table
    data = {
        'Method': ['Autoencoder', 'H.264'],
        'Avg PSNR (dB)': [ae_psnr, h264_psnr],
        'Avg SSIM': [ae_ssim, h264_ssim],
        'Compression Ratio': [ae_ratio, h264_ratio]
    }
    
    summary_df = pd.DataFrame(data)
    print("\nCompression Comparison Summary:")
    print(summary_df.to_string(index=False))
    
    # Create bar plot of metrics
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.bar(['Autoencoder', 'H.264'], [ae_psnr, h264_psnr])
    plt.title('PSNR Comparison (higher is better)')
    plt.ylabel('PSNR (dB)')
    
    plt.subplot(1, 2, 2)
    plt.bar(['Autoencoder', 'H.264'], [ae_ssim, h264_ssim])
    plt.title('SSIM Comparison (higher is better)')
    plt.ylabel('SSIM')
    
    plt.tight_layout()
    plt.savefig("metrics_comparison.png")
    plt.close()


def main():
    """Example usage of the compression evaluation"""
    # Parameters
    frames_dir = "extracted_frames"  # From stage 1
    model_path = "video_autoencoder.pth"  # From stage 2
    batch_size = 16
    image_size = 256
    latent_dim = 64
    num_bits = 8
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    h264_crf = 23  # H.264 quality parameter (lower is better, 18-28 is typical)
    
    try:
        print("Starting compression quality evaluation...")
        
        # Create output directory for results
        os.makedirs("evaluation_results", exist_ok=True)
        
        # Setup for autoencoder evaluation
        # Load the trained model
        model = VideoAutoencoder(latent_dim=latent_dim, num_bits=num_bits)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()
        
        # Create dataset and dataloader
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
        
        dataset = FrameDataset(frames_dir=frames_dir, transform=transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        print(f"Loaded {len(dataset)} frames from {frames_dir}")
        
        # Evaluate autoencoder
        print("Evaluating autoencoder compression...")
        start_time = time.time()
        autoencoder_results = evaluate_autoencoder(model, dataloader, device)
        ae_time = time.time() - start_time
        print(f"Autoencoder evaluation completed in {ae_time:.2f} seconds")
        print(f"Autoencoder PSNR: {autoencoder_results[1]:.2f} dB")
        print(f"Autoencoder SSIM: {autoencoder_results[2]:.4f}")
        print(f"Autoencoder compression ratio: {autoencoder_results[3]:.2f}x")
        
        # Evaluate H.264
        print("Evaluating H.264 compression...")
        start_time = time.time()
        h264_results = evaluate_h264(frames_dir, crf=h264_crf)
        h264_time = time.time() - start_time
        print(f"H.264 evaluation completed in {h264_time:.2f} seconds")
        print(f"H.264 PSNR: {h264_results[1]:.2f} dB")
        print(f"H.264 SSIM: {h264_results[2]:.4f}")
        print(f"H.264 compression ratio: {h264_results[3]:.2f}x")
        
        # Visualize comparison
        print("Generating visualization...")
        visualize_comparison(autoencoder_results, h264_results, num_samples=3)
        print("Visualizations saved to metrics_comparison.png and compression_comparison.png")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")


if __name__ == "__main__":
    main()