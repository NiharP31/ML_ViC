import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers
import os
import pandas as pd
from pathlib import Path
import time
import matplotlib.gridspec as gridspec
from tqdm import tqdm
import seaborn as sns
import sys

# Import from our previous modules
from frame_extractor import extract_frames
from frame_encoder import VideoAutoencoder, FrameDataset
from compression_evaluation import (
    calculate_psnr, calculate_ssim, compress_with_h264, 
    evaluate_autoencoder, evaluate_h264
)


class VideoComparisonVisualizer:
    """Class for visualizing video compression results"""
    
    def __init__(self, original_frames, ae_frames, h264_frames, metrics=None):
        """
        Initialize visualizer with frame data
        
        Args:
            original_frames: List of original video frames
            ae_frames: List of autoencoder compressed frames
            h264_frames: List of H.264 compressed frames
            metrics: Dictionary with metrics data (optional)
        """
        self.original_frames = original_frames
        self.ae_frames = ae_frames
        self.h264_frames = h264_frames
        self.metrics = metrics if metrics else {}
        
        # Ensure all frame lists have the same length
        min_length = min(len(original_frames), len(ae_frames), len(h264_frames))
        self.original_frames = self.original_frames[:min_length]
        self.ae_frames = self.ae_frames[:min_length]
        self.h264_frames = self.h264_frames[:min_length]
        
        # Calculate frame-by-frame metrics if not provided
        if 'psnr_ae' not in self.metrics:
            self.calculate_frame_metrics()
    
    def calculate_frame_metrics(self):
        """Calculate PSNR and SSIM for each frame"""
        psnr_ae = []
        ssim_ae = []
        psnr_h264 = []
        ssim_h264 = []
        
        print("Calculating frame-by-frame metrics...")
        for i in tqdm(range(len(self.original_frames))):
            # Get frames
            orig = self.original_frames[i]
            ae = self.ae_frames[i]
            
            # Make sure we have a valid H.264 frame
            if i < len(self.h264_frames):
                h264 = self.h264_frames[i]
            else:
                # Use a blank frame if H.264 frame is missing
                print(f"Warning: Missing H.264 frame at index {i}")
                h264 = np.zeros_like(orig.numpy() if isinstance(orig, torch.Tensor) else orig)
            
            # Convert to numpy arrays if they are tensors
            if isinstance(orig, torch.Tensor):
                orig = orig.permute(1, 2, 0).numpy()
            if isinstance(ae, torch.Tensor):
                ae = ae.permute(1, 2, 0).numpy()
            
            # Ensure all frames have the same dimensions
            h264_shape = h264.shape[:2]
            orig_shape = orig.shape[:2]
            
            if h264_shape != orig_shape:
                print(f"Resizing H.264 frame from {h264_shape} to {orig_shape}")
                h264 = cv2.resize(h264, (orig.shape[1], orig.shape[0]))
            
            # Calculate metrics
            psnr_ae.append(calculate_psnr(orig, ae))
            ssim_ae.append(calculate_ssim(orig, ae))
            psnr_h264.append(calculate_psnr(orig, h264))
            ssim_h264.append(calculate_ssim(orig, h264))
        
        self.metrics['psnr_ae'] = psnr_ae
        self.metrics['ssim_ae'] = ssim_ae
        self.metrics['psnr_h264'] = psnr_h264
        self.metrics['ssim_h264'] = ssim_h264
    
    def create_side_by_side_image(self, frame_idx):
        """Create a side-by-side comparison image for a specific frame"""
        # Get frames
        orig = self.original_frames[frame_idx]
        ae = self.ae_frames[frame_idx]
        h264 = self.h264_frames[frame_idx]
        
        # Convert to numpy arrays if they are tensors
        if isinstance(orig, torch.Tensor):
            orig = orig.permute(1, 2, 0).numpy()
        if isinstance(ae, torch.Tensor):
            ae = ae.permute(1, 2, 0).numpy()
        
        # Ensure values are in the range [0, 1]
        if orig.max() > 1.0:
            orig = orig / 255.0
        if ae.max() > 1.0:
            ae = ae / 255.0
        if h264.max() > 1.0:
            h264 = h264 / 255.0
        
        # Get metrics for this frame
        psnr_ae = self.metrics['psnr_ae'][frame_idx]
        ssim_ae = self.metrics['ssim_ae'][frame_idx]
        psnr_h264 = self.metrics['psnr_h264'][frame_idx]
        ssim_h264 = self.metrics['ssim_h264'][frame_idx]
        
        # Create figure
        fig = plt.figure(figsize=(18, 10))
        gs = gridspec.GridSpec(2, 3, height_ratios=[4, 1])
        
        # Original frame
        ax1 = plt.subplot(gs[0, 0])
        ax1.imshow(orig)
        ax1.set_title("Original Frame")
        ax1.axis('off')
        
        # Autoencoder frame
        ax2 = plt.subplot(gs[0, 1])
        ax2.imshow(ae)
        ax2.set_title(f"Autoencoder\nPSNR: {psnr_ae:.2f}dB, SSIM: {ssim_ae:.3f}")
        ax2.axis('off')
        
        # H.264 frame
        ax3 = plt.subplot(gs[0, 2])
        ax3.imshow(h264)
        ax3.set_title(f"H.264\nPSNR: {psnr_h264:.2f}dB, SSIM: {ssim_h264:.3f}")
        ax3.axis('off')
        
        # Metrics over time
        ax4 = plt.subplot(gs[1, :])
        
        # Plot metrics
        frame_indices = range(len(self.metrics['psnr_ae']))
        ax4.plot(frame_indices, self.metrics['psnr_ae'], 'b-', label='Autoencoder PSNR')
        ax4.plot(frame_indices, self.metrics['psnr_h264'], 'r-', label='H.264 PSNR')
        ax4.axvline(x=frame_idx, color='k', linestyle='--')
        ax4.set_xlabel('Frame Number')
        ax4.set_ylabel('PSNR (dB)')
        ax4.legend(loc='upper right')
        ax4.grid(True, alpha=0.3)
        
        # Mark current frame
        ax4.scatter([frame_idx], [self.metrics['psnr_ae'][frame_idx]], c='blue', s=100, zorder=5)
        ax4.scatter([frame_idx], [self.metrics['psnr_h264'][frame_idx]], c='red', s=100, zorder=5)
        
        # Add compression stats as text
        if 'ae_ratio' in self.metrics and 'h264_ratio' in self.metrics:
            fig.text(0.01, 0.01, 
                    f"Compression Ratios - Autoencoder: {self.metrics['ae_ratio']:.2f}x, H.264: {self.metrics['h264_ratio']:.2f}x", 
                    fontsize=12)
        
        plt.tight_layout()
        return fig
    
    def save_comparison_video(self, output_path, fps=15, dpi=100, frames_to_include=None):
        """
        Create an animated video comparison
        
        Args:
            output_path: Path to save the output video
            fps: Frames per second for the animation
            dpi: Dots per inch for the output
            frames_to_include: List of frame indices to include (default: all)
        """
        if frames_to_include is None:
            frames_to_include = range(len(self.original_frames))
        
        # Check if ffmpeg is available
        try:
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            writers = writers.list()  # Get available writers
            
            if 'ffmpeg' not in writers:
                print("Warning: ffmpeg writer not available. Using pillow writer instead.")
                if 'pillow' in writers:
                    Writer = writers['pillow']
                else:
                    print("Error: No suitable video writer available. Skipping video generation.")
                    return
            else:
                Writer = writers['ffmpeg']
                
            writer = Writer(fps=fps, metadata=dict(artist='ML Video Compression'), bitrate=1800)
        except Exception as e:
            print(f"Warning: Could not initialize video writer: {e}")
            print("Skipping video generation. You can still view individual frame comparisons.")
            return
        
        # Create figure for animation
        fig = plt.figure(figsize=(18, 10))
        
        # Create initial state
        frame_idx = 0
        
        # Define update function for animation
        def update(frame_idx):
            plt.clf()
            # Get frames
            orig = self.original_frames[frame_idx]
            ae = self.ae_frames[frame_idx]
            h264 = self.h264_frames[frame_idx]
            
            # Convert to numpy arrays if they are tensors
            if isinstance(orig, torch.Tensor):
                orig = orig.permute(1, 2, 0).numpy()
            if isinstance(ae, torch.Tensor):
                ae = ae.permute(1, 2, 0).numpy()
            
            # Ensure values are in the range [0, 1]
            if orig.max() > 1.0:
                orig = orig / 255.0
            if ae.max() > 1.0:
                ae = ae / 255.0
            if h264.max() > 1.0:
                h264 = h264 / 255.0
            
            # Get metrics for this frame
            psnr_ae = self.metrics['psnr_ae'][frame_idx]
            ssim_ae = self.metrics['ssim_ae'][frame_idx]
            psnr_h264 = self.metrics['psnr_h264'][frame_idx]
            ssim_h264 = self.metrics['ssim_h264'][frame_idx]
            
            # Create gridspec
            gs = gridspec.GridSpec(2, 3, height_ratios=[4, 1])
            
            # Original frame
            ax1 = plt.subplot(gs[0, 0])
            ax1.imshow(orig)
            ax1.set_title("Original Frame")
            ax1.axis('off')
            
            # Autoencoder frame
            ax2 = plt.subplot(gs[0, 1])
            ax2.imshow(ae)
            ax2.set_title(f"Autoencoder\nPSNR: {psnr_ae:.2f}dB, SSIM: {ssim_ae:.3f}")
            ax2.axis('off')
            
            # H.264 frame
            ax3 = plt.subplot(gs[0, 2])
            ax3.imshow(h264)
            ax3.set_title(f"H.264\nPSNR: {psnr_h264:.2f}dB, SSIM: {ssim_h264:.3f}")
            ax3.axis('off')
            
            # Metrics over time
            ax4 = plt.subplot(gs[1, :])
            
            # Plot metrics
            all_frames = range(len(self.metrics['psnr_ae']))
            ax4.plot(all_frames, self.metrics['psnr_ae'], 'b-', label='Autoencoder PSNR')
            ax4.plot(all_frames, self.metrics['psnr_h264'], 'r-', label='H.264 PSNR')
            ax4.axvline(x=frame_idx, color='k', linestyle='--')
            ax4.set_xlabel('Frame Number')
            ax4.set_ylabel('PSNR (dB)')
            ax4.legend(loc='upper right')
            ax4.grid(True, alpha=0.3)
            
            # Mark current frame
            ax4.scatter([frame_idx], [self.metrics['psnr_ae'][frame_idx]], c='blue', s=100, zorder=5)
            ax4.scatter([frame_idx], [self.metrics['psnr_h264'][frame_idx]], c='red', s=100, zorder=5)
            
            # Add compression stats as text
            if 'ae_ratio' in self.metrics and 'h264_ratio' in self.metrics:
                plt.figtext(0.01, 0.01, 
                        f"Compression Ratios - Autoencoder: {self.metrics['ae_ratio']:.2f}x, H.264: {self.metrics['h264_ratio']:.2f}x", 
                        fontsize=12)
            
            plt.tight_layout()
            return []
        
        # Create animation
        print(f"Creating animation with {len(frames_to_include)} frames...")
        anim = FuncAnimation(fig, update, frames=frames_to_include, blit=True)
        
        # Save animation
        print(f"Saving animation to {output_path}...")
        anim.save(output_path, writer=writer, dpi=dpi)
        plt.close()
    
    def generate_summary_report(self, output_dir="results"):
        """Generate summary report with metrics visualizations"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a dataframe with all metrics
        metrics_df = pd.DataFrame({
            'Frame': range(len(self.original_frames)),
            'Autoencoder PSNR': self.metrics['psnr_ae'],
            'H.264 PSNR': self.metrics['psnr_h264'],
            'Autoencoder SSIM': self.metrics['ssim_ae'],
            'H.264 SSIM': self.metrics['ssim_h264']
        })
        
        # Save metrics to CSV
        metrics_csv_path = os.path.join(output_dir, "metrics.csv")
        metrics_df.to_csv(metrics_csv_path, index=False)
        print(f"Saved metrics to {metrics_csv_path}")
        
        # Create metrics over time plot
        plt.figure(figsize=(12, 10))
        
        # PSNR over time
        plt.subplot(2, 1, 1)
        plt.plot(metrics_df['Frame'], metrics_df['Autoencoder PSNR'], 'b-', label='Autoencoder')
        plt.plot(metrics_df['Frame'], metrics_df['H.264 PSNR'], 'r-', label='H.264')
        plt.xlabel('Frame Number')
        plt.ylabel('PSNR (dB)')
        plt.title('PSNR Comparison Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # SSIM over time
        plt.subplot(2, 1, 2)
        plt.plot(metrics_df['Frame'], metrics_df['Autoencoder SSIM'], 'b-', label='Autoencoder')
        plt.plot(metrics_df['Frame'], metrics_df['H.264 SSIM'], 'r-', label='H.264')
        plt.xlabel('Frame Number')
        plt.ylabel('SSIM')
        plt.title('SSIM Comparison Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        metrics_plot_path = os.path.join(output_dir, "metrics_over_time.png")
        plt.savefig(metrics_plot_path)
        plt.close()
        print(f"Saved metrics plot to {metrics_plot_path}")
        
        # Create histogram of metrics
        plt.figure(figsize=(12, 10))
        
        # PSNR histogram
        plt.subplot(2, 2, 1)
        sns.histplot(metrics_df['Autoencoder PSNR'], kde=True, color='blue')
        plt.xlabel('PSNR (dB)')
        plt.title('Autoencoder PSNR Distribution')
        
        plt.subplot(2, 2, 2)
        sns.histplot(metrics_df['H.264 PSNR'], kde=True, color='red')
        plt.xlabel('PSNR (dB)')
        plt.title('H.264 PSNR Distribution')
        
        # SSIM histogram
        plt.subplot(2, 2, 3)
        sns.histplot(metrics_df['Autoencoder SSIM'], kde=True, color='blue')
        plt.xlabel('SSIM')
        plt.title('Autoencoder SSIM Distribution')
        
        plt.subplot(2, 2, 4)
        sns.histplot(metrics_df['H.264 SSIM'], kde=True, color='red')
        plt.xlabel('SSIM')
        plt.title('H.264 SSIM Distribution')
        
        plt.tight_layout()
        hist_path = os.path.join(output_dir, "metrics_distribution.png")
        plt.savefig(hist_path)
        plt.close()
        print(f"Saved distribution plot to {hist_path}")
        
        # Create sample frame comparison
        sample_indices = [0, len(self.original_frames)//4, len(self.original_frames)//2, 
                         3*len(self.original_frames)//4, len(self.original_frames)-1]
        
        for i, idx in enumerate(sample_indices):
            fig = self.create_side_by_side_image(idx)
            sample_path = os.path.join(output_dir, f"sample_frame_{idx}.png")
            fig.savefig(sample_path)
            plt.close(fig)
            print(f"Saved sample frame {idx} to {sample_path}")
        
        # Create summary table
        summary = {
            'Metric': ['Average PSNR (dB)', 'Average SSIM', 'Min PSNR (dB)', 'Min SSIM',
                     'Max PSNR (dB)', 'Max SSIM', 'Compression Ratio'],
            'Autoencoder': [
                np.mean(self.metrics['psnr_ae']),
                np.mean(self.metrics['ssim_ae']),
                np.min(self.metrics['psnr_ae']),
                np.min(self.metrics['ssim_ae']),
                np.max(self.metrics['psnr_ae']),
                np.max(self.metrics['ssim_ae']),
                self.metrics.get('ae_ratio', 'N/A')
            ],
            'H.264': [
                np.mean(self.metrics['psnr_h264']),
                np.mean(self.metrics['ssim_h264']),
                np.min(self.metrics['psnr_h264']),
                np.min(self.metrics['ssim_h264']),
                np.max(self.metrics['psnr_h264']),
                np.max(self.metrics['ssim_h264']),
                self.metrics.get('h264_ratio', 'N/A')
            ]
        }
        
        summary_df = pd.DataFrame(summary)
        summary_csv_path = os.path.join(output_dir, "summary.csv")
        summary_df.to_csv(summary_csv_path, index=False)
        print(f"Saved summary to {summary_csv_path}")
        
        # Generate HTML report
        self.generate_html_report(metrics_df, summary_df, output_dir)
    
    def generate_html_report(self, metrics_df, summary_df, output_dir):
        """Generate an HTML report with all results"""
        html_path = os.path.join(output_dir, "report.html")
        
        with open(html_path, 'w') as f:
            f.write("""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Video Compression Comparison Report</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 0; padding: 20px; }
                    .container { max-width: 1200px; margin: 0 auto; }
                    h1, h2 { color: #333; }
                    .summary-table { width: 100%; border-collapse: collapse; margin-bottom: 20px; }
                    .summary-table th, .summary-table td { 
                        padding: 12px; text-align: left; border-bottom: 1px solid #ddd; 
                    }
                    .summary-table th { background-color: #f2f2f2; }
                    .metrics-img { max-width: 100%; height: auto; margin-bottom: 20px; }
                    .sample-frame { margin-bottom: 30px; }
                    .two-column { display: flex; }
                    .column { flex: 1; padding: 10px; }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>Video Compression Comparison Report</h1>
                    
                    <h2>Summary</h2>
                    <table class="summary-table">
                        <tr>
                            <th>Metric</th>
                            <th>Autoencoder</th>
                            <th>H.264</th>
                        </tr>
            """)
            
            # Add summary table
            for _, row in summary_df.iterrows():
                f.write(f"""
                        <tr>
                            <td>{row['Metric']}</td>
                            <td>{row['Autoencoder'] if isinstance(row['Autoencoder'], str) else f"{row['Autoencoder']:.4f}"}</td>
                            <td>{row['H.264'] if isinstance(row['H.264'], str) else f"{row['H.264']:.4f}"}</td>
                        </tr>
                """)
            
            f.write("""
                    </table>
                    
                    <h2>Metrics Over Time</h2>
                    <img src="metrics_over_time.png" alt="Metrics Over Time" class="metrics-img">
                    
                    <h2>Metrics Distribution</h2>
                    <img src="metrics_distribution.png" alt="Metrics Distribution" class="metrics-img">
                    
                    <h2>Sample Frame Comparisons</h2>
            """)
            
            # Add sample frames
            sample_indices = [0, len(self.original_frames)//4, len(self.original_frames)//2, 
                             3*len(self.original_frames)//4, len(self.original_frames)-1]
            
            for idx in sample_indices:
                f.write(f"""
                    <div class="sample-frame">
                        <h3>Frame {idx}</h3>
                        <img src="sample_frame_{idx}.png" alt="Sample Frame {idx}" class="metrics-img">
                    </div>
                """)
            
            f.write("""
                </div>
            </body>
            </html>
            """)
        
        print(f"Generated HTML report at {html_path}")


def prepare_data_for_visualization(model_path, frames_dir, h264_crf=23):
    """
    Prepare data for visualization
    
    Args:
        model_path: Path to the trained autoencoder model
        frames_dir: Directory containing original frames
        h264_crf: Constant Rate Factor for H.264 compression
        
    Returns:
        tuple: (original_frames, ae_frames, h264_frames, metrics)
    """
    # Parameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_size = 256  # Target size for all frames
    
    # Load model
    model = VideoAutoencoder(latent_dim=64, num_bits=8)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model = model.to(device)  # Ensure model is on the correct device
    model.eval()
    
    # Load original frames
    frame_files = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) 
                          if f.endswith(('.jpg', '.png'))])
    
    original_frames = []
    original_frames_full_res = []  # Keep original resolution for H.264
    
    for frame_file in frame_files:
        frame = cv2.imread(frame_file)
        if frame is not None:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            original_frames_full_res.append(frame_rgb)
            
            # Resize for consistent comparison
            resized = cv2.resize(frame_rgb, (img_size, img_size))
            original_frames.append(resized)
    
    print(f"Loaded {len(original_frames)} original frames")
    
    # Process frames with autoencoder
    # Convert frames to PyTorch tensors
    original_tensors = []
    for frame in original_frames:  # Already resized to img_size
        # Convert to tensor
        tensor = torch.from_numpy(frame).float().permute(2, 0, 1) / 255.0
        original_tensors.append(tensor)
    
    batch_size = 16
    ae_frames = []
    
    print("Processing frames with autoencoder...")
    with torch.no_grad():
        for i in tqdm(range(0, len(original_tensors), batch_size)):
            batch = original_tensors[i:i+batch_size]
            batch = torch.stack(batch).to(device)
            reconstructed, _ = model(batch)
            # Move reconstructed frames back to CPU before adding to the list
            ae_frames.extend(reconstructed.cpu())
    
    # Process frames with H.264
    print("Processing frames with H.264...")
    
    # First resize to the same dimensions used by autoencoder
    resized_frames = [cv2.resize(frame, (img_size, img_size)) for frame in original_frames_full_res]
    
    h264_frames, h264_ratio, _ = compress_with_h264(resized_frames, crf=h264_crf)
    
    # Ensure H.264 frames are RGB (they might be BGR from OpenCV)
    h264_frames_rgb = []
    for frame in h264_frames:
        if frame.shape[2] == 3:  # If it has a color channel
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h264_frames_rgb.append(frame_rgb)
    
    h264_frames = h264_frames_rgb
    
    # Make sure all frames have the same size
    if h264_frames and h264_frames[0].shape[:2] != (img_size, img_size):
        h264_frames = [cv2.resize(frame, (img_size, img_size)) for frame in h264_frames]
    
    # Calculate compression ratio for autoencoder
    # Using model's calculate_compression_ratio method
    ae_ratio = model.get_compression_ratio(original_tensors[0].unsqueeze(0).shape)
    
    # Store metrics
    metrics = {
        'ae_ratio': ae_ratio,
        'h264_ratio': h264_ratio
    }
    
    return original_tensors, ae_frames, h264_frames, metrics


def main():
    """Run the visualization pipeline"""
    # Parameters
    frames_dir = "extracted_frames"  # From stage 1
    model_path = "video_autoencoder.pth"  # From stage 2
    results_dir = "visualization_results"
    
    try:
        print("Starting results visualization...")
        
        # Create output directory
        os.makedirs(results_dir, exist_ok=True)
        
        # Add a helpful error message if the model file doesn't exist
        if not os.path.exists(model_path):
            print(f"Error: Model file {model_path} not found. Make sure you've trained the model in stage 2.")
            sys.exit(1)
            
        # Check if frames directory exists
        if not os.path.exists(frames_dir) or not os.path.isdir(frames_dir):
            print(f"Error: Frames directory {frames_dir} not found. Make sure you've extracted frames in stage 1.")
            sys.exit(1)
        
        # Prepare data
        original_frames, ae_frames, h264_frames, metrics = prepare_data_for_visualization(
            model_path, frames_dir)
        
        # Create visualizer
        visualizer = VideoComparisonVisualizer(
            original_frames, ae_frames, h264_frames, metrics)
        
        # Generate report
        visualizer.generate_summary_report(output_dir=results_dir)
        
        # Create video comparison (use subset of frames for speed)
        total_frames = len(original_frames)
        frames_to_include = range(0, total_frames, 5)  # Every 5th frame
        visualizer.save_comparison_video(
            os.path.join(results_dir, "video_comparison.mp4"),
            fps=10,
            frames_to_include=frames_to_include
        )
        
        print(f"Results visualization completed. Check the '{results_dir}' directory for outputs.")
        
    except Exception as e:
        import traceback
        print(f"Error during visualization: {e}")
        print(traceback.format_exc())


if __name__ == "__main__":
    main()