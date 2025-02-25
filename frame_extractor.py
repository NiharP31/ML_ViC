import cv2
import os
import numpy as np
from pathlib import Path

def extract_frames(video_path, output_dir=None, interval=1, max_frames=None):
    """
    Extract frames from a video file using OpenCV.
    
    Args:
        video_path (str): Path to the video file
        output_dir (str, optional): Directory to save extracted frames. If None, frames are not saved to disk.
        interval (int, optional): Extract every nth frame. Default is 1 (extract all frames).
        max_frames (int, optional): Maximum number of frames to extract. Default is None (extract all).
        
    Returns:
        list: List of numpy arrays containing the extracted frames if output_dir is None
              Otherwise, returns the number of frames extracted
    """
    # Validate input file
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Create output directory if needed
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Saving frames to {output_dir}")
    
    # Open the video file
    video = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not video.isOpened():
        raise ValueError(f"Error opening video file: {video_path}")
    
    # Get video properties
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    
    print(f"Video properties:")
    print(f"  - FPS: {fps:.2f}")
    print(f"  - Frame count: {frame_count}")
    print(f"  - Duration: {duration:.2f} seconds")
    
    # Initialize variables
    frames = [] if output_dir is None else None
    count = 0
    extracted = 0
    
    # Read frames from the video
    while True:
        success, frame = video.read()
        
        # Break the loop if we reach the end of the video
        if not success:
            break
        
        # Extract frame if it matches the interval
        if count % interval == 0:
            if output_dir is not None:
                # Save frame to disk
                frame_path = os.path.join(output_dir, f"frame_{extracted:06d}.jpg")
                cv2.imwrite(frame_path, frame)
            else:
                # Store frame in memory
                frames.append(frame)
            
            extracted += 1
            
            # Break if we've extracted the maximum number of frames
            if max_frames is not None and extracted >= max_frames:
                break
        
        count += 1
    
    # Release the video
    video.release()
    
    print(f"Extracted {extracted} frames from {frame_count} total frames")
    
    # Return frames list or count
    return frames if output_dir is None else extracted


def main():
    """
    Example usage of the extract_frames function.
    """
    # Replace with your video file path
    video_path = "input.mp4"
    output_dir = "extracted_frames"
    
    try:
        # Extract all frames and save to disk
        num_frames = extract_frames(video_path, output_dir)
        print(f"Successfully extracted {num_frames} frames")
        
        # Example: Extract only 10 frames, every 5th frame, and keep in memory
        frames = extract_frames(video_path, output_dir=None, interval=5, max_frames=10)
        print(f"Extracted {len(frames)} frames into memory")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()