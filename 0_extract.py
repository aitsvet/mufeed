import os
import subprocess
import sys
from pathlib import Path
import argparse

def check_ffmpeg():
    """Check if ffmpeg is available"""
    try:
        subprocess.run(["ffmpeg", "-version"],
                        capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def extract_video_frames(video_path, output_folder, threshold=0.35):
    """Extract frames from video using scene detection"""
    os.makedirs(output_folder, exist_ok=True)

    print(f"\n" + "="*50)
    print(f"VIDEO FRAME EXTRACTION")
    print("="*50)
    print(f"Processing video: {os.path.basename(video_path)}")
    print(f"Scene change threshold: {threshold} (higher = less sensitive)")

    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-vf", f"select='gt(scene,{threshold})',setpts=N/FRAME_RATE/TB",
        "-vsync", "vfr",
        "-frame_pts", "1",
        "-q:v", "2",
        os.path.join(output_folder, "slide_%04d.png")
    ]

    try:
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True,
            check=False
        )

        if result.returncode != 0:
            print(f"FFmpeg completed with warnings:")
            for line in result.stderr.split('\n'):
                if "Error" in line or "failed" in line or "Invalid" in line:
                    print(f"  {line}")

        frame_files = [f for f in os.listdir(output_folder) 
                      if f.startswith("slide_") and f.lower().endswith('.png')]

        if not frame_files:
            print(f"\nWarning: No frames extracted. Possible reasons:")
            print(f"  - Video may not contain clear slide transitions")
            print(f"  - Threshold ({threshold}) may be too high")
            print("Try rerunning with a lower threshold (e.g., --video-threshold 0.25)")
            return False

        print(f"\nSuccessfully extracted {len(frame_files)} frames to {output_folder}")
        return True
    except Exception as e:
        print(f"Error extracting frames: {str(e)}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract video frames for slide processing')
    parser.add_argument('--video', type=str, required=True, help='Video file to extract slides from')
    parser.add_argument('--output', type=str, required=True, help='Folder to save extracted frames')
    parser.add_argument('--threshold', type=float, default=0.35, 
                        help='Scene change threshold (0-1, higher = less sensitive)')

    args = parser.parse_args()

    if not check_ffmpeg():
        print("ERROR: ffmpeg is required for video processing but not found.")
        print("Please install ffmpeg before using this script.")
        sys.exit(1)

    extract_video_frames(args.video, args.output, args.threshold)