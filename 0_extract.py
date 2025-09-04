import os
import subprocess
import sys

try:
    subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
except (subprocess.CalledProcessError, FileNotFoundError):
    print(f"Missing dependencies: sudo apt-get install ffmpeg")
    sys.exit(1)

video_path = sys.argv[1]
output_folder = sys.argv[2]
threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 0.25

os.makedirs(output_folder, exist_ok=True)

frame_files = []

try:
    cmd = ["ffmpeg", "-i", video_path,
        "-vf", f"select='gt(scene,{threshold})',setpts=N/FRAME_RATE/TB",
        "-vsync", "vfr", "-frame_pts", "1", "-q:v", "2",
        os.path.join(output_folder, "slide_%04d.png")
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)

    if result.returncode != 0:
        print(result.stderr)

    frame_files = [f for f in os.listdir(output_folder)
                 if f.startswith("slide_") and f.lower().endswith('.png')]

    if not frame_files:
        print(f"Warning: No frames extracted. Try running with a lower threshold (now {threshold}).")
        sys.exit(1)

    print(f"\nSuccessfully extracted {len(frame_files)} frames to {output_folder}")

except Exception as e:
    print(f"Error extracting frames: {str(e)}")
    sys.exit(1)