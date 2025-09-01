import os
import subprocess
import sys
import glob
import tempfile

missing = []
for cmd, desc in [
    (["tesseract", "--version"], "tesseract-ocr")
]:
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        missing.append(desc)

if missing:
    print(f"Missing dependencies: sudo apt-get install {' '.join(missing)}")
    sys.exit(1)

image_dir = sys.argv[1]
output_file = sys.argv[2]

images = sorted(glob.glob(os.path.join(image_dir, "*.png")))
if not images:
    print("No PNG images found")
    sys.exit(1)

with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as list_file:
    for img in images:
        list_file.write(f"{img}\n")
    list_filename = list_file.name

try:
    subprocess.run(["tesseract", list_filename, os.path.splitext(output_file)[0], "pdf"], check=True)
    print(f"SUCCESS: OCR PDF saved at {output_file}")
    
finally:
    os.unlink(list_filename)