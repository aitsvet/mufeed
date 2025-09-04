import os
import subprocess
import sys
import glob
import tempfile

missing = []
for cmd, desc in [
    ("convert", "imagemagick"),
    ("pdfunite", "poppler-utils"),
    ("tesseract", "tesseract-ocr"),
    ("hocr-pdf", "hocr-tools")
]:
    try:
        subprocess.run([cmd, "--version"], check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        missing.append(desc)

if missing:
    print(f"Missing dependencies: sudo apt-get install {', '.join(missing)}")
    sys.exit(1)

image_dir = sys.argv[1]
images = sorted(glob.glob(os.path.join(image_dir, "*.png")))
if not images:
    print("No images found")
    sys.exit(1)

output_file = sys.argv[2]

with tempfile.TemporaryDirectory() as tempdir:
    pdf_files = []
    for img in images:
        pdf_path = os.path.join(tempdir, os.path.basename(img) + ".pdf")
        try:
            subprocess.run(["convert", img, pdf_path], check=True)
            pdf_files.append(pdf_path)
        except subprocess.CalledProcessError as e:
            print(f"Error converting {img}: {e}")
            sys.exit(1)
    if not pdf_files:
        sys.exit(1)
    try:
        subprocess.run(["pdfunite"] + pdf_files + [output_file], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error combining PDFs: {e}")
        sys.exit(1)

base = os.path.splitext(output_file)[0]
temp_ocr_file = output_file + "_ocr.pdf"
try:
    subprocess.run(["hocr-pdf", output_file, temp_ocr_file], check=True)
    os.replace(temp_ocr_file, output_file)
except:
    try:
        subprocess.run(["tesseract", output_file, temp_ocr_file, "pdf"], check=True)
        os.replace(temp_ocr_file, output_file)
    except subprocess.CalledProcessError as e:
        print(f"OCR failed: {e}")
        sys.exit(1)

print(f"\nSUCCESS: Final PDF saved at {output_file}")
